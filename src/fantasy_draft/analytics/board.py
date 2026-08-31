"""Board assembly: the single place where every signal comes together.

``build_board`` is the engine's main read path. It is used by ``ff board``,
``ff players``, ``ff compare``, ``ff explain``, and — with a live ``DraftState`` — by
``ff on-clock``. Building it once and passing it around keeps every command answering
from the same numbers.

Failure isolation is enforced here rather than in each signal module: any signal that
raises is caught, recorded as a warning, and its component simply arrives with zero
confidence, so the composition step redistributes its weight. Losing schedule data
costs us 7.5% of the Player Score and an honest warning, not a recommendation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from ..config import AppConfig
from ..constants import OFFENSE_POSITIONS
from ..database import Database
from ..logging import get_logger
from ..scoring.player_score import add_player_score
from ..scoring.value_score import add_value_score
from .market import adp_table, market_signals
from .offense import team_offense_scores
from .opportunity import opportunity_score
from .projections import consensus_projections
from .replacement import (
    ReplacementLevel,  # noqa: F401  (re-exported for Board)
    replacement_levels,
)
from .risk import risk_score
from .scarcity import PositionScarcity, position_scarcity, scarcity_frame
from .tiers import assign_tiers, expected_loss_by_waiting
from .vbd import add_vbd

log = get_logger(__name__)

#: Positions the projection model actually covers. Kickers and defences are excluded on
#: purpose — see ``Board.warnings`` and docs/scoring.md.
MODELLED_POSITIONS = OFFENSE_POSITIONS


@dataclass(slots=True)
class Board:
    """A fully scored draft board plus everything needed to explain and caveat it."""

    frame: pl.DataFrame
    replacement: dict[str, ReplacementLevel]
    scarcity: dict[str, PositionScarcity]
    warnings: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    current_pick: int | None = None

    @property
    def confidence(self) -> float:
        """Mean Player Score confidence over the top of the board."""
        if self.frame.is_empty() or "player_score_confidence" not in self.frame.columns:
            return 0.0
        top = self.frame.head(60)["player_score_confidence"].drop_nulls()
        return float(top.mean()) if len(top) else 0.0

    def available(self, drafted: set[str] | None = None) -> pl.DataFrame:
        if not drafted:
            return self.frame
        return self.frame.filter(~pl.col("player_key").is_in(list(drafted)))

    def row(self, player_key: str) -> dict | None:
        match = self.frame.filter(pl.col("player_key") == player_key)
        return match.to_dicts()[0] if match.height else None

    def top(self, limit: int = 25, position: str | None = None) -> pl.DataFrame:
        frame = self.frame
        if position:
            frame = frame.filter(pl.col("position") == position)
        return frame.head(limit)


def _try(
    warnings: list[str], label: str, fn: Callable[[], pl.DataFrame]
) -> pl.DataFrame | None:
    """Run a signal, converting failure into a warning instead of an exception."""
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 — isolation is the point
        message = f"{label} unavailable ({type(exc).__name__}: {exc})"
        log.warning("signal failed", extra={"signal": label, "error": str(exc)})
        warnings.append(message)
        return None
    if result is None or result.is_empty():
        warnings.append(f"{label} returned no rows; its weight was redistributed")
        return None
    return result


def build_board(
    db: Database,
    cfg: AppConfig,
    drafted: set[str] | None = None,
    current_pick: int | None = None,
    picks_until_next: int | None = None,
    next_pick: int | None = None,
) -> Board:
    """Assemble the scored board.

    ``drafted`` removes already-selected players before scarcity and scoring, so the
    board a live draft sees is the board of who is actually left.
    """
    warnings: list[str] = []

    projections = consensus_projections(db, cfg)
    if projections.is_empty():
        warnings.append(
            "No projections available. Run `ff data refresh`, then `ff board --rebuild`."
        )
        return Board(pl.DataFrame(), {}, {}, warnings, current_pick=current_pick)

    projections = projections.filter(pl.col("position").is_in(list(MODELLED_POSITIONS)))

    # Replacement level must be derived from the FULL player universe, before removing
    # drafted players.
    #
    # Replacement level answers "how good is a freely available player at this position",
    # which is a property of the league's structure and the player pool -- not of who
    # happens to be left right now. Computing it on the shrinking available pool makes
    # the baseline drift down as the draft proceeds, inflating every remaining player's
    # VBD, and inflating it *unevenly* by position (mid-draft this was worth +37 points
    # of phantom VBD to receivers against +7 to tight ends). That corrupts exactly the
    # cross-positional comparison VBD exists to make.
    levels = replacement_levels(cfg, projections, positions=MODELLED_POSITIONS)

    if drafted:
        projections = projections.filter(~pl.col("player_key").is_in(list(drafted)))

    unmodelled = [p for p in cfg.positions if p not in MODELLED_POSITIONS]
    if unmodelled:
        warnings.append(
            f"{', '.join(unmodelled)} are not modelled: nflverse carries no team-defence "
            "scoring and we do not ingest kicking stats, so no honest projection exists "
            "for them. Draft them by consensus rank in the last rounds — their value "
            "over replacement is near zero either way."
        )

    # --- value over replacement, then tiers on top of it ---
    # Tiers, unlike replacement, *are* computed on what is available: mid-draft the
    # useful question is how far the drop is to the next group of players we could
    # actually still take.
    board = add_vbd(projections, levels)
    board = assign_tiers(cfg, board)
    # A first pass at "what does waiting cost", using a flat expectation. The
    # recommendation engine recomputes this with the real per-position demand once the
    # survival model has run.
    flat_slide = max(1.0, (picks_until_next or 12) / 4.0)
    board = expected_loss_by_waiting(board, {}, default_slide=flat_slide)

    # --- market ---
    adp = _try(warnings, "ADP/consensus rank", lambda: adp_table(db, cfg))
    if adp is not None:
        board = market_signals(
            board, adp, current_pick=current_pick,
            picks_until_next=picks_until_next, next_pick=next_pick,
        )
    else:
        board = board.with_columns(
            pl.lit(50.0).alias("market_value_score"),
            pl.lit(50.0).alias("projection_vs_market_score"),
            pl.lit(0.0).alias("market_confidence"),
            pl.lit(None, dtype=pl.Float64).alias("adp"),
            pl.lit(None, dtype=pl.Float64).alias("adp_delta"),
            pl.lit(None, dtype=pl.Float64).alias("market_disagreement"),
        )

    # --- opportunity ---
    opportunity = _try(warnings, "Opportunity", lambda: opportunity_score(db, cfg))
    if opportunity is not None:
        board = board.join(opportunity.drop("position"), on="player_key", how="left")
    else:
        board = board.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("opportunity_score"),
            pl.lit(0.0).alias("opportunity_confidence"),
        )
    board = board.with_columns(pl.col("opportunity_confidence").fill_null(0.0))

    # --- offensive environment ---
    offense = _try(warnings, "Offensive environment", lambda: team_offense_scores(db, cfg))
    if offense is not None:
        board = board.join(
            offense.rename({"confidence": "offense_confidence"}), on="team", how="left"
        )
    else:
        board = board.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("offense_score"),
            pl.lit(0.0).alias("offense_confidence"),
        )
    board = board.with_columns(pl.col("offense_confidence").fill_null(0.0))

    # --- schedule (Phase 6; absent for now, and honestly reported as such) ---
    if "schedule_score" not in board.columns:
        board = board.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("schedule_score"),
            pl.lit(0.0).alias("schedule_confidence"),
            pl.lit(None, dtype=pl.Float64).alias("schedule_raw"),
        )
        warnings.append(
            "Schedule strength is not implemented yet (Phase 6); its 7.5% weight is "
            "redistributed across the components we do have."
        )

    # --- bye weeks ---
    byes = db.query("SELECT team, bye_week FROM byes WHERE season = ?", [cfg.league.season])
    if not byes.is_empty():
        board = board.join(byes, on="team", how="left")
    else:
        board = board.with_columns(pl.lit(None, dtype=pl.Int64).alias("bye_week"))
        warnings.append("No bye weeks derived for this season.")

    # --- ECR spread, needed by risk ---
    ecr = db.query(
        "SELECT player_key, ecr AS overall_ecr, best AS ecr_best, worst AS ecr_worst "
        "FROM rankings WHERE season = ? AND ranking_type = 'redraft-overall'",
        [cfg.league.season],
    )
    board = board.join(ecr, on="player_key", how="left")

    risk = _try(warnings, "Risk", lambda: risk_score(db, cfg, board))
    if risk is not None:
        board = board.join(risk, on="player_key", how="left")
    else:
        board = board.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("risk_score"),
            pl.lit(0.0).alias("risk_confidence"),
        )
    board = board.with_columns(
        pl.col("risk_confidence").fill_null(0.0),
        # Risk enters the Player Score inverted: safe players score high.
        (100.0 - pl.col("risk_score")).alias("risk_inverted_score"),
    )

    # --- scarcity, computed on what is actually left ---
    drafted_by_position: dict[str, int] = {}
    if drafted:
        counts = db.query(
            "SELECT position, count(*) AS n FROM players WHERE player_key IN "
            f"({', '.join('?' for _ in drafted)}) GROUP BY position",
            list(drafted),
        )
        drafted_by_position = dict(zip(counts["position"], counts["n"], strict=True))

    scarcity = position_scarcity(
        cfg, board, levels, drafted_by_position, positions=MODELLED_POSITIONS
    )
    board = board.join(scarcity_frame(scarcity), on="position", how="left").with_columns(
        pl.lit(0.8).alias("scarcity_confidence"),
        pl.lit(1.0).alias("tier_confidence"),
    )

    # --- normalize projection and VBD onto the 0-100 scale ---
    board = board.with_columns(
        (
            100.0
            / (
                1.0
                + (
                    -(
                        (pl.col("projected_points") - pl.col("projected_points").mean().over("position"))
                        / (pl.col("projected_points").std().over("position") + 1e-9)
                    )
                    / 1.5
                ).exp()
            )
        ).alias("projection_score"),
        (
            100.0 * pl.col("vbd") / (pl.col("vbd").max() + 1e-9)
        ).clip(0, 100).alias("vbd_score"),
        pl.lit(1.0).alias("vbd_confidence"),
    )
    if "projection_confidence" not in board.columns:
        board = board.with_columns(pl.lit(0.85).alias("projection_confidence"))

    # --- compose ---
    board = add_player_score(cfg, board)
    board = add_value_score(cfg, board)

    board = board.sort("player_score", descending=True, nulls_last=True).with_columns(
        pl.int_range(1, pl.len() + 1).cast(pl.Int32).alias("board_rank")
    )

    return Board(
        frame=board,
        replacement=levels,
        scarcity=scarcity,
        warnings=warnings,
        current_pick=current_pick,
    )
