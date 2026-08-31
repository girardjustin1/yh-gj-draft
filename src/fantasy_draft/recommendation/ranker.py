"""The recommendation engine.

One entry point, :func:`recommend`, which takes a live ``DraftState`` and returns a
fully-explained :class:`~fantasy_draft.models.Recommendation`. This is the function the
CLI calls, and the one the MCP tool layer will wrap — so ``ff on-clock`` and Claude
always answer from identical numbers.

The pipeline, in order, because each step depends on the last:

1. Score the board over the players who are actually left.
2. Estimate what the managers ahead of our next turn need.
3. Turn that into a survival probability per candidate.
4. Read the room: runs, demand, and the value those runs push toward us.
5. Infer our roster-construction drift.
6. Compose the Draft Now Score and rank.

Every step that can fail degrades into a warning and a zeroed confidence rather than an
exception. A recommendation with honest caveats beats no recommendation on the clock.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from ..analytics.board import Board, build_board
from ..analytics.draft_room import DraftRoomRead, read_room
from ..analytics.roster_fit import add_roster_fit
from ..config import AppConfig
from ..constants import OFFENSE_POSITIONS
from ..database import Database
from ..draft.availability import SurvivalResult, attach_survival, survival_probabilities
from ..draft.opponent_needs import opponent_needs
from ..draft.simulator import SimulationResult, blend_survival, simulate_to_next_pick
from ..draft.state import DraftState
from ..draft.strategies import StrategyState, classify
from ..logging import get_logger
from ..models import Candidate, DataFreshness, Recommendation
from ..scoring.draft_now import add_draft_now_score, add_draft_room_score, add_tier_scarcity
from ..scoring.player_score import player_score_bundle
from ..scoring.value_score import value_score_bundle
from .explanation import write_explanation

log = get_logger(__name__)

#: How many candidates get the full treatment. Beyond this the survival model is noise.
CANDIDATE_DEPTH = 40


class RecommendationResult:
    """The recommendation plus the intermediate objects the CLI wants to display."""

    def __init__(
        self,
        recommendation: Recommendation,
        board: Board,
        room: DraftRoomRead,
        strategy: StrategyState,
        survival: SurvivalResult,
        frame: pl.DataFrame,
        simulation: SimulationResult | None = None,
    ) -> None:
        self.recommendation = recommendation
        self.board = board
        self.room = room
        self.strategy = strategy
        self.survival = survival
        self.frame = frame
        self.simulation = simulation


def _freshness(db: Database, cfg: AppConfig, state: DraftState) -> list[DataFreshness]:
    """Staleness of everything the recommendation rests on."""
    rows: list[DataFreshness] = []
    for source in ("fantasypros_ecr", "nflverse_players", "nflverse_injuries",
                   "nflverse_ff_opportunity"):
        entry = db.last_refresh(source)
        spec = cfg.data_sources.spec(source)
        rows.append(
            DataFreshness(
                source=source,
                updated_at=entry["ingested_at"] if entry else None,
                rows=entry["rows"] if entry else None,
                max_age_hours=spec.max_age_hours,
                ok=entry is not None,
            )
        )
    rows.append(
        DataFreshness(
            source="draft_sync",
            updated_at=state.synced_at,
            rows=state.picks_made,
            max_age_hours=cfg.data_sources.spec("sleeper_draft").max_age_hours,
            ok=True,
        )
    )
    return rows


def _candidate(cfg: AppConfig, row: dict) -> Candidate:
    """Build a fully-populated Candidate from one board row."""
    from ..models import SurvivalEstimate
    from ..scoring.draft_now import draft_now_bundle

    survival = None
    if row.get("probability_available") is not None:
        available = float(row["probability_available"])
        survival = SurvivalEstimate(
            player_key=row["player_key"],
            probability_available=available,
            probability_gone=1.0 - available,
            method="blended",
            confidence=float(row.get("survival_confidence") or 0.0),
        )

    return Candidate(
        player_key=row["player_key"],
        name=row.get("player_name") or row["player_key"],
        position=row.get("position") or "",
        team=row.get("team"),
        bye_week=int(row["bye_week"]) if row.get("bye_week") is not None else None,
        projected_points=row.get("projected_points"),
        vbd=row.get("vbd"),
        adp=row.get("adp"),
        adp_sd=row.get("adp_sd"),
        positional_rank=row.get("positional_rank"),
        tier=row.get("tier"),
        tier_rank=row.get("tier_rank"),
        points_to_next_player=row.get("points_to_next_player"),
        points_to_next_tier=row.get("points_to_next_tier"),
        player_score=player_score_bundle(cfg, row),
        value_score=value_score_bundle(cfg, row),
        draft_now_score=draft_now_bundle(cfg, row),
        survival=survival,
    )


#: Two-pick EV must beat the Draft Now leader by at least this much to override it.
#: Simulation noise at a few thousand iterations is worth roughly a point, so a smaller
#: margin is not evidence of anything.
TWO_PICK_OVERRIDE_MARGIN = 1.5


def recommend(
    db: Database,
    cfg: AppConfig,
    state: DraftState,
    limit: int = 10,
    positions: tuple[str, ...] = OFFENSE_POSITIONS,
    simulate: bool = True,
    iterations: int | None = None,
) -> RecommendationResult:
    """Produce a ranked, explained recommendation for the pick on the clock."""
    warnings: list[str] = []
    target_pick = state.my_current_pick or state.current_pick

    # 1. Score only the players still on the board.
    board = build_board(
        db, cfg, drafted=state.drafted_keys, current_pick=target_pick,
        picks_until_next=state.picks_until_next, next_pick=state.my_next_pick,
    )
    warnings.extend(board.warnings)
    frame = board.frame
    if frame.is_empty():
        return RecommendationResult(
            Recommendation(
                pick_label=state.pick_label, overall_pick=target_pick,
                confidence=0.0, warnings=warnings or ["No players available to score."],
            ),
            board, DraftRoomRead(), StrategyState(),
            SurvivalResult({}, {}, 0, "none"), frame,
        )

    if state.unresolved_pick_count:
        warnings.append(
            f"{state.unresolved_pick_count} drafted player(s) could not be matched to our "
            "database. They are treated as off the board, but their positions are unknown "
            "to the draft-room read."
        )

    # 2-3. Who is picking before us, what do they need, and who survives it.
    needs = opponent_needs(state, cfg.league, frame, positions)
    if not needs and state.my_next_pick is None:
        warnings.append(
            "This is our last pick of the draft, so survival probability is not "
            "meaningful; urgency contributes nothing."
        )
    survival = survival_probabilities(state, cfg.league, frame, needs)
    frame = attach_survival(frame, survival, state.my_next_pick)

    # 4. What is the room doing, and what is it pushing toward us?
    room = read_room(state, frame, needs, positions=positions)
    frame = add_draft_room_score(frame, room.demand, room.value_created)

    # 5. Roster fit and strategy drift.
    roster = state.my_roster()
    rounds_remaining = max(0, state.board.rounds - state.current_round + 1)
    frame = add_roster_fit(frame, cfg.league, roster, rounds_remaining)

    strategy = classify(
        cfg.league, roster, cfg.weights.strategy_priors, state.current_round,
        rb_value_available=room.value_for("RB"),
        rb_run_intensity=room.run_intensity.get("RB", 0.0),
    )
    strategy_fit = {
        position: strategy.fit(position, roster, cfg.league) for position in positions
    }
    frame = frame.with_columns(
        pl.col("position").replace_strict(strategy_fit, default=50.0).alias("strategy_fit_score"),
        pl.lit(strategy.confidence).alias("strategy_confidence"),
    )

    # 6. Compose and rank.
    frame = add_tier_scarcity(frame, survival.expected_position_losses)
    frame = add_draft_now_score(cfg, frame)

    # 7. Simulate forward: cross-check survival, and price the *pair* of picks.
    simulation: SimulationResult | None = None
    if simulate and needs and not frame.is_empty():
        shortlist = frame.head(cfg.weights.simulation.two_pick_candidates)[
            "player_key"
        ].to_list()
        try:
            simulation = simulate_to_next_pick(
                cfg, state, frame, needs, candidates=shortlist, iterations=iterations
            )
        except Exception as exc:  # noqa: BLE001 — a failed simulation must not cost the pick
            log.warning("simulation failed", extra={"error": str(exc)})
            warnings.append(f"Simulation unavailable ({type(exc).__name__}); using the "
                            "analytic survival model alone.")

        if simulation is not None and simulation.survival:
            # The analytic model treats the intervening picks as independent; the
            # simulation captures their correlation. Averaging is more honest than
            # picking a favourite, and the gap between them is worth reporting.
            blended = blend_survival(
                {k: v.probability_available for k, v in survival.estimates.items()},
                simulation.survival,
            )
            frame = frame.with_columns(
                pl.col("player_key")
                .replace_strict(blended, default=None)
                .cast(pl.Float64)
                .alias("simulated_available")
            ).with_columns(
                pl.coalesce("simulated_available", "probability_available")
                .alias("probability_available")
            ).with_columns(
                (1.0 - pl.col("probability_available")).alias("probability_gone"),
                (100.0 * (1.0 - pl.col("probability_available")))
                .clip(0, 100)
                .alias("next_pick_urgency_score"),
                pl.lit(0.85).alias("survival_confidence"),
            )
            frame = add_draft_now_score(cfg, frame)

    top = frame.head(max(limit, CANDIDATE_DEPTH)).to_dicts()
    candidates = [_candidate(cfg, row) for row in top[:limit]]

    if simulation is not None:
        for candidate in candidates:
            value = simulation.two_pick.get(candidate.player_key)
            if value is not None:
                candidate.two_pick_expected_value = round(value.combined, 2)
                candidate.expected_next_pick_value = round(value.expected_next_value, 2)

    primary = candidates[0] if candidates else None

    # Two-pick expected value may overturn the ranking — that is its purpose. It only
    # does so on a margin larger than simulation noise, and the override is stated in
    # the explanation rather than applied silently.
    two_pick_override: str | None = None
    if primary is not None and primary.two_pick_expected_value is not None:
        better = max(
            (c for c in candidates if c.two_pick_expected_value is not None),
            key=lambda c: c.two_pick_expected_value,
        )
        margin = better.two_pick_expected_value - primary.two_pick_expected_value
        if better.player_key != primary.player_key and margin >= TWO_PICK_OVERRIDE_MARGIN:
            two_pick_override = (
                f"Draft Now rated {primary.name} highest on this pick alone, but across "
                f"both picks {better.name} is worth "
                f"{better.two_pick_expected_value:.1f} against "
                f"{primary.two_pick_expected_value:.1f} — a {margin:.1f}-point edge on "
                f"the pair, so he is the recommendation."
            )
            candidates.remove(better)
            candidates.insert(0, better)
            primary = better

    alternatives = candidates[1:3]

    staleness = _freshness(db, cfg, state)
    confidence = _overall_confidence(frame, board, strategy, staleness, warnings)

    recommendation = Recommendation(
        generated_at=datetime.now(),
        pick_label=state.board.label(target_pick),
        overall_pick=target_pick,
        next_pick_overall=state.my_next_pick,
        picks_until_next=state.picks_until_next,
        primary=primary,
        alternatives=alternatives,
        board=candidates,
        strategy=strategy.primary,
        strategy_confidence=strategy.confidence,
        alternative_strategy=strategy.alternative,
        strategy_reason=strategy.reason,
        position_demand=room.demand,
        confidence=confidence,
        warnings=warnings,
        staleness=staleness,
    )
    recommendation.explanation = write_explanation(
        recommendation, state, room, strategy, simulation, two_pick_override
    )

    return RecommendationResult(
        recommendation, board, room, strategy, survival, frame, simulation
    )


def _overall_confidence(
    frame: pl.DataFrame,
    board: Board,
    strategy: StrategyState,
    staleness: list[DataFreshness],
    warnings: list[str],
) -> float:
    """How much should this recommendation be trusted?

    Falls when components are missing, when data is stale, when the top candidates are
    bunched together (the choice genuinely is close), and when survival is uncertain.
    """
    top = frame.head(5)
    if top.is_empty():
        return 0.0

    base = float(top["draft_now_confidence"].fill_null(0.0).mean())

    # A clear gap between first and second is itself evidence.
    scores = top["draft_now_score"].to_list()
    separation = (scores[0] - scores[1]) if len(scores) > 1 else 5.0
    separation_factor = min(1.0, 0.55 + separation / 12.0)

    stale = sum(1 for row in staleness if row.is_stale)
    staleness_factor = max(0.55, 1.0 - 0.12 * stale)

    survival_known = float(top["survival_confidence"].fill_null(0.0).mean())
    survival_factor = 0.8 + 0.2 * survival_known

    warning_factor = max(0.7, 1.0 - 0.05 * len(warnings))

    return max(
        0.05,
        min(1.0, base * separation_factor * staleness_factor * survival_factor * warning_factor),
    )
