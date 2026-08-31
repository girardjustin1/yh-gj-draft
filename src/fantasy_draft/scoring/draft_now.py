"""Draft Now Score: what should I select at *this* pick?

The score the whole system exists to produce. Player Score asks how good he is; Value
Score asks what taking him here captures; this asks what to actually do.

======================= ====== =========================================================
Component               Weight What it contributes
======================= ====== =========================================================
player_score            0.35   Season-long quality
value_score             0.20   Value captured at this pick
next_pick_urgency       0.15   Probability he is gone before our next turn
tier_scarcity           0.10   Cliff behind him, and how thin his position is
roster_fit              0.075  Lineup need — a modifier, never a veto
draft_room              0.075  Position runs and the value they push toward us
strategy_fit            0.05   Coherence with our emerging roster construction
======================= ====== =========================================================

Urgency is what makes this different from a board. A player with a slightly lower Player
Score who will certainly be gone can and should outrank one who will still be sitting
there in twelve picks.
"""

from __future__ import annotations

import polars as pl

from ..config import AppConfig
from ..models import ComponentScore, ScoreBundle
from .compose import component, compose
from .player_score import weighted_blend

DRAFT_NOW_COMPONENTS: dict[str, tuple[str, str]] = {
    "player_score": ("player_score", "player_score_confidence"),
    "value_score": ("value_score", "value_score_confidence"),
    "next_pick_urgency": ("next_pick_urgency_score", "survival_confidence"),
    "tier_scarcity": ("tier_scarcity_score", "scarcity_confidence"),
    "roster_fit": ("roster_fit_score", "roster_fit_confidence"),
    "draft_room": ("draft_room_score", "draft_room_confidence"),
    "strategy_fit": ("strategy_fit_score", "strategy_confidence"),
}


def add_tier_scarcity(
    board: pl.DataFrame, expected_losses: dict[str, float] | None = None
) -> pl.DataFrame:
    """Blend what waiting costs this player with how thin his position is.

    Two different facts that both mean "do not wait": the points we lose by taking the
    next-best player at his position instead, and the overall supply of startable players
    there. When ``expected_losses`` is supplied (the number of each position expected to
    go before our next pick), the first is recomputed against that real demand rather
    than the board's flat default.
    """
    if board.is_empty():
        return board
    if expected_losses:
        from ..analytics.tiers import expected_loss_by_waiting

        board = expected_loss_by_waiting(board, expected_losses)
    return board.with_columns(
        (
            0.6 * pl.col("tier_cliff_score").fill_null(0.0)
            + 0.4 * pl.col("scarcity_score").fill_null(50.0)
        ).clip(0, 100).alias("tier_scarcity_score")
    )


def add_draft_room_score(
    board: pl.DataFrame, demand: dict[str, float], value_created: dict[str, float]
) -> pl.DataFrame:
    """Score the room's behaviour per position.

    Demand and value-created pull in opposite directions on purpose. A position the room
    is hammering is draining (raises the score), but a position the room is ignoring is
    getting cheap (also raises the score). Chasing only the first is how a draft room
    talks itself into a bad pick.
    """
    if board.is_empty():
        return board
    combined = {
        position: 0.5 * demand.get(position, 50.0) + 0.5 * value_created.get(position, 50.0)
        for position in set(demand) | set(value_created)
    }
    return board.with_columns(
        pl.col("position").replace_strict(combined, default=50.0).alias("draft_room_score"),
        pl.lit(0.7).alias("draft_room_confidence"),
    )


def add_draft_now_score(cfg: AppConfig, board: pl.DataFrame) -> pl.DataFrame:
    """Add ``draft_now_score`` and its confidence, then rank the board by it."""
    if board.is_empty():
        return board
    board = weighted_blend(
        board,
        DRAFT_NOW_COMPONENTS,
        cfg.weights.draft_now.as_dict(),
        "draft_now_score",
        "draft_now_confidence",
    )
    return board.sort("draft_now_score", descending=True, nulls_last=True).with_columns(
        pl.int_range(1, pl.len() + 1).cast(pl.Int32).alias("draft_now_rank")
    )


def draft_now_bundle(cfg: AppConfig, row: dict) -> ScoreBundle:
    """Rebuild one player's Draft Now Score as an explainable bundle."""
    raw_sources: dict[str, tuple[str, str]] = {
        "player_score": ("projected_points", "weighted composite of 6 season-long signals"),
        "value_score": ("adp_delta", "weighted composite of 4 market signals"),
        "next_pick_urgency": ("probability_gone", "roster-aware survival to our next pick"),
        "tier_scarcity": ("points_to_next_tier", "tier cliff blended with positional supply"),
        "roster_fit": ("roster_fit_score", "unfilled starting slots, as a modifier"),
        "draft_room": ("draft_room_score", "position runs and the value they create"),
        "strategy_fit": ("strategy_fit_score", "coherence with our emerging construction"),
    }
    components: dict[str, ComponentScore] = {}
    for key, (score_col, confidence_col) in DRAFT_NOW_COMPONENTS.items():
        raw_col, method = raw_sources[key]
        components[key] = component(
            name=key,
            raw_value=row.get(raw_col),
            normalized=row.get(score_col),
            confidence=row.get(confidence_col) if row.get(confidence_col) is not None else 1.0,
            method=method,
            source="derived",
        )
    return compose("draft_now_score", components, cfg.weights.draft_now.as_dict())
