"""Value Score: how much value does taking him *at this point* capture?

Distinct from Player Score, which asks how good he is. A player can be excellent and
still a poor pick right now — if he will still be there in twelve picks, and the tier
behind another position is about to collapse.

===================== ====== ==========================================================
Component             Weight What it contributes
===================== ====== ==========================================================
market_value          0.40   ADP versus the pick on the clock (or versus our own board,
                             when no draft is live)
tier_cliff            0.25   Size of the drop to the next tier at his position
scarcity              0.20   Positional supply against remaining league demand
projection_vs_market  0.15   Where our VBD ordering disagrees with the consensus
===================== ====== ==========================================================
"""

from __future__ import annotations

import polars as pl

from ..config import AppConfig
from ..models import ComponentScore, ScoreBundle
from .compose import component, compose
from .player_score import weighted_blend

VALUE_COMPONENTS: dict[str, tuple[str, str]] = {
    "market_value": ("market_value_score", "market_confidence"),
    "tier_cliff": ("tier_cliff_score", "tier_confidence"),
    "scarcity": ("scarcity_score", "scarcity_confidence"),
    "projection_vs_market": ("projection_vs_market_score", "market_confidence"),
}


def add_value_score(cfg: AppConfig, board: pl.DataFrame) -> pl.DataFrame:
    """Add ``value_score`` and ``value_score_confidence`` to the board."""
    if board.is_empty():
        return board
    return weighted_blend(
        board,
        VALUE_COMPONENTS,
        cfg.weights.value_score.as_dict(),
        "value_score",
        "value_score_confidence",
    )


def value_score_bundle(cfg: AppConfig, row: dict) -> ScoreBundle:
    """Rebuild one player's Value Score as an explainable bundle."""
    raw_sources: dict[str, tuple[str, str, str]] = {
        "market_value": ("adp_delta", "picks of discount, scaled by observed ADP spread",
                         "FantasyPros ECR"),
        "tier_cliff": ("points_to_next_tier", "points to next tier vs position spread",
                       "derived"),
        "scarcity": ("supply_ratio", "startable supply vs remaining starter demand",
                     "derived"),
        "projection_vs_market": ("market_disagreement", "ADP minus our VBD rank",
                                 "FantasyPros ECR"),
    }
    components: dict[str, ComponentScore] = {}
    for key, (score_col, confidence_col) in VALUE_COMPONENTS.items():
        raw_col, method, source = raw_sources[key]
        components[key] = component(
            name=key,
            raw_value=row.get(raw_col),
            normalized=row.get(score_col),
            confidence=row.get(confidence_col) if row.get(confidence_col) is not None else 1.0,
            method=method,
            source=source,
        )
    return compose("value_score", components, cfg.weights.value_score.as_dict())
