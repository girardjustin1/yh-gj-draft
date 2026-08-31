"""Player Score: how good is this player, ignoring where we are in the draft?

Six components, weighted in ``config/scoring_weights.yaml``:

===================== ====== ==========================================================
Component             Weight What it contributes
===================== ====== ==========================================================
projection            0.35   Consensus points, recomputed in *our* scoring rules
vbd                   0.25   Points above positional replacement — the cross-position
                             currency, and the reason six QBs do not top the board
opportunity           0.15   Role quality: snaps, targets, carries, expected points
offense_environment   0.10   Quality of the offence he plays in
schedule              0.075  Position-specific strength of schedule
risk                  0.075  Entered *inverted*: low risk raises the score
===================== ====== ==========================================================

Composition is confidence-weighted (see :mod:`fantasy_draft.scoring.compose`): a
component we cannot compute has its weight redistributed rather than contributing a
confident-looking 50.
"""

from __future__ import annotations

import polars as pl

from ..config import AppConfig
from ..models import ComponentScore, ScoreBundle
from .compose import component, compose

#: Board column holding each component's 0-100 score and its 0-1 confidence.
PLAYER_COMPONENTS: dict[str, tuple[str, str]] = {
    "projection": ("projection_score", "projection_confidence"),
    "vbd": ("vbd_score", "vbd_confidence"),
    "opportunity": ("opportunity_score", "opportunity_confidence"),
    "offense_environment": ("offense_score", "offense_confidence"),
    "schedule": ("schedule_score", "schedule_confidence"),
    "risk": ("risk_inverted_score", "risk_confidence"),
}


def weighted_blend(
    frame: pl.DataFrame,
    components: dict[str, tuple[str, str]],
    weights: dict[str, float],
    out_value: str,
    out_confidence: str,
) -> pl.DataFrame:
    """Confidence-weighted blend, vectorized across the whole board.

    Mirrors :func:`fantasy_draft.scoring.compose.compose` exactly, but in Polars so 800
    players cost one pass instead of 800 Python objects. The per-player object form is
    used only by ``ff explain``, where the derivation matters more than throughput.
    """
    present = {
        key: (score_col, confidence_col)
        for key, (score_col, confidence_col) in components.items()
        if score_col in frame.columns and weights.get(key, 0) > 0
    }
    if not present:
        return frame.with_columns(
            pl.lit(50.0).alias(out_value), pl.lit(0.0).alias(out_confidence)
        )

    numerator = pl.lit(0.0)
    denominator = pl.lit(0.0)
    covered = pl.lit(0.0)
    for key, (score_col, confidence_col) in present.items():
        weight = weights[key]
        confidence = (
            pl.col(confidence_col).fill_null(0.0).clip(0, 1)
            if confidence_col in frame.columns
            else pl.lit(1.0)
        )
        # A null score is unknown regardless of what the confidence column claims.
        confidence = pl.when(pl.col(score_col).is_null()).then(0.0).otherwise(confidence)
        score = pl.col(score_col).fill_null(50.0).clip(0, 100)

        numerator = numerator + score * weight * confidence
        denominator = denominator + weight * confidence
        covered = covered + weight * confidence

    intended = sum(weights[key] for key in present) or 1.0
    return frame.with_columns(
        pl.when(denominator > 0)
        .then(numerator / denominator)
        .otherwise(50.0)
        .clip(0, 100)
        .alias(out_value),
        (covered / intended).clip(0, 1).alias(out_confidence),
    )


def add_player_score(cfg: AppConfig, board: pl.DataFrame) -> pl.DataFrame:
    """Add ``player_score`` and ``player_score_confidence`` to the board."""
    if board.is_empty():
        return board
    return weighted_blend(
        board,
        PLAYER_COMPONENTS,
        cfg.weights.player_score.as_dict(),
        "player_score",
        "player_score_confidence",
    )


def player_score_bundle(cfg: AppConfig, row: dict) -> ScoreBundle:
    """Rebuild one player's Player Score as an explainable bundle."""
    raw_sources: dict[str, tuple[str, str, str]] = {
        "projection": ("projected_points", "position z-score → logistic", "consensus"),
        "vbd": ("vbd", "points above replacement, scaled to the board leader", "derived"),
        "opportunity": ("exp_points_per_game", "position percentile", "nflverse ff_opportunity"),
        "offense_environment": ("points_per_game", "league percentile", "nflverse team stats"),
        "schedule": ("schedule_raw", "position percentile vs defense-adjusted opponents", "derived"),
        "risk": ("risk_score", "inverted composite of 6 uncertainty signals", "derived"),
    }
    components: dict[str, ComponentScore] = {}
    for key, (score_col, confidence_col) in PLAYER_COMPONENTS.items():
        raw_col, method, source = raw_sources[key]
        components[key] = component(
            name=key,
            raw_value=row.get(raw_col),
            normalized=row.get(score_col),
            confidence=row.get(confidence_col) if row.get(confidence_col) is not None else 1.0,
            method=method,
            source=source,
        )
    return compose("player_score", components, cfg.weights.player_score.as_dict())
