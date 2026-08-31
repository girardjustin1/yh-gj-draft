"""Floor, median, and ceiling: the range of seasons a player might have.

A single projected total hides the difference between a player who will score 210 points
almost regardless of what happens, and one who will score 120 or 300 depending on
whether the job is his. Those are different draft picks.

**The model, and what it rests on.**

Two independent sources of uncertainty are combined:

1. **Where he finishes at his position.** The consensus board tells us his rank, and the
   ECR ``best``/``worst``/``sd`` spread tells us how much the experts disagree about it.
   A player ranked 30th whom one analyst has 12th and another 60th has a genuinely wide
   range of outcomes.
2. **What that finish has been worth.** The historical positional value curve carries a
   ``points_sd`` — how much the points scored by, say, the eighth-best running back have
   varied from season to season.

These are combined in quadrature (they are close to independent: expert disagreement is
about the player, curve variance is about the league environment), then converted to
percentiles on the value curve rather than on a fitted normal — so the floor of an elite
running back is bounded by what actual RB20 seasons have looked like, not by an
extrapolated tail.

Finally the range is skewed by the risk components we already compute. Injury and age
risk lower the floor without lowering the ceiling; a young player with a rising role and
a small sample gets a wider upside. This is why floor and ceiling are not symmetric
around the median.

**What this is not.** It is not a simulation of a season, and it does not know that a
particular player's backup just got hurt. It is an honest width, derived from disagreement
and history, and it should be read as "roughly how wrong could this be" rather than as a
calibrated interval. ``outcome_confidence`` reports how much is behind it.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..config import AppConfig
from ..logging import get_logger

log = get_logger(__name__)

#: z-scores for the reported percentiles.
Z_P10 = -1.2816
Z_P90 = 1.2816

#: Rank uncertainty floor, in positional ranks. Even a consensus pick can bust.
MIN_RANK_SD = 1.5

#: Cap on rank uncertainty so a single wild outlier opinion cannot produce a silly range.
MAX_RANK_SD_FRACTION = 0.9

#: How much injury/age risk pushes the floor down, as a fraction of the floor gap.
RISK_FLOOR_SKEW = 0.45

#: How much an unproven player's ceiling is widened, as a fraction of the ceiling gap.
UPSIDE_SKEW = 0.35


def _curve_lookup(curve: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per position: (ranks, points, points_sd) as aligned arrays for interpolation."""
    lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (position,), group in curve.group_by(["position"], maintain_order=True):
        ordered = group.sort("rank")
        lookup[str(position)] = (
            ordered["rank"].to_numpy().astype(float),
            ordered["points"].to_numpy().astype(float),
            ordered["points_sd"].fill_null(0.0).to_numpy().astype(float),
        )
    return lookup


def add_outcome_range(
    cfg: AppConfig, board: pl.DataFrame, curve: pl.DataFrame
) -> pl.DataFrame:
    """Add ``floor_points``, ``median_points``, ``ceiling_points`` and confidence.

    ``board`` needs ``position``, ``positional_rank``, ``projected_points`` and, where
    available, ``overall_ecr`` / ``ecr_sd`` / ``ecr_best`` / ``ecr_worst`` and the risk
    components. ``curve`` is the output of
    :func:`fantasy_draft.analytics.projections.positional_value_curve`.
    """
    if board.is_empty():
        return board

    if curve.is_empty():
        log.warning("no value curve; outcome range unavailable")
        return board.with_columns(
            pl.col("projected_points").alias("median_points"),
            pl.lit(None, dtype=pl.Float64).alias("floor_points"),
            pl.lit(None, dtype=pl.Float64).alias("ceiling_points"),
            pl.lit(0.0).alias("outcome_confidence"),
        )

    lookup = _curve_lookup(curve)
    n = board.height

    positions = board["position"].to_list()
    ranks = board["positional_rank"].fill_null(999).to_numpy().astype(float)
    projected = board["projected_points"].fill_null(0.0).to_numpy().astype(float)

    def column(name: str, default: float) -> np.ndarray:
        if name not in board.columns:
            return np.full(n, default, dtype=float)
        return board[name].cast(pl.Float64).fill_null(default).to_numpy().astype(float)

    ecr = column("overall_ecr", 250.0)
    ecr_sd = column("ecr_sd", 0.0)
    ecr_best = column("ecr_best", 0.0)
    ecr_worst = column("ecr_worst", 0.0)
    risk_injury = column("risk_injury", 0.0)
    risk_age = column("risk_age", 0.0)
    risk_sample = column("risk_sample", 0.5)

    # --- 1. Uncertainty in where he finishes, expressed in positional ranks -----------
    #
    # ECR spread is an *overall* rank spread. Positional rank grows roughly in proportion
    # to overall rank within a position, so the same relative uncertainty carries over:
    # a player at overall 60 / RB20 with an overall sd of 12 has a positional sd near 4.
    spread = np.where(
        ecr_sd > 0,
        ecr_sd,
        np.where(ecr_worst > ecr_best, (ecr_worst - ecr_best) / 4.0, 0.0),
    )
    scale = np.divide(
        ranks, np.maximum(ecr, 1.0), out=np.ones(n), where=ecr > 0
    )
    rank_sd = np.clip(
        spread * scale,
        MIN_RANK_SD,
        np.maximum(ranks * MAX_RANK_SD_FRACTION, MIN_RANK_SD),
    )
    # A player with no market data at all is maximally uncertain.
    unknown_market = (ecr_sd <= 0) & (ecr_worst <= ecr_best)
    rank_sd = np.where(unknown_market, np.maximum(ranks * 0.5, 6.0), rank_sd)

    # --- 2. Read the value curve at the optimistic and pessimistic ranks ---------------
    ceiling = np.empty(n)
    floor = np.empty(n)
    median = np.empty(n)
    curve_sd = np.zeros(n)
    has_curve = np.zeros(n, dtype=bool)

    for index, position in enumerate(positions):
        entry = lookup.get(str(position))
        if entry is None:
            ceiling[index] = floor[index] = median[index] = projected[index]
            continue
        curve_ranks, curve_points, curve_points_sd = entry
        has_curve[index] = True

        best_rank = max(1.0, ranks[index] + Z_P10 * rank_sd[index])
        worst_rank = ranks[index] + Z_P90 * rank_sd[index]

        median[index] = float(np.interp(ranks[index], curve_ranks, curve_points))
        ceiling[index] = float(np.interp(best_rank, curve_ranks, curve_points))
        floor[index] = float(np.interp(worst_rank, curve_ranks, curve_points))
        curve_sd[index] = float(np.interp(ranks[index], curve_ranks, curve_points_sd))

    # Anchor on the consensus projection so the median matches the board, and shift the
    # range with it rather than letting the curve and the projection disagree.
    offset = projected - median
    median = median + offset
    ceiling = ceiling + offset
    floor = floor + offset

    # --- 3. Add the curve's own season-to-season variance, in quadrature ---------------
    rank_gap_up = np.maximum(ceiling - median, 0.0)
    rank_gap_down = np.maximum(median - floor, 0.0)
    environment = np.abs(Z_P90) * curve_sd * 0.5
    ceiling = median + np.sqrt(rank_gap_up**2 + environment**2)
    floor = median - np.sqrt(rank_gap_down**2 + environment**2)

    # --- 4. Skew for risk. Injury and age cut the floor; an unproven role adds upside --
    downside_risk = np.clip(0.6 * risk_injury + 0.4 * risk_age, 0.0, 1.0)
    floor = floor - RISK_FLOOR_SKEW * downside_risk * np.maximum(median - floor, 0.0)
    ceiling = ceiling + UPSIDE_SKEW * np.clip(risk_sample, 0.0, 1.0) * np.maximum(
        ceiling - median, 0.0
    )

    floor = np.maximum(floor, 0.0)
    ceiling = np.maximum(ceiling, median)

    # --- 5. Confidence: how much is actually behind the range -------------------------
    confidence = np.where(has_curve, 0.55, 0.0)
    confidence = np.where(unknown_market, confidence * 0.4, confidence)
    confidence = np.clip(confidence + 0.15 * (1.0 - np.clip(risk_sample, 0, 1)), 0.0, 0.8)

    return board.with_columns(
        pl.Series("floor_points", floor, dtype=pl.Float64),
        pl.Series("median_points", median, dtype=pl.Float64),
        pl.Series("ceiling_points", ceiling, dtype=pl.Float64),
        pl.Series("outcome_confidence", confidence, dtype=pl.Float64),
    ).with_columns(
        (pl.col("ceiling_points") - pl.col("floor_points")).alias("outcome_range"),
        # Upside skew: >0 means more room above the median than below it.
        (
            (pl.col("ceiling_points") - pl.col("median_points"))
            - (pl.col("median_points") - pl.col("floor_points"))
        ).alias("upside_skew"),
    )


#: Thresholds calibrated against the observed spread on the live 2026 board rather than
#: guessed: among the top 200 players the relative range width runs 0.16 at the 10th
#: percentile to 0.73 at the 90th, with a median of 0.33, and relative skew runs -0.04 to
#: +0.11. Labels are set at those quartiles so "safe" and "volatile" describe where a
#: player actually sits among his peers instead of against an arbitrary constant.
WIDTH_TIGHT = 0.21
WIDTH_WIDE = 0.48
SKEW_UP = 0.055
SKEW_DOWN = -0.02


def outcome_label(row: dict) -> str:
    """A short human descriptor of the shape of a player's range, relative to the board."""
    floor = row.get("floor_points")
    ceiling = row.get("ceiling_points")
    median = row.get("median_points")
    if floor is None or ceiling is None or not median:
        return "unknown range"

    denominator = max(float(median), 1.0)
    width = (float(ceiling) - float(floor)) / denominator
    skew = float(row.get("upside_skew") or 0.0) / denominator

    if width >= WIDTH_WIDE:
        return "boom or bust" if skew > SKEW_UP else "volatile"
    if width <= WIDTH_TIGHT:
        return "safe"
    if skew >= SKEW_UP:
        return "upside"
    if skew <= SKEW_DOWN:
        return "floor play"
    return "balanced"
