"""Positional tiers.

A tier boundary is a *cliff*: the point where waiting one more round costs you real
points rather than a rounding error. Tiers are what turn "he's ranked 14th" into "he's
the last player at his level", which is the fact that actually changes a draft decision.

The default method is deterministic and interpretable: walk each position in projection
order and start a new tier wherever the gap to the next player is unusually large. The
same board always tiers the same way, and every boundary can be pointed at and explained.

**Why gaps are measured locally.** Projection gaps shrink monotonically down a position:
among 2026 running backs the top gaps run 17, 13, 14, 9 points, and by RB20 they are a
flat 4. A single position-wide threshold therefore fires on every one of the top eight
players -- making each his own tier -- and then never fires again, dumping the remaining
130 backs into one tier. Useless in both directions.

So each gap is standardized against its own *neighbourhood* (a centered rolling window of
nearby gaps) rather than against the whole position. A cliff is a gap that is large
relative to the gaps around it, which is what "cliff" means to a drafter: everyone near
here is close together, except right here.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..config import AppConfig

#: Gaps are compared against this many neighbours on each side.
LOCAL_WINDOW = 6


def _local_gap_scores(gaps: np.ndarray) -> np.ndarray:
    """Standardize each gap against its neighbours: ``(gap - local mean) / local sd``."""
    n = len(gaps)
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - LOCAL_WINDOW)
        hi = min(n, i + LOCAL_WINDOW + 1)
        neighbourhood = np.delete(gaps[lo:hi], i - lo)
        if len(neighbourhood) < 2:
            continue
        centre = float(neighbourhood.mean())
        scale = float(neighbourhood.std())
        # A floor on the scale stops a perfectly flat stretch from turning a rounding
        # difference into an infinitely significant cliff.
        scale = max(scale, 0.15 * centre, 1e-6)
        scores[i] = (gaps[i] - centre) / scale
    return scores


def _tier_one_position(points: list[float], gap_sigma: float, max_tiers: int,
                       min_tier_size: int) -> list[int]:
    """Assign tier numbers to a descending list of projected points."""
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [1]

    gaps = np.diff(np.array(points, dtype=float)) * -1.0  # positive = drop to next player
    scores = _local_gap_scores(gaps)

    tiers = [1]
    current = 1
    size = 1
    for score in scores:
        starts_new = score >= gap_sigma and size >= min_tier_size and current < max_tiers
        if starts_new:
            current += 1
            size = 1
        else:
            size += 1
        tiers.append(current)
    return tiers


def assign_tiers(cfg: AppConfig, frame: pl.DataFrame) -> pl.DataFrame:
    """Add tier columns to a projection frame.

    Adds ``tier``, ``tier_rank``, ``tier_size``, ``points_to_next_player``,
    ``points_to_next_tier``, and ``tier_cliff_score``.
    """
    if frame.is_empty():
        return frame

    config = cfg.weights.tiers
    out: list[pl.DataFrame] = []

    for (position,), group in frame.group_by(["position"], maintain_order=True):
        group = group.sort("projected_points", descending=True, nulls_last=True)
        depth = config.depth.get(str(position), 60)
        max_tiers = config.max_tiers.get(str(position), 10)

        head = group.head(depth)
        tail = group.slice(depth)
        points = head["projected_points"].fill_null(0.0).to_list()
        tiers = _tier_one_position(points, config.gap_sigma, max_tiers, config.min_tier_size)

        # Everything past the tiering depth is one residual tier: below that point the
        # differences are inside the noise of the projection itself.
        if tail.height:
            tiers = tiers + [(max(tiers) + 1 if tiers else 1)] * tail.height
            group = pl.concat([head, tail])

        group = group.with_columns(pl.Series("tier", tiers, dtype=pl.Int32))
        out.append(group)

    tiered = pl.concat(out)

    tiered = tiered.with_columns(
        pl.int_range(1, pl.len() + 1).over(["position", "tier"]).cast(pl.Int32).alias("tier_rank"),
        pl.len().over(["position", "tier"]).cast(pl.Int32).alias("tier_size"),
        (
            pl.col("projected_points") - pl.col("projected_points").shift(-1).over("position")
        ).alias("points_to_next_player"),
    )

    # Points from this player down to the best player in the *next* tier.
    next_tier_best = (
        tiered.group_by(["position", "tier"])
        .agg(pl.col("projected_points").max().alias("tier_best"))
        .with_columns((pl.col("tier") - 1).alias("prev_tier"))
        .select(
            pl.col("position"),
            pl.col("prev_tier").alias("tier"),
            pl.col("tier_best").alias("next_tier_best"),
        )
    )
    tiered = tiered.join(next_tier_best, on=["position", "tier"], how="left").with_columns(
        (pl.col("projected_points") - pl.col("next_tier_best")).alias("points_to_next_tier")
    )

    return tiered.drop("next_tier_best").sort(
        "vbd" if "vbd" in tiered.columns else "projected_points",
        descending=True, nulls_last=True,
    )


def expected_loss_by_waiting(
    board: pl.DataFrame, slides: dict[str, float], default_slide: float = 3.0
) -> pl.DataFrame:
    """What does waiting until our next pick actually cost, in projected points?

    This is the honest form of "tier cliff". Tier membership alone is not the right
    signal: a cliff sits at the *bottom* edge of a tier, but every player in the tier
    inherits it. Mid-draft that gave the first of fourteen available tier-3 quarterbacks
    a maximum cliff score when the next quarterback was ten points away and thirteen more
    sat between him and the drop.

    So instead of asking "how far is the drop behind my tier", we ask the question that
    actually bears on the decision: **if I skip him, who do I get instead?** Slide down
    his position's available list by the number of players at that position expected to
    go before our next turn, and take the difference. Tier cliffs fall out of this
    naturally — if the next few players really are far worse, the loss is large — and it
    is directly comparable across positions because it is measured in points.

    ``slides`` maps position to the expected number taken before our next pick; positions
    absent from it use ``default_slide``.

    Adds ``expected_loss_points`` and a normalized ``tier_cliff_score``.
    """
    if board.is_empty():
        return board.with_columns(
            pl.lit(0.0).alias("expected_loss_points"),
            pl.lit(50.0).alias("tier_cliff_score"),
        )

    losses: dict[str, float] = {}
    for (position,), group in board.group_by(["position"], maintain_order=True):
        ordered = group.sort("projected_points", descending=True, nulls_last=True)
        points = ordered["projected_points"].fill_null(0.0).to_list()
        keys = ordered["player_key"].to_list()
        n = len(points)
        slide = max(1, int(round(slides.get(str(position), default_slide))))
        for index, key in enumerate(keys):
            fallback = points[min(index + slide, n - 1)]
            losses[key] = max(0.0, points[index] - fallback)

    frame = board.with_columns(
        pl.col("player_key")
        .replace_strict(losses, default=0.0)
        .cast(pl.Float64)
        .alias("expected_loss_points")
    )
    # Waiting can never cost more than a player's edge over a freely available
    # replacement -- that is the floor of what we end up with either way.
    if "vbd" in frame.columns:
        frame = frame.with_columns(
            pl.min_horizontal(
                pl.col("expected_loss_points"), pl.col("vbd").fill_null(0.0).clip(0, None)
            ).alias("expected_loss_points")
        )

    ceiling = float(frame["expected_loss_points"].max() or 0.0)
    return frame.with_columns(
        (100.0 * pl.col("expected_loss_points") / (ceiling + 1e-9))
        .clip(0, 100)
        .alias("tier_cliff_score")
    )
