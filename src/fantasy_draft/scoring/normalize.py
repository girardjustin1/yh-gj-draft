"""Normalization helpers: raw football quantities to comparable 0-100 scores.

Min/max scaling is avoided deliberately — a single outlier rescales everyone, and one
mis-projected quarterback would compress the entire board. The methods here are chosen
per quantity and recorded in each :class:`ComponentScore`'s ``method`` field so
``ff explain`` can say how a number was produced.
"""

from __future__ import annotations

import polars as pl


def percentile_within(frame: pl.DataFrame, column: str, group: str = "position") -> pl.Expr:
    """Percentile rank of ``column`` within ``group``, as 0-100.

    Used where cross-position comparison of the raw quantity is meaningless — a tight
    end's target share against a receiver's, for example.
    """
    return (
        pl.col(column).rank("average").over(group) / pl.len().over(group) * 100.0
    ).alias(f"{column}_pct")


def logistic_z(
    frame: pl.DataFrame, column: str, group: str | None = None, spread: float = 1.5
) -> pl.Expr:
    """Z-score, squashed through a logistic curve into 0-100.

    Preserves the *size* of gaps — which is the entire point of VBD, where the distance
    between RB3 and RB4 is the decision — while keeping the tails on the scale instead
    of clipping them flat.
    """
    mean = pl.col(column).mean().over(group) if group else pl.col(column).mean()
    std = pl.col(column).std().over(group) if group else pl.col(column).std()
    z = (pl.col(column) - mean) / (std + 1e-9)
    return (100.0 / (1.0 + (-z / spread).exp())).alias(f"{column}_score")


def scale_to_best(column: str, group: str | None = None) -> pl.Expr:
    """Score relative to the best value present: the leader scores 100.

    Appropriate for quantities with a meaningful zero, such as VBD, where "half as much
    value above replacement" is a sensible statement.
    """
    top = pl.col(column).max().over(group) if group else pl.col(column).max()
    floor = pl.lit(0.0)
    return (
        100.0 * (pl.col(column) - floor) / (top - floor + 1e-9)
    ).clip(0, 100).alias(f"{column}_score")
