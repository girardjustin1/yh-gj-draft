"""Value Based Drafting.

    VBD = projected points − replacement-level points at that position

VBD is the only number on the board that is comparable *across* positions, which is why
it, not raw projected points, drives cross-positional decisions. Raw points put six
quarterbacks at the top of a half-PPR board; VBD correctly says the sixth-best QB is
worth less than the twelfth-best running back, because his replacement is nearly as good.
"""

from __future__ import annotations

import polars as pl

from ..config import AppConfig
from .replacement import ReplacementLevel, replacement_levels


def add_vbd(
    projections: pl.DataFrame, levels: dict[str, ReplacementLevel]
) -> pl.DataFrame:
    """Add ``replacement_points``, ``vbd``, ``positional_rank`` and ``overall_rank``."""
    if projections.is_empty():
        return projections

    mapping = {position: level.points for position, level in levels.items()}
    frame = projections.with_columns(
        pl.col("position").replace_strict(mapping, default=None).alias("replacement_points")
    ).with_columns(
        (pl.col("projected_points") - pl.col("replacement_points")).alias("vbd")
    )
    return frame.with_columns(
        pl.col("projected_points")
        .rank("ordinal", descending=True)
        .over("position")
        .cast(pl.Int32)
        .alias("positional_rank"),
        pl.col("vbd").rank("ordinal", descending=True).cast(pl.Int32).alias("overall_rank"),
    ).sort("vbd", descending=True, nulls_last=True)


def compute_vbd(
    cfg: AppConfig, projections: pl.DataFrame, positions: tuple[str, ...] | None = None
) -> tuple[pl.DataFrame, dict[str, ReplacementLevel]]:
    """Convenience wrapper: derive replacement levels, then apply them.

    ``positions`` restricts the calculation to positions we actually project, so
    unmodelled ones (kickers, defences) do not generate spurious empty-pool warnings.
    """
    levels = replacement_levels(cfg, projections, positions)
    return add_vbd(projections, levels), levels
