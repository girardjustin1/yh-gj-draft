"""Replacement level: the baseline VBD is measured against.

The number that matters is not "how many of this position start" but "how many come off
the board before the position stops being startable". Those differ a lot: a 12-team
league starts 24 RBs but drafts closer to 43, because managers hoard the position.

Three methods, selected by ``replacement.method`` in ``config/scoring_weights.yaml``:

``starter_demand``
    ``teams × (dedicated slots + share of flex slots) × bench_multiplier[pos]``.
    Derived entirely from the league's own rules, so a superflex or 3-WR league moves
    the baseline automatically.

``fixed_rank``
    The literal ranks in config. A blunt instrument, but immune to a strange roster
    configuration producing an absurd baseline.

``blended`` (default)
    A weighted mix of the two, so an unusual league still lands somewhere sane.

Replacement *points* are then averaged over a window of ranks around that index, because
one noisy projection at exactly the replacement rank would otherwise shift every VBD
number at that position.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..config import AppConfig
from ..logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ReplacementLevel:
    """The replacement baseline for one position, with its derivation attached."""

    position: str
    rank: int
    points: float
    method: str
    starter_demand: float
    bench_multiplier: float
    players_available: int

    @property
    def explanation(self) -> str:
        return (
            f"{self.position}: {self.starter_demand:.1f} league-wide starting slots "
            f"× {self.bench_multiplier:.2f} bench factor → replacement at "
            f"{self.position}{self.rank} ({self.points:.1f} pts, {self.method})"
        )


def replacement_rank(cfg: AppConfig, position: str) -> tuple[int, str, float, float]:
    """Return ``(rank, method, starter_demand, bench_multiplier)`` for ``position``."""
    config = cfg.weights.replacement
    demand = cfg.league.starter_demand(position)
    multiplier = config.bench_multiplier.get(position, 1.0)
    fixed = config.fixed_rank.get(position, max(1, int(round(demand))))

    demand_rank = demand * multiplier
    if config.method == "fixed_rank" or demand <= 0:
        return max(1, fixed), "fixed_rank", demand, multiplier
    if config.method == "starter_demand":
        return max(1, int(round(demand_rank))), "starter_demand", demand, multiplier

    blended = config.blend_weight * demand_rank + (1 - config.blend_weight) * fixed
    return max(1, int(round(blended))), "blended", demand, multiplier


def replacement_levels(
    cfg: AppConfig, projections: pl.DataFrame, positions: tuple[str, ...] | None = None
) -> dict[str, ReplacementLevel]:
    """Compute the replacement baseline for each position from a projection frame.

    ``projections`` needs ``position`` and ``projected_points``.
    """
    window = cfg.weights.replacement.smoothing_window
    levels: dict[str, ReplacementLevel] = {}

    for position in positions or cfg.positions:
        pool = (
            projections.filter(pl.col("position") == position)
            .drop_nulls("projected_points")
            .sort("projected_points", descending=True)
        )
        rank, method, demand, multiplier = replacement_rank(cfg, position)
        available = pool.height
        if available == 0:
            levels[position] = ReplacementLevel(
                position, rank, 0.0, method + " (no players)", demand, multiplier, 0
            )
            log.warning("no projections for position", extra={"position": position})
            continue

        index = min(rank, available) - 1
        lo = max(0, index - window)
        hi = min(available, index + window + 1)
        points = float(pool["projected_points"][lo:hi].mean())
        levels[position] = ReplacementLevel(
            position=position,
            rank=min(rank, available),
            points=points,
            method=method,
            starter_demand=demand,
            bench_multiplier=multiplier,
            players_available=available,
        )
    return levels
