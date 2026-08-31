"""Positional scarcity: what does it cost me to *not* take this position now?

Scarcity is a supply-and-demand question about the remaining board, not a popularity
contest. It deliberately does **not** simply reward whatever position has just been
drafted heavily — a run on running backs can mean RB is scarce, or it can mean the room
overdrafted and pushed elite receivers down to us. Those need different answers.

The measure combines four things, all computed on the *undrafted* pool:

1. **Startable supply versus remaining league demand.** How many players at this
   position are still worth starting, against how many starting slots the league still
   has to fill.
2. **Drop to the next player** at the position, in points.
3. **Drop to the next tier** — the cliff.
4. **Distance above replacement.** A position whose 30th player is nearly as good as its
   15th is not scarce however few are left.

During a live draft, ``expected_gone_before_next_pick`` folds in how many players at this
position are likely to disappear before our next turn, which is what converts scarcity
into urgency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ..config import AppConfig
from .replacement import ReplacementLevel


@dataclass(slots=True)
class PositionScarcity:
    """Scarcity diagnostics for one position, on the current board."""

    position: str
    available: int
    startable_available: int
    remaining_demand: float
    supply_ratio: float
    replacement_points: float
    best_available_points: float
    points_above_replacement: float
    expected_gone_before_next_pick: float | None = None
    score: float = 50.0
    notes: list[str] = field(default_factory=list)

    @property
    def explanation(self) -> str:
        gone = (
            f", ~{self.expected_gone_before_next_pick:.1f} likely gone before our next pick"
            if self.expected_gone_before_next_pick is not None
            else ""
        )
        return (
            f"{self.position}: {self.startable_available} startable left against "
            f"{self.remaining_demand:.1f} remaining starting slots "
            f"(supply ratio {self.supply_ratio:.2f}){gone}"
        )


def position_scarcity(
    cfg: AppConfig,
    board: pl.DataFrame,
    levels: dict[str, ReplacementLevel],
    drafted_by_position: dict[str, int] | None = None,
    expected_gone: dict[str, float] | None = None,
    positions: tuple[str, ...] | None = None,
) -> dict[str, PositionScarcity]:
    """Compute scarcity per position over the currently available players.

    ``board`` must already be filtered to undrafted players.
    """
    drafted_by_position = drafted_by_position or {}
    expected_gone = expected_gone or {}
    result: dict[str, PositionScarcity] = {}

    for position in positions or cfg.positions:
        pool = (
            board.filter(pl.col("position") == position)
            .drop_nulls("projected_points")
            .sort("projected_points", descending=True)
        )
        level = levels.get(position)
        replacement_points = level.points if level else 0.0

        total_demand = cfg.league.starter_demand(position)
        remaining_demand = max(0.0, total_demand - drafted_by_position.get(position, 0))

        startable = int(
            pool.filter(pl.col("projected_points") > replacement_points).height
        )
        best = float(pool["projected_points"][0]) if pool.height else replacement_points

        # Supply ratio: startable players per remaining starting slot. Below 1.0 means
        # the league cannot fill its lineups from what is left.
        supply_ratio = startable / remaining_demand if remaining_demand > 0 else 99.0

        result[position] = PositionScarcity(
            position=position,
            available=pool.height,
            startable_available=startable,
            remaining_demand=remaining_demand,
            supply_ratio=supply_ratio,
            replacement_points=replacement_points,
            best_available_points=best,
            points_above_replacement=best - replacement_points,
            expected_gone_before_next_pick=expected_gone.get(position),
        )

    _score(result, cfg)
    return result


def _score(scarcity: dict[str, PositionScarcity], cfg: AppConfig) -> None:
    """Turn the raw diagnostics into a comparable 0-100 score, in place."""
    if not scarcity:
        return

    # Supply pressure: scarce when few startable players remain per slot to fill.
    # A ratio of 1 is "exactly enough" and scores high; 4+ is comfortable.
    def supply_component(entry: PositionScarcity) -> float:
        ratio = max(entry.supply_ratio, 0.01)
        if ratio >= 4.0:
            return 0.0
        return float(min(100.0, 100.0 * (1.0 - (ratio - 1.0) / 3.0))) if ratio > 1.0 else 100.0

    # Value pressure: how much better the best available is than replacement, relative
    # to the same quantity at other positions.
    edges = {p: max(0.0, e.points_above_replacement) for p, e in scarcity.items()}
    best_edge = max(edges.values()) if edges else 1.0
    best_edge = best_edge if best_edge > 0 else 1.0

    for position, entry in scarcity.items():
        supply = supply_component(entry)
        edge = 100.0 * edges[position] / best_edge

        urgency = 0.0
        if entry.expected_gone_before_next_pick is not None and entry.startable_available:
            share_gone = entry.expected_gone_before_next_pick / max(
                entry.startable_available, 1
            )
            urgency = float(min(100.0, 100.0 * share_gone))

        if entry.expected_gone_before_next_pick is None:
            entry.score = round(0.45 * supply + 0.55 * edge, 2)
        else:
            entry.score = round(0.30 * supply + 0.40 * edge + 0.30 * urgency, 2)

        if entry.supply_ratio < 1.0:
            entry.notes.append("fewer startable players left than starting slots to fill")
        if entry.points_above_replacement < 5:
            entry.notes.append("best available is barely above replacement")


def scarcity_frame(scarcity: dict[str, PositionScarcity]) -> pl.DataFrame:
    """Scarcity diagnostics as a frame, for joining onto the board."""
    if not scarcity:
        return pl.DataFrame(schema={"position": pl.Utf8, "scarcity_score": pl.Float64})
    return pl.DataFrame(
        [
            {
                "position": entry.position,
                "scarcity_score": entry.score,
                "startable_available": entry.startable_available,
                "remaining_demand": entry.remaining_demand,
                "supply_ratio": entry.supply_ratio,
                "points_above_replacement": entry.points_above_replacement,
                "expected_gone_before_next_pick": entry.expected_gone_before_next_pick,
            }
            for entry in scarcity.values()
        ]
    )
