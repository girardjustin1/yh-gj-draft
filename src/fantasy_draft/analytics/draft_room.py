"""Reading the room.

Position runs are facts; what they *mean* is the interesting part, and it is not
"everyone is taking running backs, take a running back".

Eight RBs in twelve picks says two things at once. Running back supply is draining, and
— because those twelve managers each spent their pick on one position — receiver value
has been pushed down toward us. A model that only counts the run chases it. This module
reports both halves and lets the Draft Now Score weigh them.

Outputs per position:

``demand``
    0-100. How hard the room is currently competing for the position, from recent picks
    and the estimated intent of the managers ahead of our next turn.
``run_intensity``
    How concentrated the last few picks were on this position, against what the board's
    own ADP mix says was expected there.
``value_created``
    The other half. When a position is overdrafted, the positions being skipped are
    falling — this is how much value has accumulated at each position relative to the
    pick we are at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from ..constants import OFFENSE_POSITIONS
from ..draft.opponent_needs import expected_position_demand, market_prior
from ..draft.state import DraftState
from ..models import PositionNeed


@dataclass(slots=True)
class DraftRoomRead:
    """The room's behaviour, per position."""

    demand: dict[str, float] = field(default_factory=dict)
    run_intensity: dict[str, float] = field(default_factory=dict)
    value_created: dict[str, float] = field(default_factory=dict)
    recent_mix: dict[str, float] = field(default_factory=dict)
    expected_mix: dict[str, float] = field(default_factory=dict)
    expected_losses: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def demand_for(self, position: str) -> float:
        return self.demand.get(position, 50.0)

    def value_for(self, position: str) -> float:
        return self.value_created.get(position, 50.0)


def read_room(
    state: DraftState,
    board: pl.DataFrame,
    needs: list[PositionNeed],
    window: int = 12,
    positions: tuple[str, ...] = OFFENSE_POSITIONS,
) -> DraftRoomRead:
    """Analyse recent picks and upcoming intent.

    ``board`` should be the *available* board, carrying ``adp`` and ``player_score``.
    """
    read = DraftRoomRead()
    recent = [p.position for p in state.recent_picks(window) if p.position in positions]

    # What the room actually took recently.
    if recent:
        read.recent_mix = {
            position: recent.count(position) / len(recent) for position in positions
        }
    else:
        read.recent_mix = dict.fromkeys(positions, 0.0)

    # What the consensus board says should have gone in that range — the honest baseline
    # for "is this a run?".
    lo = max(1, state.current_pick - window)
    prior = market_prior(board, lo, state.current_pick, positions)
    read.expected_mix = prior.mix

    for position in positions:
        actual = read.recent_mix.get(position, 0.0)
        expected = max(prior.mix.get(position, 0.0), 0.02)
        # A run is a position going at more than its expected share.
        read.run_intensity[position] = round(
            min(100.0, 100.0 * max(0.0, (actual - expected) / expected) / 2.0), 1
        )

    # Forward-looking demand from the managers ahead of our next turn.
    read.expected_losses = expected_position_demand(needs)
    upcoming = len(needs) or 1
    forward = {
        position: read.expected_losses.get(position, 0.0) / upcoming for position in positions
    }

    for position in positions:
        backward = read.recent_mix.get(position, 0.0)
        combined = 0.45 * backward + 0.55 * forward.get(position, 0.0)
        # Scale against an even split: 1/len(positions) is "normal" and maps near 50.
        even = 1.0 / len(positions)
        read.demand[position] = round(min(100.0, 50.0 * combined / even), 1)

    # The other half of a run: what value has fallen to us because the room ignored it.
    read.value_created = _value_created(board, state.current_pick, positions)

    hot = [p for p in positions if read.run_intensity.get(p, 0) >= 50]
    if hot:
        cold = sorted(positions, key=lambda p: read.demand.get(p, 50))[:2]
        read.notes.append(
            f"Run on {', '.join(hot)}. That drains {'/'.join(hot)} supply, but it also "
            f"means {' and '.join(cold)} value is falling toward us — check both before "
            f"chasing the run."
        )
    return read


def _value_created(
    board: pl.DataFrame, current_pick: int, positions: tuple[str, ...]
) -> dict[str, float]:
    """How far the best available player at each position has fallen past his ADP.

    Positive means the room has skipped the position and value is sitting there.
    """
    result: dict[str, float] = {}
    if board.is_empty() or "adp" not in board.columns:
        return dict.fromkeys(positions, 50.0)

    for position in positions:
        pool = (
            board.filter((pl.col("position") == position) & pl.col("adp").is_not_null())
            .sort("adp")
            .head(3)
        )
        if pool.is_empty():
            result[position] = 50.0
            continue
        # Average fall across the top few available at the position, so one outlier
        # sliding does not read as the whole position being cheap.
        fall = float((pl.Series(pool["adp"]) - current_pick).mean())
        result[position] = round(max(0.0, min(100.0, 50.0 + 50.0 * fall / 24.0)), 1)
    return result
