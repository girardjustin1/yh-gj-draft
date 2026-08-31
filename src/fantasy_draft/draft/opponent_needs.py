"""What is each manager picking before our next turn likely to take?

This is the input that turns a static board into a decision. Whether a player survives
twelve picks depends far less on his ADP than on whether the eight managers in between
still need his position.

**No personal profiling.** We model roster *shape*, not people. Every manager is treated
as a generic drafter who mostly follows the market and reaches for a position when their
lineup demands it. Nothing here depends on knowing who anyone is.

**The market prior comes from the market, not from a table of guesses.** Rather than
hardcoding "round 3 is 40% RB", we read the positional mix of the consensus board in the
ADP range the picks are about to land in. If the 2026 board is receiver-heavy at picks
40-60, that falls out of the data.

That prior is then modulated by each roster's unfilled starting slots, and renormalized.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..config import LeagueConfig
from ..constants import FLEX_ELIGIBILITY, OFFENSE_POSITIONS
from ..models import PositionNeed, RosterSnapshot
from .state import DraftState

#: How strongly an unfilled starting slot raises a position's probability.
NEED_STRENGTH = 1.6

#: Floor so a "full" position never drops to literally zero — managers take value.
MIN_POSITION_PROBABILITY = 0.02


@dataclass(frozen=True, slots=True)
class MarketPrior:
    """Positional mix of the consensus board over a range of picks."""

    lo: int
    hi: int
    mix: dict[str, float]


def market_prior(board: pl.DataFrame, lo: int, hi: int, positions: tuple[str, ...] = OFFENSE_POSITIONS) -> MarketPrior:
    """Positional mix of the players whose ADP falls in ``[lo, hi]``.

    Falls back to widening the window when the range is sparse, and finally to a flat
    prior, so an unusual board never produces an empty distribution.
    """
    if board.is_empty() or "adp" not in board.columns:
        return MarketPrior(lo, hi, dict.fromkeys(positions, 1.0 / len(positions)))

    span = max(hi - lo, 1)
    for widen in (0, span, span * 3):
        window = board.filter(
            pl.col("adp").is_not_null()
            & (pl.col("adp") >= lo - widen)
            & (pl.col("adp") <= hi + widen)
            & pl.col("position").is_in(list(positions))
        )
        if window.height >= 6:
            counts = window.group_by("position").len()
            total = int(counts["len"].sum())
            mix = {p: 0.0 for p in positions}
            for row in counts.iter_rows(named=True):
                mix[row["position"]] = row["len"] / total
            return MarketPrior(lo, hi, mix)

    return MarketPrior(lo, hi, dict.fromkeys(positions, 1.0 / len(positions)))


def unfilled_starters(
    league: LeagueConfig, roster: RosterSnapshot | None, positions: tuple[str, ...] = OFFENSE_POSITIONS
) -> dict[str, float]:
    """Starting slots this roster still needs to fill, per position.

    Dedicated slots are filled first; whatever is left over is applied to flex slots,
    split across the positions eligible for them. A roster with three running backs and
    no tight end shows a full TE need and a fractional RB need from the flex.
    """
    counts = dict(roster.position_counts) if roster else {}
    remaining = {p: float(counts.get(p, 0)) for p in positions}
    need: dict[str, float] = {}

    for position in positions:
        required = float(league.roster.dedicated.get(position, 0))
        have = remaining[position]
        used = min(have, required)
        remaining[position] = have - used
        need[position] = max(0.0, required - used)

    # Flex slots: satisfied by whatever depth is left over, shared across eligibles.
    for slot_name, count in league.roster.flex_counts.items():
        eligible = [p for p in FLEX_ELIGIBILITY[slot_name] if p in positions]
        if not eligible:
            continue
        spare = sum(remaining[p] for p in eligible)
        unfilled = max(0.0, count - spare)
        consumed = min(spare, count)
        for position in eligible:
            if spare > 0:
                remaining[position] -= consumed * (remaining[position] / spare)
            need[position] += unfilled / len(eligible)

    return need


def position_probabilities(
    league: LeagueConfig,
    roster: RosterSnapshot | None,
    prior: MarketPrior,
    round_number: int,
    positions: tuple[str, ...] = OFFENSE_POSITIONS,
) -> dict[str, float]:
    """Probability this manager takes each position with their next selection."""
    need = unfilled_starters(league, roster, positions)

    # Need matters more as the draft goes on: in round 2 everyone takes the best player,
    # by round 8 they are filling a lineup.
    depth = min(1.0, round_number / max(league.draft.rounds * 0.6, 1.0))
    strength = 1.0 + NEED_STRENGTH * depth

    weights: dict[str, float] = {}
    for position in positions:
        base = max(prior.mix.get(position, 0.0), MIN_POSITION_PROBABILITY)
        multiplier = 1.0 + strength * min(need.get(position, 0.0), 2.0)
        weights[position] = base * multiplier

    total = sum(weights.values())
    if total <= 0:
        return dict.fromkeys(positions, 1.0 / len(positions))
    return {position: weight / total for position, weight in weights.items()}


def opponent_needs(
    state: DraftState,
    league: LeagueConfig,
    board: pl.DataFrame,
    positions: tuple[str, ...] = OFFENSE_POSITIONS,
) -> list[PositionNeed]:
    """Estimate positional intent for every pick between our turn and our next one."""
    upcoming = state.slots_before_next_turn()
    if not upcoming:
        return []

    rosters = state.rosters()
    lo = min(overall for overall, _ in upcoming)
    hi = max(overall for overall, _ in upcoming)
    prior = market_prior(board, lo, hi, positions)

    # Track picks as they are notionally made, so a manager who takes a running back at
    # pick 43 is less likely to take another at 54.
    projected: dict[str, list[str]] = {}
    needs: list[PositionNeed] = []

    for overall, slot in upcoming:
        team_id = state.slot_to_team.get(slot, f"slot-{slot}")
        roster = rosters.get(team_id)
        if projected.get(team_id):
            merged = RosterSnapshot(
                team_id=team_id,
                slot=slot,
                is_me=bool(roster and roster.is_me),
                player_keys=list(roster.player_keys) if roster else [],
                positions=(list(roster.positions) if roster else []) + projected[team_id],
            )
        else:
            merged = roster

        round_number = state.board.round_for(overall)
        probabilities = position_probabilities(league, merged, prior, round_number, positions)
        top = max(probabilities, key=probabilities.get)
        projected.setdefault(team_id, []).append(top)

        shape = (
            ", ".join(f"{p}{n}" for p, n in sorted(merged.position_counts.items()))
            if merged else "empty"
        )
        needs.append(
            PositionNeed(
                team_id=team_id,
                slot=slot,
                pick_overall=overall,
                probabilities={p: round(v, 4) for p, v in probabilities.items()},
                rationale=f"roster {shape}; round {round_number}",
            )
        )
    return needs


def expected_position_demand(needs: list[PositionNeed]) -> dict[str, float]:
    """Expected number of each position taken before our next pick."""
    demand: dict[str, float] = {}
    for need in needs:
        for position, probability in need.probabilities.items():
            demand[position] = demand.get(position, 0.0) + probability
    return demand
