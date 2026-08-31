"""Roster fit: a modifier, never a rule.

A team with three running backs may still take a fourth if the value is extreme. Roster
need should tilt a close call, not veto a clear one — the fastest way to draft badly is
to reach two rounds early because a slot is empty.

So fit is scored 0-100 around a neutral 50:

* Filling an empty *starting* slot is a real bonus.
* Adding useful depth at a flex-eligible position is a small bonus.
* Stacking a position we have already filled is a mild penalty, growing with the pile.
* Only structurally irrational builds are hit hard: a fifth quarterback in a one-QB
  league, or leaving a starting slot unfillable with the picks remaining.
"""

from __future__ import annotations

import polars as pl

from ..config import LeagueConfig
from ..constants import FLEX_ELIGIBILITY, OFFENSE_POSITIONS
from ..draft.opponent_needs import unfilled_starters
from ..models import RosterSnapshot

#: Beyond this many at one position, further additions look irrational.
SANE_MAXIMUM: dict[str, int] = {"QB": 3, "RB": 7, "WR": 8, "TE": 3, "K": 2, "DST": 2}


def roster_fit_scores(
    league: LeagueConfig,
    roster: RosterSnapshot | None,
    rounds_remaining: int,
    positions: tuple[str, ...] = OFFENSE_POSITIONS,
) -> dict[str, float]:
    """Fit score 0-100 per position for the roster we currently hold."""
    counts = dict(roster.position_counts) if roster else {}
    need = unfilled_starters(league, roster, positions)
    scores: dict[str, float] = {}

    for position in positions:
        have = counts.get(position, 0)
        unfilled = need.get(position, 0.0)

        score = 50.0
        # Empty starting slot: a genuine, bounded bonus.
        score += 28.0 * min(unfilled, 1.5) / 1.5

        # Flex-eligible depth retains value even when the dedicated slot is full.
        if unfilled <= 0 and league.roster.flex_slots_accepting(position) > 0:
            score += 6.0

        # Stacking: mild and progressive, so an elite fourth RB is still draftable.
        dedicated = league.roster.dedicated.get(position, 0)
        flex_capacity = league.roster.flex_slots_accepting(position)
        comfortable = dedicated + flex_capacity + 1
        if have > comfortable:
            score -= 7.0 * (have - comfortable)

        # Structural irrationality gets hit hard; nothing else does.
        if have >= SANE_MAXIMUM.get(position, 6):
            score -= 35.0

        # Late in the draft, an unfillable starting slot is an emergency.
        if unfilled >= 1 and rounds_remaining <= 2:
            score += 20.0

        scores[position] = max(0.0, min(100.0, score))

    return scores


def add_roster_fit(
    board: pl.DataFrame,
    league: LeagueConfig,
    roster: RosterSnapshot | None,
    rounds_remaining: int,
) -> pl.DataFrame:
    """Attach ``roster_fit_score`` to every row of the board."""
    if board.is_empty():
        return board
    scores = roster_fit_scores(league, roster, rounds_remaining)
    return board.with_columns(
        pl.col("position").replace_strict(scores, default=50.0).alias("roster_fit_score"),
        pl.lit(1.0).alias("roster_fit_confidence"),
    )


def eligible_slots(league: LeagueConfig, position: str) -> list[str]:
    """Which lineup slots this position can occupy. Used in explanations."""
    slots = [position] if league.roster.dedicated.get(position, 0) else []
    slots += [
        name for name in league.roster.flex_counts
        if position in FLEX_ELIGIBILITY[name]
    ]
    return slots
