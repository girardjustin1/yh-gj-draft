"""Snake-draft pick arithmetic.

This module has no dependencies beyond the standard library and is exhaustively unit
tested, because a single off-by-one here silently corrupts every downstream number:
which picks are ours, how many selections happen before our next turn, and therefore
every survival probability and two-pick expected value.

Conventions used everywhere in this project:

* ``overall`` picks are 1-indexed (the first pick of the draft is 1).
* ``round`` is 1-indexed.
* ``slot`` is the 1-indexed draft position, 1..teams, counted left to right in round 1.

Supported orders:

``snake``
    Odd rounds run slot 1 -> N, even rounds run N -> 1.
``snake`` with ``third_round_reversal``
    Rounds 1 and 2 snake normally, then round 3 *repeats* round 2's order and the snake
    resumes from there: forward, backward, backward, forward, backward, forward, ...
    Equivalently, every round from 3 onward is inverted relative to a standard snake.
    This removes the 2/3 turn back-to-back from the slot-1 manager and hands it to the
    slot-N manager, which is the whole point of the format. Sleeper calls this a
    "reversal round"; FFPC calls it 3RR.
``linear``
    Every round runs slot 1 -> N.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DraftType = Literal["snake", "linear"]

FORWARD = 1
BACKWARD = -1


def _validate(teams: int, slot: int | None = None, round_: int | None = None) -> None:
    if teams < 1:
        raise ValueError(f"teams must be >= 1, got {teams}")
    if slot is not None and not 1 <= slot <= teams:
        raise ValueError(f"slot must be between 1 and {teams}, got {slot}")
    if round_ is not None and round_ < 1:
        raise ValueError(f"round must be >= 1, got {round_}")


def round_direction(
    round_: int,
    draft_type: DraftType = "snake",
    third_round_reversal: bool = False,
) -> int:
    """Return ``FORWARD`` (1) or ``BACKWARD`` (-1) for ``round_``."""
    if round_ < 1:
        raise ValueError(f"round must be >= 1, got {round_}")
    if draft_type == "linear":
        return FORWARD
    standard = FORWARD if round_ % 2 == 1 else BACKWARD
    # Under 3RR, round 3 repeats round 2's order, so rounds 3+ are all inverted.
    if third_round_reversal and round_ >= 3:
        return -standard
    return standard


def pick_number(
    round_: int,
    slot: int,
    teams: int,
    draft_type: DraftType = "snake",
    third_round_reversal: bool = False,
) -> int:
    """Overall pick number for ``slot`` in ``round_``.

    >>> pick_number(1, 7, 12)
    7
    >>> pick_number(2, 7, 12)
    18
    >>> pick_number(3, 7, 12)
    31
    """
    _validate(teams, slot, round_)
    forward = round_direction(round_, draft_type, third_round_reversal) == FORWARD
    within = slot if forward else teams - slot + 1
    return (round_ - 1) * teams + within


def round_for_pick(overall: int, teams: int) -> int:
    """Which round contains ``overall``."""
    _validate(teams)
    if overall < 1:
        raise ValueError(f"overall must be >= 1, got {overall}")
    return (overall - 1) // teams + 1


def slot_for_pick(
    overall: int,
    teams: int,
    draft_type: DraftType = "snake",
    third_round_reversal: bool = False,
) -> int:
    """Which draft slot owns ``overall``. Inverse of :func:`pick_number`.

    >>> slot_for_pick(18, 12)
    7
    """
    round_ = round_for_pick(overall, teams)
    within = (overall - 1) % teams + 1
    forward = round_direction(round_, draft_type, third_round_reversal) == FORWARD
    return within if forward else teams - within + 1


def pick_label(overall: int, teams: int) -> str:
    """Format an overall pick as ``round.pick_in_round``, e.g. ``"4.06"``."""
    round_ = round_for_pick(overall, teams)
    within = (overall - 1) % teams + 1
    return f"{round_}.{within:02d}"


def picks_for_slot(
    slot: int,
    teams: int,
    rounds: int,
    draft_type: DraftType = "snake",
    third_round_reversal: bool = False,
) -> list[int]:
    """Every overall pick belonging to ``slot``, in order.

    >>> picks_for_slot(7, 12, 4)
    [7, 18, 31, 42]
    """
    _validate(teams, slot)
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    return [
        pick_number(r, slot, teams, draft_type, third_round_reversal)
        for r in range(1, rounds + 1)
    ]


def next_pick(
    current_overall: int,
    slot: int,
    teams: int,
    rounds: int,
    draft_type: DraftType = "snake",
    third_round_reversal: bool = False,
) -> int | None:
    """Our next pick strictly after ``current_overall``, or ``None`` if the draft ends.

    ``current_overall`` does not need to be one of our picks — this answers "given the
    board is at pick N, when do we choose next?". If pick N *is* ours, the answer is our
    following turn, which is what the two-pick optimizer needs.
    """
    ours = picks_for_slot(slot, teams, rounds, draft_type, third_round_reversal)
    return next((p for p in ours if p > current_overall), None)


def current_pick_for_slot(
    current_overall: int,
    slot: int,
    teams: int,
    rounds: int,
    draft_type: DraftType = "snake",
    third_round_reversal: bool = False,
) -> int | None:
    """Our first pick at or after ``current_overall``, or ``None`` if the draft ends."""
    ours = picks_for_slot(slot, teams, rounds, draft_type, third_round_reversal)
    return next((p for p in ours if p >= current_overall), None)


def picks_until_next_turn(
    current_overall: int,
    slot: int,
    teams: int,
    rounds: int,
    draft_type: DraftType = "snake",
    third_round_reversal: bool = False,
) -> int | None:
    """How many selections other managers make between our pick and our next one.

    If ``current_overall`` is our pick at 1.07 in a 12-team snake, our next is 2.06
    (overall 18), so 10 other managers pick in between.
    """
    following = next_pick(
        current_overall, slot, teams, rounds, draft_type, third_round_reversal
    )
    if following is None:
        return None
    return following - current_overall - 1


@dataclass(frozen=True, slots=True)
class SnakeBoard:
    """Bound pick arithmetic for one league, so callers stop passing five arguments."""

    teams: int
    rounds: int
    draft_type: DraftType = "snake"
    third_round_reversal: bool = False

    def __post_init__(self) -> None:
        _validate(self.teams)
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")

    @property
    def total_picks(self) -> int:
        return self.teams * self.rounds

    def direction(self, round_: int) -> int:
        return round_direction(round_, self.draft_type, self.third_round_reversal)

    def pick_number(self, round_: int, slot: int) -> int:
        return pick_number(round_, slot, self.teams, self.draft_type, self.third_round_reversal)

    def round_for(self, overall: int) -> int:
        return round_for_pick(overall, self.teams)

    def slot_for(self, overall: int) -> int:
        return slot_for_pick(overall, self.teams, self.draft_type, self.third_round_reversal)

    def label(self, overall: int) -> str:
        return pick_label(overall, self.teams)

    def picks_for(self, slot: int) -> list[int]:
        return picks_for_slot(
            slot, self.teams, self.rounds, self.draft_type, self.third_round_reversal
        )

    def next_pick(self, current_overall: int, slot: int) -> int | None:
        return next_pick(
            current_overall, slot, self.teams, self.rounds,
            self.draft_type, self.third_round_reversal,
        )

    def picks_until_next(self, current_overall: int, slot: int) -> int | None:
        return picks_until_next_turn(
            current_overall, slot, self.teams, self.rounds,
            self.draft_type, self.third_round_reversal,
        )

    def slots_between(self, start_overall: int, end_overall: int) -> list[tuple[int, int]]:
        """``(overall, slot)`` for every pick strictly between the two, in order.

        This is exactly the set of managers whose behaviour determines whether a player
        survives to our next selection.
        """
        lo = max(1, start_overall + 1)
        hi = min(self.total_picks, end_overall - 1)
        return [(p, self.slot_for(p)) for p in range(lo, hi + 1)]

    @classmethod
    def from_league(cls, league: object) -> SnakeBoard:
        """Build from a :class:`~fantasy_draft.config.LeagueConfig`."""
        draft = league.draft  # type: ignore[attr-defined]
        return cls(
            teams=league.teams,  # type: ignore[attr-defined]
            rounds=draft.rounds,
            draft_type=draft.type,
            third_round_reversal=draft.third_round_reversal,
        )
