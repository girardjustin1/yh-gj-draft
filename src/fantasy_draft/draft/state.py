"""The canonical ``DraftState``.

Everything the recommendation engine knows about the live board lives here, and every
question it needs answered — whose pick is it, who is gone, what does my roster look
like, how many selections until my next turn — is a property of this object rather than
something each caller recomputes.

``DraftState`` is deliberately platform-agnostic and fully reconstructible from a saved
fixture, so the simulator and the tests never need a live Sleeper league.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..config import LeagueConfig
from ..models import DraftPick, RosterSnapshot
from .snake import SnakeBoard


@dataclass(slots=True)
class DraftState:
    """A point-in-time view of one draft."""

    draft_id: str
    platform: str
    season: int
    board: SnakeBoard
    picks: list[DraftPick] = field(default_factory=list)
    slot_to_team: dict[int, str] = field(default_factory=dict)
    team_names: dict[str, str] = field(default_factory=dict)
    my_slot: int | None = None
    my_team_id: str | None = None
    status: str = "unknown"
    synced_at: datetime = field(default_factory=datetime.now)

    # --- board position ---------------------------------------------------------------

    @property
    def picks_made(self) -> int:
        return len(self.picks)

    @property
    def is_complete(self) -> bool:
        return self.picks_made >= self.board.total_picks

    @property
    def current_pick(self) -> int:
        """The overall pick now on the clock (one past the last selection made)."""
        return min(self.picks_made + 1, self.board.total_picks)

    @property
    def current_round(self) -> int:
        return self.board.round_for(self.current_pick)

    @property
    def current_slot(self) -> int:
        return self.board.slot_for(self.current_pick)

    @property
    def pick_label(self) -> str:
        return self.board.label(self.current_pick)

    @property
    def on_the_clock_team(self) -> str | None:
        return self.slot_to_team.get(self.current_slot)

    @property
    def is_my_pick(self) -> bool:
        return self.my_slot is not None and self.current_slot == self.my_slot

    # --- our turns ---------------------------------------------------------------------

    @property
    def my_picks(self) -> list[int]:
        return self.board.picks_for(self.my_slot) if self.my_slot else []

    @property
    def my_current_pick(self) -> int | None:
        """Our first pick at or after the current one."""
        if self.my_slot is None:
            return None
        return next((p for p in self.my_picks if p >= self.current_pick), None)

    @property
    def my_next_pick(self) -> int | None:
        """Our pick *after* the one we are making now."""
        anchor = self.my_current_pick
        if anchor is None:
            return None
        return next((p for p in self.my_picks if p > anchor), None)

    @property
    def picks_until_next(self) -> int | None:
        """How many other managers select between our current turn and our next one."""
        anchor, following = self.my_current_pick, self.my_next_pick
        if anchor is None or following is None:
            return None
        return following - anchor - 1

    @property
    def picks_until_my_turn(self) -> int | None:
        """How many selections happen before we are on the clock. 0 means now."""
        anchor = self.my_current_pick
        return None if anchor is None else anchor - self.current_pick

    def slots_before_next_turn(self) -> list[tuple[int, int]]:
        """``(overall, slot)`` for every pick between our current turn and our next.

        This is exactly the set of managers whose choices decide whether a player
        survives to us, and it is what the survival model iterates over.
        """
        anchor, following = self.my_current_pick, self.my_next_pick
        if anchor is None or following is None:
            return []
        return self.board.slots_between(anchor, following)

    # --- who is gone --------------------------------------------------------------------

    @property
    def drafted_keys(self) -> set[str]:
        return {p.player_key for p in self.picks if p.player_key}

    @property
    def unresolved_pick_count(self) -> int:
        """Picks we could not map to a canonical player — they still block the board."""
        return sum(1 for p in self.picks if not p.player_key)

    def recent_picks(self, count: int = 12) -> list[DraftPick]:
        return self.picks[-count:] if count else []

    def position_counts(self, last: int | None = None) -> dict[str, int]:
        """How many of each position have been taken, optionally over the last N picks."""
        picks = self.picks[-last:] if last else self.picks
        return dict(Counter(p.position for p in picks if p.position))

    def positions_since_my_last_pick(self) -> dict[str, int]:
        """What the room took while we were waiting."""
        if self.my_slot is None:
            return self.position_counts(12)
        mine = [p.overall for p in self.picks if p.slot == self.my_slot]
        if not mine:
            return self.position_counts()
        return dict(
            Counter(p.position for p in self.picks if p.overall > mine[-1] and p.position)
        )

    # --- rosters ------------------------------------------------------------------------

    def rosters(self) -> dict[str, RosterSnapshot]:
        """Reconstruct every team's roster from the pick history."""
        snapshots: dict[str, RosterSnapshot] = {}
        for slot, team_id in self.slot_to_team.items():
            snapshots[team_id] = RosterSnapshot(
                team_id=team_id, slot=slot, is_me=(team_id == self.my_team_id)
            )
        for pick in self.picks:
            team_id = pick.team_id or self.slot_to_team.get(pick.slot)
            if team_id is None:
                continue
            snapshot = snapshots.get(team_id)
            if snapshot is None:
                snapshot = RosterSnapshot(
                    team_id=team_id, slot=pick.slot, is_me=(team_id == self.my_team_id)
                )
                snapshots[team_id] = snapshot
            if pick.player_key:
                snapshot.player_keys.append(pick.player_key)
            if pick.position:
                snapshot.positions.append(pick.position)
        return snapshots

    def my_roster(self) -> RosterSnapshot | None:
        if self.my_team_id is None and self.my_slot is None:
            return None
        rosters = self.rosters()
        if self.my_team_id and self.my_team_id in rosters:
            return rosters[self.my_team_id]
        return next((r for r in rosters.values() if r.slot == self.my_slot), None)

    def roster_for_slot(self, slot: int) -> RosterSnapshot | None:
        team_id = self.slot_to_team.get(slot)
        if team_id is None:
            return None
        return self.rosters().get(team_id)

    # --- draft-room behaviour ------------------------------------------------------------

    def position_runs(self, window: int = 6) -> dict[str, float]:
        """Share of the last ``window`` picks spent on each position.

        A raw observation, not a recommendation: a run can mean the position is scarce,
        or that the room overdrafted it and pushed value to us at another position.
        Interpreting it is :mod:`fantasy_draft.analytics.draft_room`'s job.
        """
        recent = [p.position for p in self.picks[-window:] if p.position]
        if not recent:
            return {}
        counts = Counter(recent)
        return {position: count / len(recent) for position, count in counts.items()}

    # --- serialization ---------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Compact, JSON-safe summary — the shape the Claude tool layer will return."""
        roster = self.my_roster()
        return {
            "draft_id": self.draft_id,
            "platform": self.platform,
            "season": self.season,
            "status": self.status,
            "teams": self.board.teams,
            "rounds": self.board.rounds,
            "draft_type": self.board.draft_type,
            "third_round_reversal": self.board.third_round_reversal,
            "my_slot": self.my_slot,
            "picks_made": self.picks_made,
            "current_pick": self.current_pick,
            "pick_label": self.pick_label,
            "current_round": self.current_round,
            "is_my_pick": self.is_my_pick,
            "my_next_pick": self.my_next_pick,
            "picks_until_next": self.picks_until_next,
            "my_roster": {
                "players": roster.player_keys if roster else [],
                "positions": roster.position_counts if roster else {},
            },
            "position_counts": self.position_counts(),
            "recent_positions": [p.position for p in self.recent_picks(12)],
            "unresolved_picks": self.unresolved_pick_count,
            "synced_at": self.synced_at.isoformat(),
        }

    @classmethod
    def from_league(
        cls, league: LeagueConfig, draft_id: str, platform: str = "manual", **kwargs: Any
    ) -> DraftState:
        """Build an empty state from a league configuration. Used by fixtures and tests."""
        return cls(
            draft_id=draft_id,
            platform=platform,
            season=league.season,
            board=SnakeBoard.from_league(league),
            my_slot=league.draft.slot,
            **kwargs,
        )
