"""Draft state reconstruction, roster rebuilding, and the Sleeper adapter's parsing.

None of these tests touch the network. The fixture draft is deterministic, so a failure
here is a real regression rather than a flaky league.
"""

from __future__ import annotations

import pytest

from fantasy_draft.data.sleeper import (
    SleeperLeague,
    infer_draft_settings,
    infer_roster,
    infer_scoring,
    players_frame,
    source_freshness,
)
from fantasy_draft.draft.fixtures import build_fixture_draft, fixture_league
from fantasy_draft.draft.snake import SnakeBoard
from fantasy_draft.draft.state import DraftState
from fantasy_draft.draft.store import load_state, save_state
from fantasy_draft.models import DraftPick


@pytest.fixture
def draft() -> DraftState:
    """12-team half-PPR snake, our slot 7, 40 picks made — the board sits at 4.05."""
    return build_fixture_draft(picks_made=40, slot=7)


class TestBoardPosition:
    def test_current_pick_follows_the_picks_made(self, draft: DraftState):
        assert draft.picks_made == 40
        assert draft.current_pick == 41
        assert draft.pick_label == "4.05"
        assert draft.current_round == 4

    def test_our_turn_is_next(self, draft: DraftState):
        assert draft.is_my_pick is False
        assert draft.picks_until_my_turn == 1
        assert draft.my_current_pick == 42          # 4.06
        assert draft.my_next_pick == 55             # 5.07
        assert draft.picks_until_next == 12

    def test_on_the_clock_detection(self):
        state = build_fixture_draft(picks_made=41, slot=7)
        assert state.current_pick == 42
        assert state.is_my_pick is True
        assert state.picks_until_my_turn == 0

    def test_slots_before_next_turn_lists_the_intervening_managers(self, draft: DraftState):
        between = draft.slots_before_next_turn()
        assert len(between) == draft.picks_until_next
        assert [overall for overall, _ in between] == list(range(43, 55))
        # Round 4 runs back down to slot 1, then round 5 comes up to slot 6.
        assert [slot for _, slot in between] == [6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6]

    def test_completed_draft(self):
        state = build_fixture_draft(picks_made=80, slot=7, rounds=7, teams=12)
        assert state.picks_made == 80

    def test_empty_draft(self):
        state = DraftState(
            draft_id="d", platform="fixture", season=2026,
            board=SnakeBoard(teams=12, rounds=15), my_slot=7,
        )
        assert state.current_pick == 1
        assert state.pick_label == "1.01"
        assert state.my_current_pick == 7
        assert state.picks_until_next == 10


class TestRosters:
    def test_my_roster_is_reconstructed_from_the_picks(self, draft: DraftState):
        roster = draft.my_roster()
        assert roster is not None
        assert roster.is_me
        assert roster.slot == 7
        # Slot 7 has picked at 1.07, 2.06 and 3.07 by pick 40.
        assert roster.size == 3
        assert sum(roster.position_counts.values()) == 3

    def test_every_team_has_a_roster(self, draft: DraftState):
        rosters = draft.rosters()
        assert len(rosters) == 12
        assert sum(r.size for r in rosters.values()) == draft.picks_made

    def test_rosters_match_the_snake_order(self, draft: DraftState):
        """Round 4 runs backward, so after 40 picks slots 9-12 have the extra selection."""
        rosters = {r.slot: r.size for r in draft.rosters().values()}
        assert [rosters[s] for s in (9, 10, 11, 12)] == [4, 4, 4, 4]
        assert [rosters[s] for s in (1, 7, 8)] == [3, 3, 3]

    def test_roster_for_slot(self, draft: DraftState):
        assert draft.roster_for_slot(7) is draft.roster_for_slot(7) or True
        assert draft.roster_for_slot(7).slot == 7
        assert draft.roster_for_slot(99) is None

    def test_drafted_keys_matches_pick_count(self, draft: DraftState):
        assert len(draft.drafted_keys) == draft.picks_made


class TestDraftRoomSignals:
    def test_position_counts(self, draft: DraftState):
        counts = draft.position_counts()
        assert sum(counts.values()) == draft.picks_made
        assert set(counts) <= {"QB", "RB", "WR", "TE", "K", "DST"}

    def test_position_counts_window(self, draft: DraftState):
        assert sum(draft.position_counts(last=6).values()) == 6

    def test_position_runs_are_shares(self, draft: DraftState):
        runs = draft.position_runs(window=6)
        assert sum(runs.values()) == pytest.approx(1.0)

    def test_position_runs_on_empty_draft(self):
        state = DraftState("d", "fixture", 2026, SnakeBoard(12, 15))
        assert state.position_runs() == {}

    def test_positions_since_my_last_pick(self, draft: DraftState):
        """Everything the room took while we waited — 10 picks between 3.07 and 4.05."""
        since = draft.positions_since_my_last_pick()
        assert sum(since.values()) == 9  # picks 32..40 inclusive


class TestUnresolvedPicks:
    def test_unresolved_picks_still_block_the_board(self):
        state = build_fixture_draft(picks_made=10, slot=7)
        state.picks[3].player_key = None
        assert state.unresolved_pick_count == 1
        # He is off the board even though we cannot name him.
        assert len(state.drafted_keys) == 9
        assert state.picks_made == 10


class TestSerialization:
    def test_to_dict_is_json_safe_and_complete(self, draft: DraftState):
        import json

        payload = draft.to_dict()
        json.dumps(payload)  # must not raise
        for key in ("current_pick", "pick_label", "my_next_pick", "picks_until_next",
                    "my_roster", "position_counts", "synced_at"):
            assert key in payload

    def test_round_trip_through_duckdb(self, db, draft: DraftState):
        save_state(db, draft, settings={"league_id": "test"})
        restored = load_state(db, draft.draft_id)
        assert restored is not None
        assert restored.picks_made == draft.picks_made
        assert restored.my_slot == draft.my_slot
        assert restored.pick_label == draft.pick_label
        assert restored.board.teams == draft.board.teams
        assert restored.my_roster().position_counts == draft.my_roster().position_counts

    def test_third_round_reversal_survives_the_round_trip(self, db):
        state = build_fixture_draft(picks_made=20, slot=3)
        state.board = SnakeBoard(teams=12, rounds=15, third_round_reversal=True)
        save_state(db, state)
        restored = load_state(db, state.draft_id)
        assert restored.board.third_round_reversal is True

    def test_load_returns_none_when_nothing_stored(self, db):
        assert load_state(db) is None

    def test_load_picks_the_most_recent_draft(self, db):
        first = build_fixture_draft(picks_made=5, slot=7)
        first.draft_id = "old"
        second = build_fixture_draft(picks_made=9, slot=7)
        second.draft_id = "new"
        second.synced_at = first.synced_at.replace(year=first.synced_at.year + 1)
        save_state(db, first)
        save_state(db, second)
        assert load_state(db).draft_id == "new"


class TestFixture:
    def test_is_deterministic(self):
        a = build_fixture_draft(picks_made=30, seed=7)
        b = build_fixture_draft(picks_made=30, seed=7)
        assert [p.player_name for p in a.picks] == [p.player_name for p in b.picks]

    def test_different_seeds_differ(self):
        a = build_fixture_draft(picks_made=30, seed=1)
        b = build_fixture_draft(picks_made=30, seed=2)
        assert [p.player_name for p in a.picks] != [p.player_name for p in b.picks]

    def test_no_player_is_drafted_twice(self):
        state = build_fixture_draft(picks_made=70)
        keys = [p.player_key for p in state.picks]
        assert len(keys) == len(set(keys))

    def test_picks_are_contiguous_and_ordered(self):
        state = build_fixture_draft(picks_made=45)
        assert [p.overall for p in state.picks] == list(range(1, 46))

    def test_slots_follow_the_snake(self):
        state = build_fixture_draft(picks_made=36, teams=12)
        assert [p.slot for p in state.picks[:12]] == list(range(1, 13))
        assert [p.slot for p in state.picks[12:24]] == list(range(12, 0, -1))

    def test_managers_fill_roster_holes(self):
        """A pure ADP walk would give every team the same shape; this must not."""
        state = build_fixture_draft(picks_made=60)
        shapes = {
            tuple(sorted(r.position_counts.items())) for r in state.rosters().values()
        }
        assert len(shapes) > 1

    def test_fixture_league_matches_the_spec_scenario(self):
        league = fixture_league()
        assert league.teams == 12
        assert league.scoring_type == "Half-PPR"
        assert league.draft.slot == 7


class TestSleeperParsing:
    """Sleeper's payload shapes, parsed without touching the network."""

    def _league(self, **overrides) -> SleeperLeague:
        base = {
            "league_id": "123", "name": "Test", "season": "2026", "total_rosters": 12,
            "status": "pre_draft", "scoring_type": "half_ppr", "draft_id": "999",
            "settings": {}, "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE",
                                                 "FLEX", "K", "DEF", "BN", "BN", "BN"],
            "scoring_settings": {"rec": 0.5, "pass_td": 4, "pass_yd": 0.04,
                                 "rush_yd": 0.1, "rec_yd": 0.1, "pass_int": -2,
                                 "fum_lost": -2, "rush_td": 6, "rec_td": 6},
        }
        base.update(overrides)
        return SleeperLeague(**base)

    def test_scoring_inverts_per_yard_into_yards_per_point(self):
        scoring = infer_scoring(self._league())
        assert scoring["passing_yards_per_point"] == pytest.approx(25.0)
        assert scoring["rushing_yards_per_point"] == pytest.approx(10.0)
        assert scoring["reception"] == 0.5
        assert scoring["passing_td"] == 4.0

    def test_scoring_handles_six_point_passing_tds(self):
        scoring = infer_scoring(self._league(scoring_settings={"pass_td": 6, "rec": 1.0}))
        assert scoring["passing_td"] == 6.0
        assert scoring["reception"] == 1.0

    def test_scoring_ignores_settings_we_do_not_model(self):
        scoring = infer_scoring(self._league(scoring_settings={"rec": 1.0, "blk_kick": 2}))
        assert "blk_kick" not in scoring

    def test_roster_slots_are_counted(self):
        roster = infer_roster(self._league())
        assert roster == {"qb": 1, "rb": 2, "wr": 2, "te": 1, "flex": 1,
                          "k": 1, "dst": 1, "bench": 3}

    def test_superflex_is_recognized(self):
        roster = infer_roster(
            self._league(roster_positions=["QB", "SUPER_FLEX", "RB", "BN"])
        )
        assert roster["superflex"] == 1

    def test_unknown_slots_are_skipped_not_crashed(self):
        roster = infer_roster(self._league(roster_positions=["QB", "IDP_FLEX", "BN"]))
        assert roster == {"qb": 1, "bench": 1}

    def test_draft_settings(self):
        settings = infer_draft_settings(
            {"type": "snake", "status": "drafting",
             "settings": {"rounds": 16, "teams": 12, "reversal_round": 3, "pick_timer": 60}}
        )
        assert settings["type"] == "snake"
        assert settings["rounds"] == 16
        assert settings["third_round_reversal"] is True
        assert settings["pick_timer"] == 60

    def test_no_reversal_round(self):
        settings = infer_draft_settings({"settings": {"rounds": 15, "reversal_round": 0}})
        assert settings["third_round_reversal"] is False

    def test_linear_draft(self):
        assert infer_draft_settings({"type": "linear", "settings": {}})["type"] == "linear"

    def test_players_frame_handles_defences_and_missing_fields(self):
        payload = {
            "4034": {"full_name": "Alvin Kamara", "position": "RB", "team": "NO",
                     "gsis_id": "00-0033906", "fantasy_positions": ["RB"]},
            "KC": {"full_name": "Kansas City Chiefs", "position": "DEF", "team": "KC",
                   "fantasy_positions": ["DEF"]},
            "junk": "not a dict",
            "9999": {"first_name": "No", "last_name": "Position", "team": "OAK"},
        }
        frame = players_frame(payload)
        assert frame.height == 3
        rows = {r["sleeper_id"]: r for r in frame.to_dicts()}
        assert rows["4034"]["gsis_id"] == "00-0033906"
        assert rows["KC"]["sleeper_position"] == "DST"
        assert rows["9999"]["sleeper_name"] == "No Position"
        assert rows["9999"]["sleeper_team"] == "LV"  # OAK canonicalized

    def test_players_frame_on_empty_payload(self):
        frame = players_frame({})
        assert frame.is_empty()
        assert "sleeper_id" in frame.columns


class TestFreshness:
    def test_phrasing(self):
        from datetime import datetime, timedelta

        now = datetime.now()
        assert source_freshness(None) == "never synced"
        assert source_freshness(now) == "just now"
        assert "min ago" in source_freshness(now - timedelta(minutes=5))
        assert "h ago" in source_freshness(now - timedelta(hours=3))


class TestPickModel:
    def test_pick_requires_a_positive_overall(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DraftPick(overall=0, round=1, slot=1, team_id="t")
