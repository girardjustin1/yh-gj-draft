"""The orchestration layer: analyze_current_pick, the five areas, and lineup filling.

These guard the contract the CLI, the web API and Claude all share. If they diverge,
the GUI and the assistant start disagreeing mid-draft, which is worse than either being
slightly wrong on its own.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from fantasy_draft.analytics.lineup_value import (
    best_lineup_points,
    lineup_upgrades,
    positional_needs,
)
from fantasy_draft.analytics.outcomes import add_outcome_range, outcome_label
from fantasy_draft.config import LeagueConfig
from fantasy_draft.draft.fixtures import build_fixture_draft
from fantasy_draft.draft.providers import (
    PROVIDERS,
    ProviderNotImplemented,
    YahooDraftProvider,
    provider_for,
    provider_status,
)
from fantasy_draft.draft.store import (
    create_manual_draft,
    load_state,
    record_pick,
    save_state,
    undo_pick,
)
from fantasy_draft.models import RosterSnapshot
from fantasy_draft.service import (
    NoDraftError,
    UnknownSlotError,
    analyze_current_pick,
    fill_lineup,
    lineup_slots,
)

# --- outcome range ----------------------------------------------------------------------


def _curve() -> pl.DataFrame:
    rows = []
    for position, top in (("RB", 300.0), ("WR", 280.0)):
        for rank in range(1, 61):
            rows.append(
                {
                    "position": position,
                    "rank": rank,
                    "points": top - 3.5 * (rank - 1),
                    "points_sd": 18.0,
                    "seasons_used": 4,
                }
            )
    return pl.DataFrame(rows)


def _board(**overrides) -> pl.DataFrame:
    base = {
        "player_key": ["a", "b", "c"],
        "position": ["RB", "RB", "WR"],
        "positional_rank": [1, 20, 10],
        "projected_points": [300.0, 233.5, 248.5],
        "overall_ecr": [2.0, 55.0, 25.0],
        "ecr_sd": [1.0, 14.0, 6.0],
        "ecr_best": [1, 30, 15],
        "ecr_worst": [4, 90, 40],
        "risk_injury": [0.0, 0.0, 0.0],
        "risk_age": [0.0, 0.0, 0.0],
        "risk_sample": [0.0, 0.5, 0.2],
    }
    base.update(overrides)
    return pl.DataFrame(base)


class TestOutcomeRange:
    def test_ordering_holds(self, tmp_config):
        out = add_outcome_range(tmp_config, _board(), _curve())
        assert (out["floor_points"] <= out["median_points"]).all()
        assert (out["median_points"] <= out["ceiling_points"]).all()

    def test_median_tracks_the_consensus_projection(self, tmp_config):
        out = add_outcome_range(tmp_config, _board(), _curve())
        for projected, median in zip(
            out["projected_points"], out["median_points"], strict=True
        ):
            assert median == pytest.approx(projected, abs=0.01)

    def test_more_expert_disagreement_widens_the_range(self, tmp_config):
        tight = add_outcome_range(tmp_config, _board(ecr_sd=[1.0, 1.0, 1.0]), _curve())
        loose = add_outcome_range(tmp_config, _board(ecr_sd=[1.0, 40.0, 6.0]), _curve())
        assert loose["outcome_range"][1] > tight["outcome_range"][1]

    def test_injury_risk_lowers_the_floor_without_lowering_the_ceiling(self, tmp_config):
        healthy = add_outcome_range(tmp_config, _board(), _curve())
        hurt = add_outcome_range(
            tmp_config, _board(risk_injury=[0.9, 0.0, 0.0]), _curve()
        )
        assert hurt["floor_points"][0] < healthy["floor_points"][0]
        assert hurt["ceiling_points"][0] == pytest.approx(healthy["ceiling_points"][0])

    def test_an_unproven_player_gets_more_upside(self, tmp_config):
        proven = add_outcome_range(tmp_config, _board(risk_sample=[0.0, 0.0, 0.0]), _curve())
        rookie = add_outcome_range(tmp_config, _board(risk_sample=[1.0, 0.0, 0.0]), _curve())
        assert rookie["ceiling_points"][0] > proven["ceiling_points"][0]
        assert rookie["upside_skew"][0] > proven["upside_skew"][0]

    def test_floor_is_never_negative(self, tmp_config):
        out = add_outcome_range(
            tmp_config,
            _board(projected_points=[8.0, 5.0, 3.0], ecr_sd=[60.0, 60.0, 60.0]),
            _curve(),
        )
        assert (out["floor_points"] >= 0).all()

    def test_no_market_data_is_maximally_uncertain(self, tmp_config):
        known = add_outcome_range(tmp_config, _board(), _curve())
        unknown = add_outcome_range(
            tmp_config,
            _board(ecr_sd=[0.0, 0.0, 0.0], ecr_best=[0, 0, 0], ecr_worst=[0, 0, 0]),
            _curve(),
        )
        assert unknown["outcome_range"][2] > known["outcome_range"][2]
        assert unknown["outcome_confidence"][2] < known["outcome_confidence"][2]

    def test_missing_curve_degrades_rather_than_raising(self, tmp_config):
        out = add_outcome_range(tmp_config, _board(), pl.DataFrame())
        assert out["outcome_confidence"].max() == 0.0
        assert "floor_points" in out.columns

    def test_empty_board(self, tmp_config):
        empty = pl.DataFrame(schema={"position": pl.Utf8, "positional_rank": pl.Int32,
                                     "projected_points": pl.Float64})
        assert add_outcome_range(tmp_config, empty, _curve()).is_empty()

    @pytest.mark.parametrize(
        ("floor", "median", "ceiling", "expected"),
        [
            (195, 200, 205, "safe"),
            (100, 200, 400, "boom or bust"),
            (150, 200, 220, "floor play"),
        ],
    )
    def test_labels(self, floor, median, ceiling, expected):
        row = {
            "floor_points": floor, "median_points": median, "ceiling_points": ceiling,
            "upside_skew": (ceiling - median) - (median - floor),
        }
        assert outcome_label(row) == expected

    def test_label_handles_missing_data(self):
        assert outcome_label({}) == "unknown range"


# --- lineup ------------------------------------------------------------------------------


class TestLineup:
    def test_slot_order(self, league: LeagueConfig):
        assert lineup_slots(league.roster) == [
            "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DST"
        ]

    def test_superflex_appears(self, superflex_league: LeagueConfig):
        assert "SUPERFLEX" in lineup_slots(superflex_league.roster)

    def test_empty_roster_shows_every_hole(self, league: LeagueConfig):
        slots = fill_lineup(league.roster, None, {})
        assert all(not s.filled for s in slots)
        assert len(slots) == league.roster.starters

    def test_dedicated_slots_fill_before_flex(self, league: LeagueConfig):
        """A flex slot must not swallow our only tight end."""
        roster = RosterSnapshot(
            team_id="t", slot=1, player_keys=["k1", "k2"], positions=["TE", "RB"]
        )
        lookup = {
            "k1": {"name": "Tight End", "position": "TE", "team": "KC"},
            "k2": {"name": "Back", "position": "RB", "team": "SF"},
        }
        slots = fill_lineup(league.roster, roster, lookup)
        by_slot = {s.slot: s for s in slots if s.filled}
        assert by_slot["TE"].name == "Tight End"
        assert by_slot["RB"].name == "Back"

    def test_surplus_goes_to_the_flex_then_the_bench(self, league: LeagueConfig):
        positions = ["RB"] * 5
        roster = RosterSnapshot(
            team_id="t", slot=1,
            player_keys=[f"k{i}" for i in range(5)], positions=positions,
        )
        lookup = {
            f"k{i}": {"name": f"RB{i}", "position": "RB", "team": "KC"} for i in range(5)
        }
        slots = fill_lineup(league.roster, roster, lookup)
        starters = [s for s in slots if not s.is_bench]
        bench = [s for s in slots if s.is_bench]
        assert sum(1 for s in starters if s.filled) == 3   # RB, RB, FLEX
        assert len(bench) == 2

    def test_no_player_appears_twice(self, league: LeagueConfig):
        roster = RosterSnapshot(
            team_id="t", slot=1, player_keys=["a", "b", "c"],
            positions=["RB", "WR", "TE"],
        )
        lookup = {k: {"name": k, "position": p, "team": "KC"}
                  for k, p in zip("abc", ["RB", "WR", "TE"], strict=True)}
        slots = fill_lineup(league.roster, roster, lookup)
        keys = [s.player_key for s in slots if s.filled]
        assert len(keys) == len(set(keys)) == 3


# --- providers ---------------------------------------------------------------------------


class TestProviders:
    def test_registry_covers_the_config_options(self):
        assert {"sleeper", "yahoo", "manual"} <= set(PROVIDERS)

    def test_yahoo_raises_with_instructions_not_silence(self, tmp_config, db):
        cfg = tmp_config.model_copy(
            update={"league": tmp_config.league.model_copy(update={"platform": "yahoo"})}
        )
        provider = provider_for(cfg, db)
        assert isinstance(provider, YahooDraftProvider)
        with pytest.raises(ProviderNotImplemented) as exc:
            provider.fetch_state("x")
        assert "OAuth" in str(exc.value)
        assert "HUMAN_TODO" in str(exc.value)

    def test_status_reports_honestly(self, tmp_config, db):
        status = provider_status(tmp_config, db)
        assert status["platform"] == "sleeper"
        assert status["implemented"] is True
        assert status["connected"] is False   # nothing configured in a temp db

    def test_yahoo_status_is_marked_unimplemented(self, tmp_config, db):
        cfg = tmp_config.model_copy(
            update={"league": tmp_config.league.model_copy(update={"platform": "yahoo"})}
        )
        assert provider_status(cfg, db)["implemented"] is False

    def test_unknown_platform_names_the_alternatives(self, tmp_config, db):
        cfg = tmp_config.model_copy(
            update={"league": tmp_config.league.model_copy(update={"platform": "espn"})}
        )
        with pytest.raises(ProviderNotImplemented, match="sleeper"):
            provider_for(cfg, db)


# --- manual draft entry --------------------------------------------------------------------


class TestManualDraft:
    def test_recorded_picks_are_indistinguishable_from_synced_ones(self, db, league):
        state = create_manual_draft(db, league, draft_id="m1")
        assert state.picks_made == 0
        record_pick(db, state, "k1", "Player One", "RB", "KC")
        record_pick(db, state, "k2", "Player Two", "WR", "SF")
        restored = load_state(db, "m1")
        assert restored.picks_made == 2
        assert restored.pick_label == "1.03"
        assert restored.picks[0].slot == 1
        assert restored.picks[1].slot == 2

    def test_slots_follow_the_snake(self, db, league):
        state = create_manual_draft(db, league, draft_id="m2")
        for i in range(14):
            record_pick(db, state, f"k{i}", f"P{i}", "RB", "KC")
        assert [p.slot for p in state.picks[:12]] == list(range(1, 13))
        assert state.picks[12].slot == 12   # round 2 reverses
        assert state.picks[13].slot == 11

    def test_duplicate_player_is_refused(self, db, league):
        state = create_manual_draft(db, league, draft_id="m3")
        record_pick(db, state, "k1", "Player One", "RB", "KC")
        with pytest.raises(ValueError, match="already been drafted"):
            record_pick(db, state, "k1", "Player One", "RB", "KC")

    def test_cannot_overfill_the_draft(self, db):
        small = LeagueConfig.model_validate(
            {"teams": 2, "draft": {"rounds": 1, "slot": 1},
             "roster": {"bench": 0, "k": 0, "dst": 0, "flex": 0, "rb": 1, "wr": 0, "te": 0}}
        )
        state = create_manual_draft(db, small, draft_id="m4")
        record_pick(db, state, "a", "A", "RB", "KC")
        record_pick(db, state, "b", "B", "RB", "SF")
        with pytest.raises(ValueError, match="complete"):
            record_pick(db, state, "c", "C", "RB", "NE")

    def test_undo_removes_the_last_pick(self, db, league):
        state = create_manual_draft(db, league, draft_id="m5")
        record_pick(db, state, "k1", "One", "RB", "KC")
        record_pick(db, state, "k2", "Two", "WR", "SF")
        removed = undo_pick(db, state)
        assert removed.player_name == "Two"
        assert load_state(db, "m5").picks_made == 1

    def test_undo_on_an_empty_draft(self, db, league):
        state = create_manual_draft(db, league, draft_id="m6")
        assert undo_pick(db, state) is None


# --- analyze_current_pick ------------------------------------------------------------------


class TestAnalyzeCurrentPick:
    def test_no_draft_raises_an_actionable_error(self, tmp_config, db):
        with pytest.raises(NoDraftError, match="ff draft mock"):
            analyze_current_pick(tmp_config, db, refresh=False)

    def test_unknown_slot_raises_an_actionable_error(self, tmp_config, db):
        state = build_fixture_draft(picks_made=10, slot=7)
        state.my_slot = None
        save_state(db, state)
        with pytest.raises(UnknownSlotError, match="draft.slot"):
            analyze_current_pick(tmp_config, db, refresh=False)

    def test_a_failed_sync_falls_back_and_says_so(self, tmp_config, db):
        """Losing the platform must not lose the pick."""
        save_state(db, build_fixture_draft(picks_made=41, slot=7))
        analysis = analyze_current_pick(tmp_config, db, refresh=True)
        assert analysis.synced is False
        assert analysis.sync_error is not None
        assert analysis.recommendation.pick_label == "4.06"

    def test_the_five_areas_are_all_present(self, tmp_config, db):
        save_state(db, build_fixture_draft(picks_made=41, slot=7))
        analysis = analyze_current_pick(tmp_config, db, refresh=False)
        payload = analysis.to_dict()
        for area in ("on_the_clock", "best_available", "my_roster",
                     "who_makes_it_back", "what_if"):
            assert area in payload

    def test_payload_is_json_serializable(self, tmp_config, db):
        save_state(db, build_fixture_draft(picks_made=41, slot=7))
        analysis = analyze_current_pick(tmp_config, db, refresh=False)
        json.dumps(analysis.to_dict())   # must not raise

    def test_on_the_clock_carries_the_snake_maths(self, tmp_config, db):
        save_state(db, build_fixture_draft(picks_made=41, slot=7))
        area = analyze_current_pick(tmp_config, db, refresh=False).on_the_clock()
        assert area["pick_label"] == "4.06"
        assert area["next_pick_label"] == "5.07"
        assert area["picks_until_next"] == 12
        assert area["my_slot"] == 7

    def test_roster_area_reports_the_holes(self, tmp_config, db):
        save_state(db, build_fixture_draft(picks_made=41, slot=7))
        roster = analyze_current_pick(tmp_config, db, refresh=False).my_roster()
        assert len(roster["starters"]) == tmp_config.league.roster.starters
        assert roster["unfilled_starters"]

    def test_empty_database_degrades_with_warnings(self, tmp_config, db):
        """No projections ingested — must warn, not crash."""
        save_state(db, build_fixture_draft(picks_made=41, slot=7))
        analysis = analyze_current_pick(tmp_config, db, refresh=False)
        assert analysis.recommendation.warnings
        assert analysis.on_the_clock()["pick_label"] == "4.06"


# --- marginal lineup value ------------------------------------------------------------


class TestLineupValue:
    """The honest answer to "what do I still need?" — value added to the starting eleven."""

    def _board(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "player_key": ["rb1", "rb2", "wr1", "qb1", "te1"],
                "player_name": ["Back One", "Back Two", "Wide One", "Passer", "End"],
                "position": ["RB", "RB", "WR", "QB", "TE"],
                "team": ["KC"] * 5,
                "projected_points": [250.0, 240.0, 230.0, 330.0, 180.0],
                "vbd": [100.0, 90.0, 95.0, 92.0, 70.0],
                "draft_now_score": [80.0, 78.0, 79.0, 77.0, 60.0],
                "tier": [1, 1, 1, 1, 2],
            }
        )

    def test_empty_lineup_is_worth_nothing(self, league: LeagueConfig):
        assert best_lineup_points(league, []) == 0.0

    def test_a_player_who_fills_an_empty_slot_adds_his_full_value(self, league: LeagueConfig):
        out = lineup_upgrades(league, [], self._board())
        by_key = dict(zip(out["player_key"], out["lineup_upgrade"], strict=True))
        assert by_key["qb1"] == pytest.approx(92.0)
        assert by_key["rb1"] == pytest.approx(100.0)

    def test_a_surplus_player_adds_little_once_his_slots_are_full(self, league: LeagueConfig):
        """Two RB slots plus a FLEX: the fourth back is a bench player."""
        roster = [("RB", 120.0), ("RB", 110.0), ("RB", 105.0)]
        out = lineup_upgrades(league, roster, self._board())
        by_key = dict(zip(out["player_key"], out["lineup_upgrade"], strict=True))
        assert by_key["rb2"] == pytest.approx(0.0)     # worse than all three held
        assert by_key["qb1"] > by_key["rb2"]           # the empty QB slot is worth more

    def test_a_better_player_upgrades_an_occupied_slot_by_the_difference(self, league):
        roster = [("RB", 60.0), ("RB", 50.0), ("RB", 40.0)]
        out = lineup_upgrades(league, roster, self._board())
        by_key = dict(zip(out["player_key"], out["lineup_upgrade"], strict=True))
        # rb1 (100) displaces the worst starting back (40) -> +60
        assert by_key["rb1"] == pytest.approx(60.0)

    def test_the_flex_absorbs_a_third_back(self, league: LeagueConfig):
        roster = [("RB", 100.0), ("RB", 90.0)]
        out = lineup_upgrades(league, roster, self._board())
        by_key = dict(zip(out["player_key"], out["lineup_upgrade"], strict=True))
        assert by_key["rb2"] == pytest.approx(90.0)    # straight into the empty FLEX

    def test_dedicated_slots_fill_before_flex(self, league: LeagueConfig):
        """A FLEX must never swallow the only tight end and leave TE empty."""
        total = best_lineup_points(league, [("TE", 50.0), ("RB", 90.0), ("RB", 80.0), ("WR", 70.0)])
        assert total == pytest.approx(50 + 90 + 80 + 70)

    def test_superflex_accepts_a_quarterback(self, superflex_league: LeagueConfig):
        one = best_lineup_points(superflex_league, [("QB", 100.0)])
        two = best_lineup_points(superflex_league, [("QB", 100.0), ("QB", 80.0)])
        assert two == pytest.approx(one + 80.0)

    def test_upgrade_is_never_negative(self, league: LeagueConfig):
        roster = [("RB", 200.0), ("RB", 190.0), ("RB", 180.0), ("WR", 170.0), ("WR", 160.0)]
        out = lineup_upgrades(league, roster, self._board())
        assert all(v >= 0 for v in out["lineup_upgrade"] if v is not None)

    def test_empty_board(self, league: LeagueConfig):
        empty = pl.DataFrame(
            schema={"player_key": pl.Utf8, "position": pl.Utf8, "vbd": pl.Float64}
        )
        assert "lineup_upgrade" in lineup_upgrades(league, [], empty).columns

    def test_needs_lists_unfilled_slots_first(self, league: LeagueConfig):
        board = lineup_upgrades(league, [], self._board())
        filled = {"QB": False, "RB": True, "WR": True, "TE": False, "FLEX": False,
                  "K": False, "DST": False}
        needs = positional_needs(league, [], filled, board)
        unfilled = [n.slot for n in needs if not n.filled]
        assert needs[0].slot in unfilled
        assert [n.filled for n in needs] == sorted(n.filled for n in needs)

    def test_needs_name_the_best_player_for_each_slot(self, league: LeagueConfig):
        board = lineup_upgrades(league, [], self._board())
        needs = {n.slot: n for n in positional_needs(league, [], {}, board)}
        assert needs["QB"].best_position == "QB"
        assert needs["TE"].best_position == "TE"
        assert needs["FLEX"].best_position in {"RB", "WR", "TE"}


class TestWhatINeedArea:
    def test_area_is_present_and_serializable(self, tmp_config, db):
        from fantasy_draft.draft.store import save_state

        save_state(db, build_fixture_draft(picks_made=41, slot=7))
        analysis = analyze_current_pick(tmp_config, db, refresh=False)
        payload = analysis.to_dict()
        assert "what_i_need" in payload
        json.dumps(payload)

    def test_unfilled_slots_are_reported(self, tmp_config, db):
        from fantasy_draft.draft.store import save_state

        save_state(db, build_fixture_draft(picks_made=41, slot=7))
        need = analyze_current_pick(tmp_config, db, refresh=False).what_i_need()
        assert isinstance(need["unfilled"], list)
        assert {s["slot"] for s in need["slots"]} >= {"QB", "RB", "WR", "TE"}
