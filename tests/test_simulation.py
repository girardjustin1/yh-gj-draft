"""Monte Carlo simulation: reproducibility, calibration, and two-pick expected value."""

from __future__ import annotations

import polars as pl
import pytest

from fantasy_draft.draft.availability import survival_probabilities
from fantasy_draft.draft.fixtures import build_fixture_draft
from fantasy_draft.draft.opponent_needs import opponent_needs
from fantasy_draft.draft.simulator import (
    blend_survival,
    simulate_to_next_pick,
    state_fingerprint,
)


@pytest.fixture
def sim_board() -> pl.DataFrame:
    positions = ["RB", "WR", "QB", "TE"] * 30
    return pl.DataFrame(
        {
            "player_key": [f"p{i}" for i in range(len(positions))],
            "player_name": [f"Player {i}" for i in range(len(positions))],
            "position": positions,
            "adp": [float(i + 40) for i in range(len(positions))],
            "adp_sd": [9.0] * len(positions),
            "player_score": [95.0 - 0.4 * i for i in range(len(positions))],
        }
    )


@pytest.fixture
def scenario(tmp_config, sim_board):
    state = build_fixture_draft(picks_made=41, slot=7)
    needs = opponent_needs(state, tmp_config.league, sim_board)
    return tmp_config, state, sim_board, needs


class TestReproducibility:
    def test_same_seed_gives_identical_results(self, scenario):
        cfg, state, board, needs = scenario
        a = simulate_to_next_pick(cfg, state, board, needs, iterations=400)
        b = simulate_to_next_pick(cfg, state, board, needs, iterations=400)
        assert a.survival == b.survival

    def test_a_different_seed_changes_results(self, scenario):
        cfg, state, board, needs = scenario
        a = simulate_to_next_pick(cfg, state, board, needs, iterations=400)
        cfg.weights.simulation.seed += 1
        b = simulate_to_next_pick(cfg, state, board, needs, iterations=400)
        assert a.survival != b.survival

    def test_fingerprint_tracks_the_draft_state(self):
        a = build_fixture_draft(picks_made=40, slot=7)
        b = build_fixture_draft(picks_made=41, slot=7)
        assert state_fingerprint(a) == state_fingerprint(build_fixture_draft(40, slot=7))
        assert state_fingerprint(a) != state_fingerprint(b)


class TestSimulationMechanics:
    def test_exactly_one_player_is_taken_per_pick(self, scenario):
        """Expected losses must total the number of picks simulated."""
        cfg, state, board, needs = scenario
        result = simulate_to_next_pick(cfg, state, board, needs, iterations=500)
        assert result.picks_simulated == len(needs)
        assert sum(result.expected_position_losses.values()) == pytest.approx(
            len(needs), abs=1e-6
        )

    def test_probabilities_are_valid(self, scenario):
        cfg, state, board, needs = scenario
        result = simulate_to_next_pick(cfg, state, board, needs, iterations=500)
        assert all(0.0 <= v <= 1.0 for v in result.survival.values())

    def test_better_players_survive_less(self, scenario):
        cfg, state, board, needs = scenario
        result = simulate_to_next_pick(cfg, state, board, needs, iterations=1000)
        assert result.survival["p0"] < result.survival["p60"]

    def test_no_needs_means_no_simulation(self, tmp_config, sim_board):
        state = build_fixture_draft(picks_made=41, slot=7)
        result = simulate_to_next_pick(tmp_config, state, sim_board, [])
        assert result.iterations == 0
        assert "no intervening picks" in result.approximation_note

    def test_empty_board(self, tmp_config, sim_board):
        state = build_fixture_draft(picks_made=41, slot=7)
        needs = opponent_needs(state, tmp_config.league, sim_board)
        empty = sim_board.head(0)
        assert simulate_to_next_pick(tmp_config, state, empty, needs).iterations == 0

    def test_more_iterations_converge(self, scenario):
        """The estimate must be stable, not drifting with sample size."""
        cfg, state, board, needs = scenario
        coarse = simulate_to_next_pick(cfg, state, board, needs, iterations=500)
        fine = simulate_to_next_pick(cfg, state, board, needs, iterations=6000)
        gaps = [abs(coarse.survival[k] - fine.survival[k]) for k in coarse.survival]
        assert max(gaps) < 0.12
        assert sum(gaps) / len(gaps) < 0.03

    def test_a_shorter_wait_means_more_survivors(self, tmp_config, sim_board):
        """The longer we wait, the fewer players are left.

        Which slot waits longest is not obvious — slot 12 takes its back-to-back at the
        turn and then waits 22 picks, the longest gap on the board — so the ordering is
        read off the states rather than assumed.
        """
        states = [build_fixture_draft(picks_made=12, slot=s) for s in (6, 12)]
        states.sort(key=lambda st: st.picks_until_next)
        short, long = states
        assert short.picks_until_next < long.picks_until_next

        results = [
            simulate_to_next_pick(
                tmp_config, st, sim_board,
                opponent_needs(st, tmp_config.league, sim_board), iterations=600,
            )
            for st in (short, long)
        ]
        assert results[0].picks_simulated == short.picks_until_next
        assert results[1].picks_simulated == long.picks_until_next
        assert (
            sum(results[0].survival.values()) / len(results[0].survival)
            > sum(results[1].survival.values()) / len(results[1].survival)
        )


class TestCalibrationAgainstTheAnalyticModel:
    def test_the_two_models_broadly_agree(self, scenario):
        """They make different approximations; sharp disagreement means a bug."""
        cfg, state, board, needs = scenario
        analytic = survival_probabilities(state, cfg.league, board, needs)
        simulated = simulate_to_next_pick(cfg, state, board, needs, iterations=4000)

        shared = [k for k in list(analytic.estimates)[:25] if k in simulated.survival]
        assert len(shared) >= 15
        gaps = [
            abs(analytic.estimates[k].probability_available - simulated.survival[k])
            for k in shared
        ]
        assert sum(gaps) / len(gaps) < 0.15

    def test_the_analytic_model_is_the_optimistic_one(self, scenario):
        """Treating picks as independent understates how fast the board drains."""
        cfg, state, board, needs = scenario
        analytic = survival_probabilities(state, cfg.league, board, needs)
        simulated = simulate_to_next_pick(cfg, state, board, needs, iterations=4000)
        shared = [k for k in list(analytic.estimates)[:25] if k in simulated.survival]
        mean_analytic = sum(
            analytic.estimates[k].probability_available for k in shared
        ) / len(shared)
        mean_simulated = sum(simulated.survival[k] for k in shared) / len(shared)
        assert mean_analytic >= mean_simulated - 0.02


class TestTwoPickValue:
    def test_combined_is_the_sum_of_its_parts(self, scenario):
        cfg, state, board, needs = scenario
        candidates = board["player_key"].to_list()[:5]
        result = simulate_to_next_pick(
            cfg, state, board, needs, candidates=candidates, iterations=800
        )
        for value in result.two_pick.values():
            assert value.combined == pytest.approx(
                value.value_now + value.expected_next_value
            )

    def test_every_requested_candidate_is_priced(self, scenario):
        cfg, state, board, needs = scenario
        candidates = board["player_key"].to_list()[:4]
        result = simulate_to_next_pick(
            cfg, state, board, needs, candidates=candidates, iterations=500
        )
        assert set(result.two_pick) == set(candidates)

    def test_the_candidate_is_excluded_from_his_own_next_pick(self, scenario):
        """We just took him — he cannot also be the best available next time."""
        cfg, state, board, needs = scenario
        best = board["player_key"][0]
        result = simulate_to_next_pick(
            cfg, state, board, needs, candidates=[best], iterations=800
        )
        value = result.two_pick[best]
        assert value.expected_next_value < value.value_now

    def test_the_candidate_limit_is_honoured(self, scenario):
        cfg, state, board, needs = scenario
        cfg.weights.simulation.two_pick_candidates = 3
        result = simulate_to_next_pick(
            cfg, state, board, needs,
            candidates=board["player_key"].to_list()[:10], iterations=400,
        )
        assert len(result.two_pick) == 3

    def test_the_confidence_interval_brackets_the_mean(self, scenario):
        cfg, state, board, needs = scenario
        result = simulate_to_next_pick(
            cfg, state, board, needs,
            candidates=board["player_key"].to_list()[:3], iterations=1500,
        )
        for value in result.two_pick.values():
            assert value.next_value_low <= value.expected_next_value <= value.next_value_high

    def test_taking_a_scarcer_player_leaves_a_weaker_next_pick(self, scenario):
        """The whole point of pricing both picks rather than one."""
        cfg, state, board, needs = scenario
        keys = board["player_key"].to_list()
        result = simulate_to_next_pick(
            cfg, state, board, needs, candidates=[keys[0], keys[50]], iterations=2000
        )
        top, deep = result.two_pick[keys[0]], result.two_pick[keys[50]]
        assert top.value_now > deep.value_now
        assert top.combined > deep.combined

    def test_position_mix_is_reported(self, scenario):
        cfg, state, board, needs = scenario
        result = simulate_to_next_pick(
            cfg, state, board, needs, candidates=[board["player_key"][0]], iterations=800
        )
        value = result.two_pick[board["player_key"][0]]
        assert value.likely_next_position in {"QB", "RB", "WR", "TE"}
        assert sum(value.position_mix.values()) == pytest.approx(1.0, abs=0.02)


class TestBlending:
    def test_averages_where_both_exist(self):
        out = blend_survival({"a": 0.8}, {"a": 0.6}, weight=0.5)
        assert out["a"] == pytest.approx(0.7)

    def test_uses_whichever_is_available(self):
        out = blend_survival({"a": 0.8}, {"b": 0.6})
        assert out == {"a": 0.8, "b": 0.6}

    def test_weight_shifts_toward_the_simulation(self):
        assert blend_survival({"a": 1.0}, {"a": 0.0}, weight=1.0)["a"] == 0.0
        assert blend_survival({"a": 1.0}, {"a": 0.0}, weight=0.0)["a"] == 1.0


class TestEndToEnd:
    def test_recommendation_returns_within_a_draft_clock(self, tmp_config, db):
        """Performance is a correctness requirement on a 30-90 second timer."""
        import time

        from fantasy_draft.recommendation.ranker import recommend

        state = build_fixture_draft(picks_made=41, slot=7)
        started = time.perf_counter()
        result = recommend(db, tmp_config, state, limit=6, iterations=2000)
        elapsed = time.perf_counter() - started
        assert elapsed < 25.0
        # With an empty database there is nothing to score; it must degrade, not crash.
        assert result.recommendation.pick_label == "4.06"
        assert result.recommendation.warnings
