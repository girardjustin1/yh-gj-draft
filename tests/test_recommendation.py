"""Opponent needs, survival, roster fit, strategy, and the Draft Now composition.

These are the pieces that turn a board into a decision, so the tests focus on the
*behaviours* that make the recommendation different from a ranking — especially the
ones that were wrong at first and are easy to regress.
"""

from __future__ import annotations

import polars as pl
import pytest

from fantasy_draft.analytics.draft_room import read_room
from fantasy_draft.analytics.market import market_signals
from fantasy_draft.analytics.roster_fit import roster_fit_scores
from fantasy_draft.config import LeagueConfig
from fantasy_draft.draft.availability import adp_survival, attach_survival, survival_probabilities
from fantasy_draft.draft.fixtures import build_fixture_draft
from fantasy_draft.draft.opponent_needs import (
    expected_position_demand,
    market_prior,
    opponent_needs,
    position_probabilities,
    unfilled_starters,
)
from fantasy_draft.draft.strategies import classify
from fantasy_draft.models import RosterSnapshot, Strategy
from fantasy_draft.scoring.draft_now import add_draft_now_score, add_tier_scarcity


def roster(**counts: int) -> RosterSnapshot:
    positions = [p for p, n in counts.items() for _ in range(n)]
    return RosterSnapshot(
        team_id="t", slot=1, player_keys=[f"k{i}" for i in range(len(positions))],
        positions=positions,
    )


def board_frame(n: int = 60) -> pl.DataFrame:
    positions = ["RB", "WR", "QB", "TE"] * (n // 4)
    return pl.DataFrame(
        {
            "player_key": [f"p{i}" for i in range(len(positions))],
            "player_name": [f"Player {i}" for i in range(len(positions))],
            "position": positions,
            "adp": [float(i + 1) for i in range(len(positions))],
            "adp_sd": [8.0] * len(positions),
            "player_score": [100.0 - i for i in range(len(positions))],
            "vbd": [200.0 - 2 * i for i in range(len(positions))],
        }
    )


class TestUnfilledStarters:
    def test_empty_roster_needs_every_slot(self, league: LeagueConfig):
        need = unfilled_starters(league, None)
        assert need["QB"] == 1.0
        assert need["RB"] == pytest.approx(2 + 1 / 3)
        assert need["TE"] == pytest.approx(1 + 1 / 3)

    def test_filled_dedicated_slots_drop_out(self, league: LeagueConfig):
        need = unfilled_starters(league, roster(QB=1, RB=2, WR=2, TE=1))
        assert need["QB"] == 0.0
        # The FLEX is still unfilled, shared across RB/WR/TE.
        assert need["RB"] == pytest.approx(1 / 3)

    def test_surplus_depth_fills_the_flex(self, league: LeagueConfig):
        need = unfilled_starters(league, roster(QB=1, RB=3, WR=2, TE=1))
        assert need["RB"] == pytest.approx(0.0, abs=1e-9)
        assert need["WR"] == pytest.approx(0.0, abs=1e-9)

    def test_need_never_goes_negative(self, league: LeagueConfig):
        need = unfilled_starters(league, roster(QB=3, RB=6, WR=6, TE=3))
        assert all(v >= 0 for v in need.values())


class TestMarketPrior:
    def test_reads_the_mix_off_the_board(self):
        frame = pl.DataFrame(
            {
                "position": ["RB"] * 9 + ["WR"] * 3,
                "adp": [float(i) for i in range(1, 13)],
            }
        )
        prior = market_prior(frame, 1, 12)
        assert prior.mix["RB"] == pytest.approx(0.75)
        assert prior.mix["WR"] == pytest.approx(0.25)

    def test_sparse_range_widens_rather_than_returning_nothing(self):
        frame = pl.DataFrame(
            {"position": ["RB", "WR"] * 20, "adp": [float(i) for i in range(1, 41)]}
        )
        prior = market_prior(frame, 200, 202)
        assert sum(prior.mix.values()) == pytest.approx(1.0)

    def test_empty_board_falls_back_to_flat(self):
        prior = market_prior(pl.DataFrame(), 1, 12)
        assert len(set(prior.mix.values())) == 1


class TestPositionProbabilities:
    def test_probabilities_sum_to_one(self, league: LeagueConfig):
        prior = market_prior(board_frame(), 40, 55)
        probabilities = position_probabilities(league, roster(RB=1), prior, 5)
        assert sum(probabilities.values()) == pytest.approx(1.0)

    def test_need_raises_the_position(self, league: LeagueConfig):
        prior = market_prior(board_frame(), 40, 55)
        no_qb = position_probabilities(league, roster(RB=3, WR=3), prior, 8)
        has_qb = position_probabilities(league, roster(RB=3, WR=3, QB=1), prior, 8)
        assert no_qb["QB"] > has_qb["QB"]

    def test_need_matters_more_later_in_the_draft(self, league: LeagueConfig):
        prior = market_prior(board_frame(), 40, 55)
        early = position_probabilities(league, roster(RB=2, WR=2), prior, 2)
        late = position_probabilities(league, roster(RB=2, WR=2), prior, 12)
        assert late["TE"] > early["TE"]

    def test_a_full_position_keeps_a_floor(self, league: LeagueConfig):
        prior = market_prior(board_frame(), 40, 55)
        probabilities = position_probabilities(league, roster(QB=2, RB=6, WR=6, TE=3), prior, 10)
        assert all(v > 0 for v in probabilities.values())


class TestOpponentNeeds:
    def test_one_entry_per_intervening_pick(self, league: LeagueConfig):
        state = build_fixture_draft(picks_made=41, slot=7)
        needs = opponent_needs(state, league, board_frame())
        assert len(needs) == state.picks_until_next == 12
        assert [n.pick_overall for n in needs] == list(range(43, 55))

    def test_no_needs_at_the_final_pick(self, league: LeagueConfig):
        state = build_fixture_draft(picks_made=41, slot=7, rounds=4)
        assert opponent_needs(state, league, board_frame()) == []

    def test_expected_demand_totals_the_pick_count(self, league: LeagueConfig):
        state = build_fixture_draft(picks_made=41, slot=7)
        needs = opponent_needs(state, league, board_frame())
        demand = expected_position_demand(needs)
        # Probabilities are rounded to 4dp for display, so allow for that accumulating.
        assert sum(demand.values()) == pytest.approx(len(needs), abs=0.01)

    def test_rationale_is_populated(self, league: LeagueConfig):
        state = build_fixture_draft(picks_made=41, slot=7)
        needs = opponent_needs(state, league, board_frame())
        assert all("round" in n.rationale for n in needs)


class TestAdpSurvival:
    def test_a_player_well_past_his_adp_almost_certainly_survives(self):
        assert adp_survival(adp=120, adp_sd=10, next_pick=55) > 0.99

    def test_a_player_well_before_his_adp_almost_certainly_does_not(self):
        assert adp_survival(adp=20, adp_sd=8, next_pick=55) < 0.01

    def test_at_his_adp_it_is_a_coin_flip(self):
        assert adp_survival(adp=55, adp_sd=10, next_pick=55) == pytest.approx(0.5)

    def test_uncertainty_widens_the_distribution(self):
        tight = adp_survival(adp=50, adp_sd=3, next_pick=55)
        loose = adp_survival(adp=50, adp_sd=25, next_pick=55)
        assert loose > tight

    def test_missing_adp_is_maximally_uncertain(self):
        assert adp_survival(None, None, 55) == 0.5


class TestSurvivalModel:
    def _setup(self, league: LeagueConfig):
        state = build_fixture_draft(picks_made=41, slot=7)
        frame = board_frame()
        needs = opponent_needs(state, league, frame)
        return state, frame, needs

    def test_probabilities_are_valid(self, league: LeagueConfig):
        state, frame, needs = self._setup(league)
        result = survival_probabilities(state, league, frame, needs)
        assert result.estimates
        for estimate in result.estimates.values():
            assert 0.0 <= estimate.probability_available <= 1.0
            assert estimate.probability_gone == pytest.approx(
                1 - estimate.probability_available
            )

    def test_earlier_adp_survives_less(self, league: LeagueConfig):
        state, frame, needs = self._setup(league)
        result = survival_probabilities(state, league, frame, needs)
        early = result.estimates["p0"].probability_available     # adp 1
        late = result.estimates["p40"].probability_available     # adp 41
        assert early < late

    def test_expected_losses_are_bounded_by_the_picks_modelled(self, league: LeagueConfig):
        state, frame, needs = self._setup(league)
        result = survival_probabilities(state, league, frame, needs)
        assert sum(result.expected_position_losses.values()) <= len(needs) + 1e-6

    def test_no_next_pick_means_nothing_to_survive(self, league: LeagueConfig):
        state = build_fixture_draft(picks_made=41, slot=7, rounds=4)
        result = survival_probabilities(state, league, board_frame(), [])
        assert result.method == "none"

    def test_every_player_gets_a_probability(self, league: LeagueConfig):
        """A deep player is 'certainly still there', never 'unknown'.

        Leaving him null lets the composition step redistribute his urgency weight,
        which rewards being unmodellable — the bug that put a player 90 picks past his
        ADP at the top of the board.
        """
        state, frame, needs = self._setup(league)
        result = survival_probabilities(state, league, frame, needs)
        attached = attach_survival(frame, result, state.my_next_pick)
        assert attached["probability_available"].null_count() == 0
        assert attached["next_pick_urgency_score"].null_count() == 0

    def test_deep_players_score_near_zero_urgency(self, league: LeagueConfig):
        state, frame, needs = self._setup(league)
        result = survival_probabilities(state, league, frame, needs)
        attached = attach_survival(frame, result, state.my_next_pick)
        deep = attached.filter(pl.col("adp") > 200)
        if deep.height:
            assert deep["next_pick_urgency_score"].max() < 15


class TestMarketValue:
    def _adp(self, values: list[float]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "player_key": [f"p{i}" for i in range(len(values))],
                "adp": values,
                "adp_sd": [10.0] * len(values),
                "adp_min": [None] * len(values),
                "adp_max": [None] * len(values),
                "adp_sources": [1] * len(values),
                "adp_updated_at": [None] * len(values),
                "adp_is_proxy": [True] * len(values),
            }
        )

    def _board(self, n: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "player_key": [f"p{i}" for i in range(n)],
                "position": ["RB"] * n,
                "vbd": [100.0 - i for i in range(n)],
            }
        )

    def test_a_faller_scores_above_neutral(self):
        out = market_signals(
            self._board(2), self._adp([60.0, 42.0]), current_pick=42,
            picks_until_next=12, next_pick=55,
        )
        assert out.filter(pl.col("player_key") == "p0")["market_value_score"][0] > 55

    def test_a_reach_scores_below_neutral(self):
        out = market_signals(
            self._board(1), self._adp([80.0]), current_pick=100,
            picks_until_next=12, next_pick=113,
        )
        assert out["market_value_score"][0] < 50

    def test_an_unreachable_discount_is_not_treated_as_value(self):
        """A player 90 picks past his ADP is not a bargain — we could just wait.

        Without the cap and the survival weighting, every deep player scored a perfect
        100 here and floated to the top of the board.
        """
        out = market_signals(
            self._board(2), self._adp([132.0, 48.0]), current_pick=42,
            picks_until_next=12, next_pick=55,
        )
        deep = out.filter(pl.col("player_key") == "p0")["market_value_score"][0]
        near = out.filter(pl.col("player_key") == "p1")["market_value_score"][0]
        assert deep < 100
        assert deep < near

    def test_discount_is_weighted_by_the_chance_of_losing_him(self):
        """The same nominal discount is worth less if he will still be there."""
        with_survival = market_signals(
            self._board(1), self._adp([54.0]), current_pick=42,
            picks_until_next=12, next_pick=55,
        )["market_value_score"][0]
        without = market_signals(
            self._board(1), self._adp([54.0]), current_pick=42, picks_until_next=12,
        )["market_value_score"][0]
        assert with_survival < without

    def test_works_without_draft_context(self):
        out = market_signals(self._board(3), self._adp([10.0, 20.0, 30.0]))
        assert "market_value_score" in out.columns
        assert out["market_value_score"].null_count() == 0


class TestRosterFit:
    def test_an_empty_starting_slot_scores_above_neutral(self, league: LeagueConfig):
        scores = roster_fit_scores(league, roster(RB=2, WR=2), rounds_remaining=10)
        assert scores["QB"] > 50
        assert scores["TE"] > 50

    def test_fit_is_a_modifier_not_a_veto(self, league: LeagueConfig):
        """With the lineup full, a fourth RB is depth — rated below a real need, never vetoed."""
        scores = roster_fit_scores(league, roster(RB=3, WR=2, QB=1, TE=1), rounds_remaining=8)
        deep = roster_fit_scores(league, roster(RB=6, WR=2, QB=1, TE=1), rounds_remaining=8)
        assert 35 < scores["RB"] < 65          # near neutral, not excluded
        assert deep["RB"] < scores["RB"]       # stacking further is progressively worse
        assert deep["RB"] > 0                  # but still never a hard veto

    def test_irrational_builds_are_hit_hard(self, league: LeagueConfig):
        scores = roster_fit_scores(league, roster(QB=3), rounds_remaining=8)
        assert scores["QB"] < 30

    def test_unfillable_starting_slot_late_is_urgent(self, league: LeagueConfig):
        late = roster_fit_scores(league, roster(RB=4, WR=4), rounds_remaining=1)
        early = roster_fit_scores(league, roster(RB=4, WR=4), rounds_remaining=10)
        assert late["QB"] > early["QB"]

    def test_scores_stay_in_range(self, league: LeagueConfig):
        for r in (None, roster(RB=9), roster(QB=1, RB=2, WR=2, TE=1)):
            scores = roster_fit_scores(league, r, rounds_remaining=5)
            assert all(0 <= v <= 100 for v in scores.values())


class TestStrategy:
    def test_no_rb_and_two_wr_reads_as_zero_rb(self, league: LeagueConfig, weights):
        state = classify(league, roster(WR=3), weights.strategy_priors, current_round=4)
        assert state.primary is Strategy.ZERO_RB
        assert "no RB" in state.reason

    def test_one_rb_plus_receivers_reads_as_hero_rb(self, league: LeagueConfig, weights):
        state = classify(league, roster(RB=1, WR=2), weights.strategy_priors, current_round=4)
        assert state.primary is Strategy.HERO_RB

    def test_three_rbs_reads_as_robust_rb(self, league: LeagueConfig, weights):
        state = classify(league, roster(RB=3, WR=1), weights.strategy_priors, current_round=5)
        assert state.primary is Strategy.ROBUST_RB

    def test_priors_apply_before_any_picks(self, league: LeagueConfig, weights):
        state = classify(league, None, weights.strategy_priors, current_round=1)
        assert state.confidence < 0.7
        assert "priors" in state.reason

    def test_probabilities_sum_to_one(self, league: LeagueConfig, weights):
        state = classify(league, roster(RB=2, WR=2), weights.strategy_priors, current_round=5)
        assert sum(state.probabilities.values()) == pytest.approx(1.0)

    def test_falling_rb_value_supports_robust_rb(self, league: LeagueConfig, weights):
        calm = classify(league, roster(RB=1, WR=1), weights.strategy_priors, 4,
                        rb_value_available=50)
        falling = classify(league, roster(RB=1, WR=1), weights.strategy_priors, 4,
                           rb_value_available=90)
        assert (falling.probabilities[Strategy.ROBUST_RB]
                > calm.probabilities[Strategy.ROBUST_RB])

    def test_an_rb_run_supports_zero_rb(self, league: LeagueConfig, weights):
        calm = classify(league, roster(WR=2), weights.strategy_priors, 4, rb_run_intensity=0)
        run = classify(league, roster(WR=2), weights.strategy_priors, 4, rb_run_intensity=80)
        assert run.probabilities[Strategy.ZERO_RB] > calm.probabilities[Strategy.ZERO_RB]

    def test_full_ppr_supports_zero_rb(self, weights):
        ppr = LeagueConfig.model_validate({"teams": 12, "scoring": {"reception": 1.0}})
        standard = LeagueConfig.model_validate({"teams": 12, "scoring": {"reception": 0.0}})
        a = classify(ppr, roster(WR=2), weights.strategy_priors, 3)
        b = classify(standard, roster(WR=2), weights.strategy_priors, 3)
        assert a.probabilities[Strategy.ZERO_RB] > b.probabilities[Strategy.ZERO_RB]

    def test_fit_is_never_a_hard_instruction(self, league: LeagueConfig, weights):
        """Strategy tilts a decision; it never dictates one.

        A Zero-RB-leaning roster still rates running backs in the middle of the range,
        because Zero RB is only ever a partial belief — the other strategies in the mix
        still want a back, and a team with four receivers and no RB genuinely needs one.
        """
        zero = classify(league, roster(WR=4), weights.strategy_priors, current_round=5)
        robust = classify(league, roster(RB=3, WR=1), weights.strategy_priors, current_round=5)
        zero_fit = zero.fit("RB", roster(WR=4), league)
        assert 25 < zero_fit < 75
        assert zero_fit < robust.fit("RB", roster(RB=3, WR=1), league)

    def test_label_shows_a_genuine_split(self, league: LeagueConfig, weights):
        state = classify(league, None, weights.strategy_priors, current_round=1)
        assert isinstance(state.label, str) and state.label


class TestDraftRoom:
    def test_a_run_is_measured_against_the_expected_mix(self, league: LeagueConfig):
        state = build_fixture_draft(picks_made=41, slot=7)
        frame = board_frame()
        needs = opponent_needs(state, league, frame)
        read = read_room(state, frame, needs)
        assert set(read.demand) >= {"QB", "RB", "WR", "TE"}
        assert all(0 <= v <= 100 for v in read.demand.values())
        assert all(0 <= v <= 100 for v in read.value_created.values())

    def test_value_created_rises_for_a_position_the_room_skipped(self, league: LeagueConfig):
        state = build_fixture_draft(picks_made=41, slot=7)
        frame = pl.DataFrame(
            {
                "player_key": ["a", "b"],
                "position": ["RB", "WR"],
                "adp": [20.0, 90.0],   # the WR has fallen a long way past this pick
                "player_score": [90.0, 88.0],
            }
        )
        read = read_room(state, frame, opponent_needs(state, league, frame))
        assert read.value_created["WR"] > read.value_created["RB"]


class TestDraftNowComposition:
    def _scored(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "player_key": ["urgent", "safe"],
                "position": ["RB", "RB"],
                "player_score": [80.0, 82.0],
                "player_score_confidence": [1.0, 1.0],
                "value_score": [70.0, 70.0],
                "value_score_confidence": [1.0, 1.0],
                "next_pick_urgency_score": [95.0, 5.0],
                "survival_confidence": [0.8, 0.8],
                "tier_cliff_score": [50.0, 50.0],
                "scarcity_score": [60.0, 60.0],
                "scarcity_confidence": [0.8, 0.8],
                "roster_fit_score": [60.0, 60.0],
                "roster_fit_confidence": [1.0, 1.0],
                "draft_room_score": [55.0, 55.0],
                "draft_room_confidence": [0.7, 0.7],
                "strategy_fit_score": [55.0, 55.0],
                "strategy_confidence": [0.6, 0.6],
            }
        )

    def test_urgency_can_outrank_a_better_player(self, tmp_config):
        """The whole point: a slightly worse player who will be gone comes first."""
        frame = add_draft_now_score(tmp_config, add_tier_scarcity(self._scored()))
        assert frame["player_key"][0] == "urgent"

    def test_ranks_are_assigned(self, tmp_config):
        frame = add_draft_now_score(tmp_config, add_tier_scarcity(self._scored()))
        assert frame["draft_now_rank"].to_list() == [1, 2]

    def test_scores_stay_in_range(self, tmp_config):
        frame = add_draft_now_score(tmp_config, add_tier_scarcity(self._scored()))
        assert frame["draft_now_score"].min() >= 0
        assert frame["draft_now_score"].max() <= 100

    def test_empty_board(self, tmp_config):
        empty = pl.DataFrame(schema={"player_key": pl.Utf8, "tier_cliff_score": pl.Float64,
                                     "scarcity_score": pl.Float64})
        assert add_draft_now_score(tmp_config, add_tier_scarcity(empty)).is_empty()
