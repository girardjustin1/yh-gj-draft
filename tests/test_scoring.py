"""Fantasy scoring, replacement level, VBD, tiers, scarcity, and score composition."""

from __future__ import annotations

import polars as pl
import pytest

from fantasy_draft.analytics.fantasy_points import fantasy_points_expr, score_weekly_stats
from fantasy_draft.analytics.replacement import replacement_levels, replacement_rank
from fantasy_draft.analytics.scarcity import position_scarcity
from fantasy_draft.analytics.tiers import _local_gap_scores, _tier_one_position, assign_tiers
from fantasy_draft.analytics.vbd import add_vbd, compute_vbd
from fantasy_draft.config import AppConfig, LeagueConfig, ScoringRules
from fantasy_draft.models import ComponentScore
from fantasy_draft.scoring.compose import compose, component


def _stat_row(**kwargs) -> pl.DataFrame:
    base = {
        "passing_yards": 0.0, "passing_tds": 0.0, "interceptions": 0.0,
        "rushing_yards": 0.0, "rushing_tds": 0.0, "receiving_yards": 0.0,
        "receiving_tds": 0.0, "receptions": 0.0, "fumbles_lost": 0.0,
        "two_point_conv": 0.0,
    }
    base.update(kwargs)
    return pl.DataFrame({k: [v] for k, v in base.items()})


class TestFantasyPoints:
    def test_half_ppr_receiving_line(self):
        """8 catches, 100 yards, 1 TD in half-PPR = 4 + 10 + 6 = 20."""
        frame = _stat_row(receptions=8, receiving_yards=100, receiving_tds=1)
        scored = score_weekly_stats(frame, ScoringRules(reception=0.5))
        assert scored["fantasy_points_league"][0] == pytest.approx(20.0)

    def test_reception_setting_changes_the_answer(self):
        frame = _stat_row(receptions=8, receiving_yards=100, receiving_tds=1)
        standard = score_weekly_stats(frame, ScoringRules(reception=0.0))
        full = score_weekly_stats(frame, ScoringRules(reception=1.0))
        assert standard["fantasy_points_league"][0] == pytest.approx(16.0)
        assert full["fantasy_points_league"][0] == pytest.approx(24.0)

    def test_passing_line(self):
        """300 yards, 2 TD, 1 INT at 25 yds/pt and 4 pt TDs = 12 + 8 - 2 = 18."""
        frame = _stat_row(passing_yards=300, passing_tds=2, interceptions=1)
        scored = score_weekly_stats(frame, ScoringRules())
        assert scored["fantasy_points_league"][0] == pytest.approx(18.0)

    def test_six_point_passing_touchdowns(self):
        frame = _stat_row(passing_yards=300, passing_tds=2)
        scored = score_weekly_stats(frame, ScoringRules(passing_td=6.0))
        assert scored["fantasy_points_league"][0] == pytest.approx(24.0)

    def test_bonuses_apply_only_above_the_threshold(self):
        rules = ScoringRules(bonus_rec_100_yards=3.0)
        under = score_weekly_stats(_stat_row(receiving_yards=99), rules)
        over = score_weekly_stats(_stat_row(receiving_yards=100), rules)
        assert under["fantasy_points_league"][0] == pytest.approx(9.9)
        assert over["fantasy_points_league"][0] == pytest.approx(13.0)

    def test_zero_bonus_is_a_no_op(self):
        frame = _stat_row(receiving_yards=150)
        assert score_weekly_stats(frame, ScoringRules())["fantasy_points_league"][0] == 15.0

    def test_fumbles_subtract(self):
        frame = _stat_row(rushing_yards=100, fumbles_lost=1)
        assert score_weekly_stats(frame, ScoringRules())["fantasy_points_league"][0] == 8.0

    def test_missing_columns_degrade_to_zero(self):
        """Historical tables vary by season; a missing column must not raise."""
        frame = pl.DataFrame({"receiving_yards": [100.0]})
        scored = score_weekly_stats(frame, ScoringRules())
        assert scored["fantasy_points_league"][0] == pytest.approx(10.0)

    def test_expression_is_reusable(self):
        expr = fantasy_points_expr(ScoringRules(), {"rushing_yards"})
        frame = pl.DataFrame({"rushing_yards": [50.0, 100.0]}).with_columns(expr)
        assert frame["fantasy_points_league"].to_list() == [5.0, 10.0]

    def test_empty_frame(self):
        empty = pl.DataFrame(schema={"receiving_yards": pl.Float64})
        assert score_weekly_stats(empty, ScoringRules()).height == 0


class TestReplacement:
    def test_rb_replacement_is_well_below_rb24(self, tmp_config: AppConfig):
        """The spec's core point: flex and bench hoarding push RB replacement past RB24."""
        rank, method, demand, multiplier = replacement_rank(tmp_config, "RB")
        assert demand == pytest.approx(28.0)  # 12 teams x (2 RB + 1/3 FLEX)
        assert multiplier > 1.0
        assert rank > 24
        assert method == "blended"

    def test_superflex_raises_qb_replacement(self, tmp_path, superflex_league):
        from fantasy_draft.config import DataSourcesConfig, Paths, WeightsConfig

        superflex = AppConfig(
            paths=Paths.resolve(tmp_path), league=superflex_league,
            weights=WeightsConfig(), data_sources=DataSourcesConfig(),
            league_file_exists=True,
        )
        assert replacement_rank(superflex, "QB")[0] > replacement_rank_for_1qb(tmp_path)

    def test_fixed_rank_method_ignores_league_shape(self, tmp_config: AppConfig):
        tmp_config.weights.replacement.method = "fixed_rank"
        rank, method, _, _ = replacement_rank(tmp_config, "RB")
        assert method == "fixed_rank"
        assert rank == tmp_config.weights.replacement.fixed_rank["RB"]

    def test_starter_demand_method(self, tmp_config: AppConfig):
        tmp_config.weights.replacement.method = "starter_demand"
        rank, method, demand, multiplier = replacement_rank(tmp_config, "RB")
        assert method == "starter_demand"
        assert rank == round(demand * multiplier)

    def test_levels_smooth_over_a_window(self, tmp_config: AppConfig):
        """One spiky projection at the replacement rank must not move the baseline much."""
        points = [300.0 - 5 * i for i in range(60)]
        clean = pl.DataFrame({"position": ["RB"] * 60, "projected_points": points})
        spiked = points.copy()
        rank = replacement_rank(tmp_config, "RB")[0]
        spiked[rank - 1] = 0.0
        noisy = pl.DataFrame({"position": ["RB"] * 60, "projected_points": spiked})

        a = replacement_levels(tmp_config, clean, ("RB",))["RB"].points
        b = replacement_levels(tmp_config, noisy, ("RB",))["RB"].points
        assert abs(a - b) < 0.25 * a

    def test_short_pool_clamps_to_what_exists(self, tmp_config: AppConfig):
        pool = pl.DataFrame({"position": ["TE"] * 3, "projected_points": [90.0, 80.0, 70.0]})
        level = replacement_levels(tmp_config, pool, ("TE",))["TE"]
        assert level.rank == 3
        assert level.players_available == 3

    def test_empty_pool_is_reported_not_crashed(self, tmp_config: AppConfig):
        empty = pl.DataFrame(schema={"position": pl.Utf8, "projected_points": pl.Float64})
        level = replacement_levels(tmp_config, empty, ("K",))["K"]
        assert level.players_available == 0
        assert "no players" in level.method

    def test_explanation_is_human_readable(self, tmp_config: AppConfig):
        pool = pl.DataFrame(
            {"position": ["RB"] * 50, "projected_points": [300.0 - 4 * i for i in range(50)]}
        )
        text = replacement_levels(tmp_config, pool, ("RB",))["RB"].explanation
        assert "starting slots" in text and "replacement at RB" in text


def replacement_rank_for_1qb(tmp_path) -> int:
    from fantasy_draft.config import DataSourcesConfig, Paths, WeightsConfig

    cfg = AppConfig(
        paths=Paths.resolve(tmp_path),
        league=LeagueConfig.model_validate({"teams": 12}),
        weights=WeightsConfig(), data_sources=DataSourcesConfig(), league_file_exists=True,
    )
    return replacement_rank(cfg, "QB")[0]


class TestVBD:
    def _pool(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "player_key": [f"p{i}" for i in range(8)],
                "position": ["RB"] * 4 + ["QB"] * 4,
                "projected_points": [300.0, 250.0, 200.0, 150.0, 400.0, 380.0, 360.0, 340.0],
            }
        )

    def test_vbd_is_points_above_replacement(self):
        levels = replacement_levels_stub({"RB": 150.0, "QB": 340.0})
        out = add_vbd(self._pool(), levels)
        rb1 = out.filter(pl.col("player_key") == "p0")
        assert rb1["vbd"][0] == pytest.approx(150.0)

    def test_vbd_corrects_the_quarterback_illusion(self):
        """Raw points put QBs on top; VBD must not."""
        levels = replacement_levels_stub({"RB": 150.0, "QB": 340.0})
        out = add_vbd(self._pool(), levels)
        assert out["position"][0] == "RB"
        top_qb = out.filter(pl.col("position") == "QB").sort("vbd", descending=True)
        assert top_qb["vbd"][0] == pytest.approx(60.0)

    def test_positional_and_overall_ranks(self):
        levels = replacement_levels_stub({"RB": 150.0, "QB": 340.0})
        out = add_vbd(self._pool(), levels)
        assert set(out.filter(pl.col("position") == "RB")["positional_rank"]) == {1, 2, 3, 4}
        assert out["overall_rank"].min() == 1

    def test_empty_input(self):
        empty = pl.DataFrame(schema={"position": pl.Utf8, "projected_points": pl.Float64})
        assert add_vbd(empty, {}).is_empty()

    def test_compute_vbd_end_to_end(self, tmp_config: AppConfig):
        pool = pl.DataFrame(
            {
                "player_key": [f"p{i}" for i in range(60)],
                "position": ["RB"] * 60,
                "projected_points": [300.0 - 3.5 * i for i in range(60)],
            }
        )
        out, levels = compute_vbd(tmp_config, pool, positions=("RB",))
        assert "vbd" in out.columns
        assert levels["RB"].points > 0
        assert out["vbd"].max() == pytest.approx(300.0 - levels["RB"].points)


def replacement_levels_stub(points: dict[str, float]):
    from fantasy_draft.analytics.replacement import ReplacementLevel

    return {
        position: ReplacementLevel(position, 1, value, "test", 1.0, 1.0, 1)
        for position, value in points.items()
    }


class TestTiers:
    def test_local_gaps_find_a_cliff_a_global_threshold_misses(self):
        """Gaps shrink down a position; the cliff at rank 20 must still register."""
        gaps = [18.0, 15.0, 13.0, 11.0] + [4.0] * 14 + [22.0] + [4.0] * 10
        scores = _local_gap_scores(__import__("numpy").array(gaps))
        assert scores[18] > 2.0  # the planted 22-point cliff
        assert max(scores[:4]) < scores[18]

    def test_tiers_are_monotonic_and_start_at_one(self):
        points = [300.0 - i * 4 for i in range(40)]
        tiers = _tier_one_position(points, gap_sigma=1.0, max_tiers=10, min_tier_size=1)
        assert tiers[0] == 1
        assert all(b >= a for a, b in zip(tiers, tiers[1:], strict=False))
        assert max(tiers) <= 10

    def test_max_tiers_is_respected(self):
        points = [300.0 - (i * i) for i in range(30)]
        tiers = _tier_one_position(points, gap_sigma=0.1, max_tiers=3, min_tier_size=1)
        assert max(tiers) == 3

    def test_degenerate_inputs(self):
        assert _tier_one_position([], 1.0, 5, 1) == []
        assert _tier_one_position([100.0], 1.0, 5, 1) == [1]
        assert _tier_one_position([100.0] * 10, 1.0, 5, 1) == [1] * 10

    def test_assign_tiers_adds_every_column(self, tmp_config: AppConfig):
        frame = pl.DataFrame(
            {
                "player_key": [f"p{i}" for i in range(30)],
                "position": ["WR"] * 30,
                "projected_points": [280.0 - 6 * i for i in range(30)],
            }
        )
        out = assign_tiers(tmp_config, frame)
        for column in ("tier", "tier_rank", "tier_size", "points_to_next_player",
                       "points_to_next_tier", "tier_cliff_score"):
            assert column in out.columns
        assert out["tier_rank"].min() == 1
        assert out["tier_cliff_score"].max() <= 100

    def test_tier_rank_restarts_each_tier(self, tmp_config: AppConfig):
        frame = pl.DataFrame(
            {
                "player_key": [f"p{i}" for i in range(20)],
                "position": ["RB"] * 20,
                "projected_points": [300.0, 298.0, 296.0, 250.0, 248.0, 246.0]
                + [200.0 - i for i in range(14)],
            }
        )
        out = assign_tiers(tmp_config, frame).sort("projected_points", descending=True)
        first_of_each = out.group_by("tier").agg(pl.col("tier_rank").min())
        assert set(first_of_each["tier_rank"]) == {1}

    def test_two_positions_are_tiered_independently(self, tmp_config: AppConfig):
        frame = pl.DataFrame(
            {
                "player_key": [f"p{i}" for i in range(20)],
                "position": ["RB"] * 10 + ["TE"] * 10,
                "projected_points": [300.0 - 5 * i for i in range(10)]
                + [150.0 - 3 * i for i in range(10)],
            }
        )
        out = assign_tiers(tmp_config, frame)
        assert out.filter(pl.col("position") == "RB")["tier"].min() == 1
        assert out.filter(pl.col("position") == "TE")["tier"].min() == 1


class TestScarcity:
    def _board(self, rb: int, wr: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "player_key": [f"r{i}" for i in range(rb)] + [f"w{i}" for i in range(wr)],
                "position": ["RB"] * rb + ["WR"] * wr,
                "projected_points": [300.0 - 4 * i for i in range(rb)]
                + [280.0 - 3 * i for i in range(wr)],
            }
        )

    def test_thin_position_scores_higher(self, tmp_config: AppConfig):
        levels = replacement_levels_stub({"RB": 150.0, "WR": 150.0})
        scarcity = position_scarcity(
            tmp_config, self._board(rb=12, wr=60), levels, positions=("RB", "WR")
        )
        assert scarcity["RB"].score > scarcity["WR"].score

    def test_supply_ratio_below_one_is_flagged(self, tmp_config: AppConfig):
        levels = replacement_levels_stub({"RB": 150.0})
        scarcity = position_scarcity(tmp_config, self._board(rb=8, wr=0), levels,
                                     positions=("RB",))
        assert scarcity["RB"].supply_ratio < 1.0
        assert any("fewer startable" in note for note in scarcity["RB"].notes)

    def test_drafted_players_reduce_remaining_demand(self, tmp_config: AppConfig):
        levels = replacement_levels_stub({"RB": 150.0})
        board = self._board(rb=30, wr=0)
        before = position_scarcity(tmp_config, board, levels, positions=("RB",))["RB"]
        after = position_scarcity(
            tmp_config, board, levels, drafted_by_position={"RB": 20}, positions=("RB",)
        )["RB"]
        assert after.remaining_demand < before.remaining_demand

    def test_expected_gone_raises_urgency(self, tmp_config: AppConfig):
        levels = replacement_levels_stub({"RB": 150.0})
        board = self._board(rb=20, wr=0)
        calm = position_scarcity(
            tmp_config, board, levels, expected_gone={"RB": 0.0}, positions=("RB",)
        )["RB"]
        panic = position_scarcity(
            tmp_config, board, levels, expected_gone={"RB": 15.0}, positions=("RB",)
        )["RB"]
        assert panic.score > calm.score

    def test_explanation_mentions_supply_and_demand(self, tmp_config: AppConfig):
        levels = replacement_levels_stub({"RB": 150.0})
        text = position_scarcity(
            tmp_config, self._board(rb=20, wr=0), levels, positions=("RB",)
        )["RB"].explanation
        assert "startable left against" in text


class TestCompose:
    def _components(self, **overrides) -> dict[str, ComponentScore]:
        base = {
            "projection": component("projection", 300.0, 90.0, 1.0),
            "vbd": component("vbd", 150.0, 80.0, 1.0),
            "opportunity": component("opportunity", 20.0, 70.0, 1.0),
        }
        base.update(overrides)
        return base

    def test_weighted_average(self):
        weights = {"projection": 0.5, "vbd": 0.3, "opportunity": 0.2}
        bundle = compose("test", self._components(), weights)
        assert bundle.value == pytest.approx(90 * 0.5 + 80 * 0.3 + 70 * 0.2)
        assert bundle.confidence == pytest.approx(1.0)

    def test_unknown_component_weight_is_redistributed_not_scored_as_average(self):
        """The rule that makes confidence mean something."""
        weights = {"projection": 0.5, "vbd": 0.3, "opportunity": 0.2}
        components = self._components(
            opportunity=component("opportunity", None, None, 0.0)
        )
        bundle = compose("test", components, weights)
        # Renormalized over the two known components: 90*(0.5/0.8) + 80*(0.3/0.8)
        assert bundle.value == pytest.approx(90 * 0.625 + 80 * 0.375)
        # A neutral-50 treatment would have produced something noticeably lower.
        assert bundle.value > 90 * 0.5 + 80 * 0.3 + 50 * 0.2
        assert bundle.confidence == pytest.approx(0.8)

    def test_partial_confidence_scales_influence(self):
        weights = {"projection": 0.5, "vbd": 0.5}
        full = compose("t", {"projection": component("projection", 1, 100.0, 1.0),
                             "vbd": component("vbd", 1, 0.0, 1.0)}, weights)
        half = compose("t", {"projection": component("projection", 1, 100.0, 1.0),
                             "vbd": component("vbd", 1, 0.0, 0.5)}, weights)
        assert full.value == pytest.approx(50.0)
        assert half.value > full.value  # the confident high score carries more weight
        assert half.confidence < full.confidence

    def test_everything_unknown_reports_zero_confidence(self):
        components = {"projection": component("projection", None, None, 0.0)}
        bundle = compose("test", components, {"projection": 1.0})
        assert bundle.confidence == 0.0
        assert bundle.value == 50.0
        assert bundle.weights == {}

    def test_dropping_a_small_weight_costs_little_confidence(self):
        weights = {"projection": 0.925, "schedule": 0.075}
        components = {
            "projection": component("projection", 1, 90.0, 1.0),
            "schedule": component("schedule", None, None, 0.0),
        }
        bundle = compose("test", components, weights)
        assert bundle.confidence == pytest.approx(0.925)
        assert bundle.value == pytest.approx(90.0)

    def test_component_helper_marks_missing_as_unknown(self):
        comp = component("x", None, None)
        assert comp.confidence == 0.0
        assert comp.notes == "no data"

    def test_component_clamps_to_range(self):
        assert component("x", 1.0, 250.0).normalized == 100.0
        assert component("x", 1.0, -50.0).normalized == 0.0

    def test_bundle_value_is_bounded(self):
        bundle = compose("t", {"a": component("a", 1, 100.0, 1.0)}, {"a": 1.0})
        assert 0 <= bundle.value <= 100
