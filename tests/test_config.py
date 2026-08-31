"""League configuration, scoring rules, and weight validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fantasy_draft.config import (
    AppConfig,
    ConfigError,
    DraftNowWeights,
    LeagueConfig,
    Paths,
    PlayerScoreWeights,
    RosterSlots,
    ScoringRules,
    StrategyPriors,
    WeightsConfig,
    load_config,
)


class TestScoringRules:
    def test_ppr_labels(self):
        assert ScoringRules(reception=1.0).ppr_label == "PPR"
        assert ScoringRules(reception=0.5).ppr_label == "Half-PPR"
        assert ScoringRules(reception=0.0).ppr_label == "Standard"
        assert ScoringRules(reception=0.25).ppr_label == "0.25-PPR"

    def test_yards_per_point_must_be_positive(self):
        with pytest.raises(ValidationError):
            ScoringRules(rushing_yards_per_point=0)

    def test_unknown_keys_are_rejected(self):
        """A typo in league.yaml must fail loudly, not be silently ignored."""
        with pytest.raises(ValidationError):
            ScoringRules(recieption=1.0)


class TestRosterSlots:
    def test_starters_and_total(self, league: LeagueConfig):
        roster = league.roster
        assert roster.starters == 9  # 1QB 2RB 2WR 1TE 1FLEX 1K 1DST
        assert roster.total == 15

    def test_flex_counts_omit_zeros(self, league: LeagueConfig):
        assert league.roster.flex_counts == {"FLEX": 1}

    def test_flex_slots_accepting(self, league: LeagueConfig):
        assert league.roster.flex_slots_accepting("RB") == 1
        assert league.roster.flex_slots_accepting("WR") == 1
        assert league.roster.flex_slots_accepting("TE") == 1
        assert league.roster.flex_slots_accepting("QB") == 0

    def test_superflex_accepts_quarterbacks(self, superflex_league: LeagueConfig):
        roster = superflex_league.roster
        assert roster.is_superflex
        assert roster.flex_slots_accepting("QB") == 1
        assert roster.flex_slots_accepting("RB") == 2  # FLEX + SUPERFLEX

    def test_negative_slots_rejected(self):
        with pytest.raises(ValidationError):
            RosterSlots(rb=-1)


class TestStarterDemand:
    def test_dedicated_slots(self, league: LeagueConfig):
        assert league.starter_demand("QB") == 12.0  # 1 per team
        assert league.starter_demand("TE") == 12.0 + 12 * (1 / 3)

    def test_flex_is_split_across_eligible_positions(self, league: LeagueConfig):
        # 2 dedicated RB + 1/3 of a FLEX, times 12 teams.
        assert league.starter_demand("RB") == pytest.approx(28.0)
        assert league.starter_demand("WR") == pytest.approx(28.0)

    def test_superflex_raises_quarterback_demand(self, superflex_league: LeagueConfig):
        assert superflex_league.starter_demand("QB") > superflex_league.teams

    def test_unrostered_position_has_no_demand(self):
        league = LeagueConfig.model_validate(
            {"teams": 10, "roster": {"k": 0, "dst": 0, "flex": 0}}
        )
        assert league.starter_demand("K") == 0.0


class TestLeagueConfig:
    def test_label(self, league: LeagueConfig):
        assert league.label == "12-team Half-PPR"

    def test_superflex_in_label(self, superflex_league: LeagueConfig):
        assert "Superflex" in superflex_league.label

    def test_slot_cannot_exceed_team_count(self):
        with pytest.raises(ValidationError, match="exceeds teams"):
            LeagueConfig.model_validate({"teams": 10, "draft": {"slot": 11}})

    def test_slot_may_be_unset(self):
        assert LeagueConfig.model_validate({"teams": 12}).draft.slot is None

    def test_weeks_are_validated_and_sorted(self):
        cfg = LeagueConfig.model_validate({"playoff_weeks": [17, 15, 16, 15]})
        assert cfg.playoff_weeks == [15, 16, 17]
        with pytest.raises(ValidationError, match="between 1 and 22"):
            LeagueConfig.model_validate({"playoff_weeks": [99]})

    def test_total_drafted_players(self, league: LeagueConfig):
        assert league.total_drafted_players == 180


class TestWeights:
    def test_defaults_sum_to_one(self):
        for block in (WeightsConfig().player_score, WeightsConfig().value_score,
                      WeightsConfig().draft_now):
            assert sum(block.as_dict().values()) == pytest.approx(1.0)

    def test_weights_that_do_not_sum_to_one_are_rejected(self):
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            PlayerScoreWeights(projection=0.9, vbd=0.9)

    def test_error_message_points_at_the_config_file(self):
        with pytest.raises(ValidationError, match="scoring_weights.yaml"):
            DraftNowWeights(player_score=0.5)

    def test_negative_weights_are_rejected(self):
        with pytest.raises(ValidationError):
            PlayerScoreWeights(projection=-0.1, vbd=0.6, opportunity=0.15,
                               offense_environment=0.1, schedule=0.075, risk=0.175)

    def test_strategy_priors_sum_to_one(self):
        assert sum(StrategyPriors().as_dict().values()) == pytest.approx(1.0)
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            StrategyPriors(balanced=0.9, hero_rb=0.9)

    def test_replacement_defaults_are_position_aware(self):
        rep = WeightsConfig().replacement
        # RBs are hoarded harder than tight ends.
        assert rep.bench_multiplier["RB"] > rep.bench_multiplier["TE"]
        assert 0.0 <= rep.blend_weight <= 1.0


class TestPathsAndLoading:
    def test_paths_derive_from_data_dir(self, tmp_path):
        paths = Paths.resolve(tmp_path)
        assert paths.raw_dir == paths.data_dir / "raw"
        assert paths.league_file.name == "league.yaml"

    def test_env_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FF_DATA_DIR", str(tmp_path / "elsewhere"))
        paths = Paths.resolve(tmp_path)
        assert paths.data_dir == tmp_path / "elsewhere"

    def test_ensure_dirs_is_idempotent(self, tmp_path):
        paths = Paths.resolve(tmp_path)
        paths.ensure_dirs()
        paths.ensure_dirs()
        assert paths.cache_dir.is_dir()

    def test_repository_config_loads(self):
        """The committed YAML must always be valid — this guards the example file."""
        cfg = load_config()
        assert cfg.league.teams >= 2
        assert sum(cfg.weights.player_score.as_dict().values()) == pytest.approx(1.0)

    def test_invalid_yaml_raises_config_error(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "league.yaml").write_text("teams: [this is not a number]\n")
        monkeypatch.setenv("FF_CONFIG_DIR", str(config_dir))
        monkeypatch.setenv("FF_DATA_DIR", str(tmp_path / "data"))
        with pytest.raises(ConfigError, match="league.yaml is invalid"):
            AppConfig.load(tmp_path)

    def test_falls_back_to_example_when_league_missing(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "league.example.yaml").write_text("teams: 14\n")
        monkeypatch.setenv("FF_CONFIG_DIR", str(config_dir))
        monkeypatch.setenv("FF_DATA_DIR", str(tmp_path / "data"))
        cfg = AppConfig.load(tmp_path)
        assert cfg.league.teams == 14
        assert cfg.league_file_exists is False


class TestScoredPositions:
    def test_kicker_and_dst_only_when_started(self, tmp_config: AppConfig):
        assert set(tmp_config.positions) == {"QB", "RB", "WR", "TE", "K", "DST"}

    def test_positions_drop_unused_slots(self, tmp_path, league):
        from fantasy_draft.config import DataSourcesConfig

        data = league.model_dump()
        data["roster"]["k"] = 0
        data["roster"]["dst"] = 0
        paths = Paths.resolve(tmp_path)
        cfg = AppConfig(
            paths=paths,
            league=LeagueConfig.model_validate(data),
            weights=WeightsConfig(),
            data_sources=DataSourcesConfig(),
            league_file_exists=True,
        )
        assert cfg.positions == ("QB", "RB", "WR", "TE")
