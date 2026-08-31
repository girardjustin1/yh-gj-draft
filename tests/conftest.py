"""Shared pytest fixtures.

Tests must never require a live Sleeper league or a warm nflverse cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fantasy_draft.config import AppConfig, DataSourcesConfig, LeagueConfig, Paths, WeightsConfig
from fantasy_draft.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def league() -> LeagueConfig:
    """The canonical test league: 12-team half-PPR snake, our slot is 7."""
    return LeagueConfig.model_validate(
        {
            "name": "Test League",
            "season": 2026,
            "teams": 12,
            "draft": {"type": "snake", "rounds": 15, "slot": 7},
            "roster": {
                "qb": 1, "rb": 2, "wr": 2, "te": 1, "flex": 1,
                "k": 1, "dst": 1, "bench": 6, "ir": 1,
            },
            "scoring": {"reception": 0.5},
        }
    )


@pytest.fixture
def superflex_league(league: LeagueConfig) -> LeagueConfig:
    data = league.model_dump()
    data["roster"]["superflex"] = 1
    data["roster"]["bench"] = 5
    return LeagueConfig.model_validate(data)


@pytest.fixture
def weights() -> WeightsConfig:
    return WeightsConfig()


@pytest.fixture
def tmp_config(tmp_path: Path, league: LeagueConfig) -> AppConfig:
    """An AppConfig pointed at an isolated tmp data directory."""
    paths = Paths(
        root=tmp_path,
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
        db_path=tmp_path / "data" / "fantasy.duckdb",
    )
    paths.ensure_dirs()
    return AppConfig(
        paths=paths,
        league=league,
        weights=WeightsConfig(),
        data_sources=DataSourcesConfig(),
        league_file_exists=True,
    )


@pytest.fixture
def db(tmp_config: AppConfig):
    """An initialized, empty DuckDB in a tmp directory."""
    database = Database(tmp_config.paths.db_path)
    database.initialize()
    yield database
    database.close()
