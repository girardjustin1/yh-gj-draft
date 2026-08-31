"""DuckDB schema, lineage columns, and write helpers."""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import polars as pl
import pytest

from fantasy_draft.database import SCHEMA_VERSION, Database, connect


class TestSchema:
    def test_initialize_creates_every_table(self, db: Database):
        tables = set(db.table_names())
        expected = {
            "players", "player_ids", "unresolved_players", "teams", "schedules", "byes",
            "historical_player_stats", "historical_team_stats", "snap_counts", "injuries",
            "depth_charts", "opportunity", "rankings", "projections", "adp_snapshots",
            "defense_vs_position", "team_offense_scores", "player_scores",
            "drafts", "draft_picks", "draft_rosters", "recommendations",
            "data_refresh_log", "schema_meta",
        }
        assert expected <= tables

    def test_initialize_is_idempotent(self, db: Database):
        before = db.table_names()
        db.initialize()
        assert db.table_names() == before

    def test_schema_version_recorded(self, db: Database):
        assert db.schema_version == SCHEMA_VERSION

    @pytest.mark.parametrize(
        "table",
        ["players", "player_ids", "schedules", "historical_player_stats",
         "opportunity", "rankings", "projections", "injuries"],
    )
    def test_ingested_tables_retain_lineage(self, db: Database, table: str):
        """Source lineage must survive normalization — see the spec's data rules."""
        columns = set(db.column_names(table))
        assert "source" in columns
        assert "ingested_at" in columns

    def test_connect_initializes_by_default(self, tmp_path):
        with connect(tmp_path / "x.duckdb") as database:
            assert database.schema_version == SCHEMA_VERSION


class TestWrites:
    def _team_frame(self, n: int = 3) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "team_abbr": [f"T{i}" for i in range(n)],
                "team_name": [f"Team {i}" for i in range(n)],
                "source": ["nflverse"] * n,
                "ingested_at": [datetime.now()] * n,
            }
        )

    def test_replace_table_fills_missing_columns_with_null(self, db: Database):
        assert db.replace_table("teams", self._team_frame()) == 3
        out = db.query("SELECT * FROM teams ORDER BY team_abbr")
        assert out.height == 3
        assert out["team_conf"].null_count() == 3

    def test_replace_table_drops_unexpected_columns(self, db: Database):
        """A vendor adding a column must not reshape our table."""
        frame = self._team_frame().with_columns(pl.lit("surprise").alias("new_vendor_field"))
        db.replace_table("teams", frame)
        assert "new_vendor_field" not in db.column_names("teams")

    def test_replace_table_is_a_full_replacement(self, db: Database):
        db.replace_table("teams", self._team_frame(5))
        db.replace_table("teams", self._team_frame(2))
        assert db.row_count("teams") == 2

    def test_replace_unknown_table_raises(self, db: Database):
        with pytest.raises(KeyError, match="unknown table"):
            db.replace_table("not_a_table", self._team_frame())

    def test_upsert_replaces_only_matching_keys(self, db: Database):
        db.replace_table("teams", self._team_frame(3))
        update = pl.DataFrame(
            {
                "team_abbr": ["T1"],
                "team_name": ["Renamed"],
                "source": ["manual"],
                "ingested_at": [datetime.now()],
            }
        )
        db.upsert_table("teams", update, keys=["team_abbr"])
        assert db.row_count("teams") == 3
        assert db.scalar("SELECT team_name FROM teams WHERE team_abbr = 'T1'") == "Renamed"

    def test_failed_write_rolls_back(self, db: Database):
        """A rejected refresh must leave the previous data intact, not a half-written table."""
        db.replace_table("teams", self._team_frame(3))
        duplicate_keys = pl.DataFrame(
            {
                "team_abbr": ["DUP", "DUP"],
                "team_name": ["a", "b"],
                "source": ["nflverse"] * 2,
                "ingested_at": [datetime.now()] * 2,
            }
        )
        with pytest.raises(duckdb.Error):
            db.replace_table("teams", duplicate_keys)
        assert db.row_count("teams") == 3
        assert set(db.query("SELECT team_abbr FROM teams")["team_abbr"]) == {"T0", "T1", "T2"}


class TestRefreshLog:
    def test_log_and_read_back(self, db: Database):
        db.log_refresh("nflverse_players", "players", "ok", rows=42, seasons="2026")
        entry = db.last_refresh("nflverse_players")
        assert entry is not None
        assert entry["rows"] == 42
        assert entry["table_name"] == "players"

    def test_failures_are_recorded_but_not_returned_as_last_success(self, db: Database):
        db.log_refresh("adp", "adp_snapshots", "ok", rows=10)
        db.log_refresh("adp", "adp_snapshots", "failed", message="HTTP 503")
        entry = db.last_refresh("adp")
        assert entry["status"] == "ok"
        assert entry["rows"] == 10
        assert db.row_count("data_refresh_log") == 2

    def test_unknown_source_returns_none(self, db: Database):
        assert db.last_refresh("never_ran") is None

    def test_most_recent_success_wins(self, db: Database):
        db.log_refresh("x", "players", "ok", rows=1)
        db.log_refresh("x", "players", "ok", rows=2)
        assert db.last_refresh("x")["rows"] == 2


class TestFreshness:
    def test_staleness_uses_configured_window(self):
        from fantasy_draft.models import DataFreshness

        fresh = DataFreshness(
            source="s", updated_at=datetime.now() - timedelta(hours=1), max_age_hours=24
        )
        stale = DataFreshness(
            source="s", updated_at=datetime.now() - timedelta(hours=48), max_age_hours=24
        )
        never = DataFreshness(source="s")
        assert not fresh.is_stale
        assert stale.is_stale
        assert never.is_stale  # unknown must never be reported as fresh
        assert never.age_hours is None
