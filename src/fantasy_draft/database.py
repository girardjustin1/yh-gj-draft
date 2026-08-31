"""DuckDB persistence: connection management, schema, and lineage helpers.

Design rules that the rest of the codebase depends on:

* Every ingested table keeps ``source``, ``source_updated_at`` and ``ingested_at``.
  Normalization never destroys lineage — we add columns, we don't overwrite provenance.
* Refreshes are idempotent: a table is replaced wholesale within a transaction, and
  ``data_refresh_log`` records the outcome whether it succeeded or failed.
* The schema is deliberately wider than the draft MVP needs (weekly stats, injuries,
  snap counts) so in-season features can be added later without a migration.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from .logging import get_logger

if TYPE_CHECKING:
    import polars as pl

log = get_logger(__name__)

#: Bumped whenever SCHEMA_STATEMENTS changes in a way that requires a rebuild.
SCHEMA_VERSION = 1


SCHEMA_STATEMENTS: tuple[str, ...] = (
    # --- meta ---------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key            VARCHAR PRIMARY KEY,
        value          VARCHAR NOT NULL,
        updated_at     TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_refresh_log (
        id                 BIGINT,
        source             VARCHAR NOT NULL,
        table_name         VARCHAR,
        status             VARCHAR NOT NULL,     -- ok | failed | skipped
        rows               BIGINT,
        seasons            VARCHAR,
        message            VARCHAR,
        duration_seconds   DOUBLE,
        source_updated_at  TIMESTAMP,
        ingested_at        TIMESTAMP NOT NULL
    )
    """,
    "CREATE SEQUENCE IF NOT EXISTS seq_refresh_log_id START 1",
    # --- identity -----------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS players (
        player_key         VARCHAR PRIMARY KEY,
        gsis_id            VARCHAR,
        full_name          VARCHAR NOT NULL,
        normalized_name    VARCHAR NOT NULL,
        first_name         VARCHAR,
        last_name          VARCHAR,
        position           VARCHAR,
        position_group     VARCHAR,
        team               VARCHAR,
        status             VARCHAR,
        birth_date         DATE,
        height             INTEGER,
        weight             INTEGER,
        college            VARCHAR,
        rookie_season      INTEGER,
        last_season        INTEGER,
        years_experience   INTEGER,
        draft_year         INTEGER,
        draft_round        INTEGER,
        draft_pick         INTEGER,
        draft_team         VARCHAR,
        source             VARCHAR NOT NULL,
        source_updated_at  TIMESTAMP,
        ingested_at        TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_ids (
        player_key         VARCHAR NOT NULL,
        gsis_id            VARCHAR,
        sleeper_id         VARCHAR,
        espn_id            VARCHAR,
        yahoo_id           VARCHAR,
        fantasypros_id     VARCHAR,
        pfr_id             VARCHAR,
        mfl_id             VARCHAR,
        sportradar_id      VARCHAR,
        pff_id             VARCHAR,
        cbs_id             VARCHAR,
        rotowire_id        VARCHAR,
        ktc_id             VARCHAR,
        fantasy_data_id    VARCHAR,
        merge_name         VARCHAR,
        source             VARCHAR NOT NULL,
        source_updated_at  TIMESTAMP,
        ingested_at        TIMESTAMP NOT NULL,
        PRIMARY KEY (player_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS unresolved_players (
        source             VARCHAR NOT NULL,
        source_id          VARCHAR,
        raw_name           VARCHAR NOT NULL,
        normalized_name    VARCHAR,
        position           VARCHAR,
        team               VARCHAR,
        reason             VARCHAR NOT NULL,
        candidates         VARCHAR,
        seen_at            TIMESTAMP NOT NULL
    )
    """,
    # --- league / schedule --------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS teams (
        team_abbr          VARCHAR PRIMARY KEY,
        team_name          VARCHAR,
        team_nick          VARCHAR,
        team_conf          VARCHAR,
        team_division      VARCHAR,
        team_color         VARCHAR,
        source             VARCHAR NOT NULL,
        ingested_at        TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS schedules (
        game_id            VARCHAR,
        season             INTEGER NOT NULL,
        game_type          VARCHAR,
        week               INTEGER NOT NULL,
        gameday            DATE,
        home_team          VARCHAR,
        away_team          VARCHAR,
        home_score         INTEGER,
        away_score         INTEGER,
        spread_line        DOUBLE,
        total_line         DOUBLE,
        div_game           BOOLEAN,
        roof               VARCHAR,
        surface            VARCHAR,
        source             VARCHAR NOT NULL,
        source_updated_at  TIMESTAMP,
        ingested_at        TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS byes (
        season             INTEGER NOT NULL,
        team               VARCHAR NOT NULL,
        bye_week           INTEGER,
        PRIMARY KEY (season, team)
    )
    """,
    # --- historical performance ---------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS historical_player_stats (
        player_key         VARCHAR,
        gsis_id            VARCHAR,
        season             INTEGER NOT NULL,
        week               INTEGER,
        season_type        VARCHAR,
        team               VARCHAR,
        opponent           VARCHAR,
        position           VARCHAR,
        games              INTEGER,
        completions        DOUBLE,
        attempts           DOUBLE,
        passing_yards      DOUBLE,
        passing_tds        DOUBLE,
        interceptions      DOUBLE,
        sacks              DOUBLE,
        carries            DOUBLE,
        rushing_yards      DOUBLE,
        rushing_tds        DOUBLE,
        targets            DOUBLE,
        receptions         DOUBLE,
        receiving_yards    DOUBLE,
        receiving_tds      DOUBLE,
        receiving_air_yards DOUBLE,
        target_share       DOUBLE,
        air_yards_share    DOUBLE,
        wopr               DOUBLE,
        fumbles_lost       DOUBLE,
        two_point_conv     DOUBLE,
        fantasy_points     DOUBLE,
        source             VARCHAR NOT NULL,
        source_updated_at  TIMESTAMP,
        ingested_at        TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS historical_team_stats (
        team               VARCHAR NOT NULL,
        season             INTEGER NOT NULL,
        week               INTEGER,
        season_type        VARCHAR,
        plays              DOUBLE,
        points             DOUBLE,
        pass_attempts      DOUBLE,
        rush_attempts      DOUBLE,
        passing_yards      DOUBLE,
        rushing_yards      DOUBLE,
        passing_tds        DOUBLE,
        rushing_tds        DOUBLE,
        source             VARCHAR NOT NULL,
        ingested_at        TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS snap_counts (
        player_key         VARCHAR,
        pfr_player_id      VARCHAR,
        player             VARCHAR,
        season             INTEGER NOT NULL,
        week               INTEGER,
        game_type          VARCHAR,
        team               VARCHAR,
        opponent           VARCHAR,
        position           VARCHAR,
        offense_snaps      DOUBLE,
        offense_pct        DOUBLE,
        source             VARCHAR NOT NULL,
        ingested_at        TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS injuries (
        player_key         VARCHAR,
        gsis_id            VARCHAR,
        full_name          VARCHAR,
        season             INTEGER NOT NULL,
        week               INTEGER,
        team               VARCHAR,
        position           VARCHAR,
        report_status      VARCHAR,
        report_primary     VARCHAR,
        practice_status    VARCHAR,
        practice_primary   VARCHAR,
        source             VARCHAR NOT NULL,
        source_updated_at  TIMESTAMP,
        ingested_at        TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS depth_charts (
        player_key         VARCHAR,
        gsis_id            VARCHAR,
        player_name        VARCHAR,
        team               VARCHAR,
        as_of              TIMESTAMP,
        pos_group          VARCHAR,
        pos_abb            VARCHAR,
        pos_name           VARCHAR,
        pos_rank           INTEGER,
        source             VARCHAR NOT NULL,
        ingested_at        TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS opportunity (
        player_key         VARCHAR,
        gsis_id            VARCHAR,
        full_name          VARCHAR,
        season             INTEGER NOT NULL,
        week               INTEGER,
        team               VARCHAR,
        position           VARCHAR,
        pass_attempt       DOUBLE,
        rush_attempt       DOUBLE,
        rec_attempt        DOUBLE,
        receptions         DOUBLE,
        rec_air_yards      DOUBLE,
        total_fantasy_points      DOUBLE,
        total_fantasy_points_exp  DOUBLE,
        rush_fantasy_points_exp   DOUBLE,
        rec_fantasy_points_exp    DOUBLE,
        pass_fantasy_points_exp   DOUBLE,
        total_touchdown_exp       DOUBLE,
        total_first_down_exp      DOUBLE,
        source             VARCHAR NOT NULL,
        source_updated_at  TIMESTAMP,
        ingested_at        TIMESTAMP NOT NULL
    )
    """,
    # --- market / projections -----------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS rankings (
        player_key         VARCHAR,
        source             VARCHAR NOT NULL,
        source_player_id   VARCHAR,
        player_name        VARCHAR NOT NULL,
        position           VARCHAR,
        team               VARCHAR,
        season             INTEGER NOT NULL,
        ranking_type       VARCHAR NOT NULL,     -- redraft-overall, redraft-rb, ...
        ecr                DOUBLE,
        sd                 DOUBLE,
        best               INTEGER,
        worst              INTEGER,
        bye_week           INTEGER,
        source_updated_at  TIMESTAMP,
        ingested_at        TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS projections (
        player_key         VARCHAR,
        source             VARCHAR NOT NULL,
        source_player_id   VARCHAR,
        player_name        VARCHAR NOT NULL,
        position           VARCHAR,
        team               VARCHAR,
        season             INTEGER NOT NULL,
        week               INTEGER,              -- NULL for season-long
        games              DOUBLE,
        pass_yards         DOUBLE,
        pass_tds           DOUBLE,
        interceptions      DOUBLE,
        rush_yards         DOUBLE,
        rush_tds           DOUBLE,
        receptions         DOUBLE,
        rec_yards          DOUBLE,
        rec_tds            DOUBLE,
        fumbles_lost       DOUBLE,
        fantasy_points     DOUBLE,               -- in OUR league scoring
        source_updated_at  TIMESTAMP,
        ingested_at        TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS adp_snapshots (
        player_key         VARCHAR,
        source             VARCHAR NOT NULL,
        source_player_id   VARCHAR,
        player_name        VARCHAR NOT NULL,
        position           VARCHAR,
        team               VARCHAR,
        season             INTEGER NOT NULL,
        scoring_format     VARCHAR,
        adp                DOUBLE,
        position_adp       DOUBLE,
        adp_sd             DOUBLE,
        adp_min            DOUBLE,
        adp_max            DOUBLE,
        sample_size        INTEGER,
        snapshot_at        TIMESTAMP NOT NULL,
        ingested_at        TIMESTAMP NOT NULL
    )
    """,
    # --- derived analytics --------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS defense_vs_position (
        team               VARCHAR NOT NULL,
        position           VARCHAR NOT NULL,
        season             INTEGER NOT NULL,
        fantasy_points_allowed      DOUBLE,
        fantasy_points_allowed_pg   DOUBLE,
        epa_allowed        DOUBLE,
        success_rate_allowed DOUBLE,
        explosive_rate_allowed DOUBLE,
        games              INTEGER,
        score              DOUBLE,              -- 0-100, 100 = easiest matchup
        confidence         DOUBLE,
        computed_at        TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS team_offense_scores (
        team               VARCHAR NOT NULL,
        season             INTEGER NOT NULL,
        plays_per_game     DOUBLE,
        points_per_game    DOUBLE,
        pass_rate          DOUBLE,
        epa_per_play       DOUBLE,
        red_zone_rate      DOUBLE,
        score              DOUBLE,
        confidence         DOUBLE,
        computed_at        TIMESTAMP NOT NULL,
        PRIMARY KEY (team, season)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_scores (
        player_key         VARCHAR NOT NULL,
        season             INTEGER NOT NULL,
        position           VARCHAR,
        projected_points   DOUBLE,
        replacement_points DOUBLE,
        vbd                DOUBLE,
        positional_rank    INTEGER,
        overall_rank       INTEGER,
        tier               INTEGER,
        tier_rank          INTEGER,
        points_to_next_player DOUBLE,
        points_to_next_tier   DOUBLE,
        tier_cliff_score   DOUBLE,
        opportunity_score  DOUBLE,
        offense_score      DOUBLE,
        schedule_score     DOUBLE,
        risk_score         DOUBLE,
        market_value_score DOUBLE,
        scarcity_score     DOUBLE,
        player_score       DOUBLE,
        value_score        DOUBLE,
        components         VARCHAR,             -- JSON blob of ComponentScore detail
        confidence         DOUBLE,
        computed_at        TIMESTAMP NOT NULL,
        PRIMARY KEY (player_key, season)
    )
    """,
    # --- live draft ---------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS drafts (
        draft_id           VARCHAR PRIMARY KEY,
        platform           VARCHAR NOT NULL,
        league_id          VARCHAR,
        season             INTEGER,
        draft_type         VARCHAR,
        teams              INTEGER,
        rounds             INTEGER,
        my_slot            INTEGER,
        my_team_id         VARCHAR,
        status             VARCHAR,
        settings           VARCHAR,             -- JSON blob from the platform
        synced_at          TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS draft_picks (
        draft_id           VARCHAR NOT NULL,
        overall            INTEGER NOT NULL,
        round              INTEGER NOT NULL,
        slot               INTEGER NOT NULL,
        team_id            VARCHAR,
        player_key         VARCHAR,
        source_player_id   VARCHAR,
        player_name        VARCHAR,
        position           VARCHAR,
        nfl_team           VARCHAR,
        is_keeper          BOOLEAN,
        picked_at          TIMESTAMP,
        synced_at          TIMESTAMP NOT NULL,
        PRIMARY KEY (draft_id, overall)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS draft_rosters (
        draft_id           VARCHAR NOT NULL,
        team_id            VARCHAR NOT NULL,
        slot               INTEGER,
        is_me              BOOLEAN,
        display_name       VARCHAR,
        synced_at          TIMESTAMP NOT NULL,
        PRIMARY KEY (draft_id, team_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recommendations (
        draft_id           VARCHAR,
        overall_pick       INTEGER NOT NULL,
        generated_at       TIMESTAMP NOT NULL,
        primary_player_key VARCHAR,
        payload            VARCHAR NOT NULL,    -- full Recommendation JSON
        confidence         DOUBLE
    )
    """,
)


class Database:
    """Thin wrapper over a DuckDB connection with lineage-aware write helpers."""

    def __init__(self, path: Path | str, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._conn: duckdb.DuckDBPyConnection | None = None

    # --- lifecycle ----------------------------------------------------------------

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(str(self.path), read_only=self.read_only)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- schema -------------------------------------------------------------------

    def initialize(self) -> None:
        """Create every table if missing. Safe to run repeatedly."""
        for statement in SCHEMA_STATEMENTS:
            self.conn.execute(statement)
        self.set_meta("schema_version", str(SCHEMA_VERSION))
        log.info("schema initialized", extra={"path": str(self.path), "version": SCHEMA_VERSION})

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO schema_meta (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT (key) DO UPDATE SET value = excluded.value,
                                            updated_at = excluded.updated_at
            """,
            [key, value, datetime.now()],
        )

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM schema_meta WHERE key = ?", [key]).fetchone()
        return row[0] if row else None

    @property
    def schema_version(self) -> int | None:
        try:
            raw = self.get_meta("schema_version")
        except duckdb.Error:
            return None
        return int(raw) if raw is not None else None

    def table_names(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        return [r[0] for r in rows]

    def table_exists(self, name: str) -> bool:
        return name in set(self.table_names())

    def row_count(self, name: str) -> int:
        if not self.table_exists(name):
            return 0
        row = self.conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()
        return int(row[0]) if row else 0

    def table_counts(self) -> dict[str, int]:
        return {name: self.row_count(name) for name in self.table_names()}

    # --- reads --------------------------------------------------------------------

    def query(self, sql: str, params: list[Any] | None = None) -> pl.DataFrame:
        """Run SQL and return a Polars DataFrame."""
        return self.conn.execute(sql, params or []).pl()

    def scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        row = self.conn.execute(sql, params or []).fetchone()
        return row[0] if row else None

    # --- writes -------------------------------------------------------------------

    @contextlib.contextmanager
    def transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
        self.conn.execute("BEGIN TRANSACTION")
        try:
            yield self.conn
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    def replace_table(self, name: str, frame: pl.DataFrame) -> int:
        """Atomically replace ``name`` with ``frame``, preserving the declared schema.

        Columns absent from the frame are filled with NULL; extra frame columns are
        dropped so a vendor schema change cannot silently reshape our tables.
        """
        import polars as pl_

        if not self.table_exists(name):
            raise KeyError(f"unknown table {name!r}; add it to SCHEMA_STATEMENTS first")

        expected = self.column_names(name)
        aligned = frame.select(
            [
                pl_.col(c) if c in frame.columns else pl_.lit(None).alias(c)
                for c in expected
            ]
        )
        with self.transaction() as conn:
            conn.register("_incoming", aligned)
            conn.execute(f'DELETE FROM "{name}"')
            conn.execute(f'INSERT INTO "{name}" SELECT * FROM _incoming')
            conn.unregister("_incoming")
        return aligned.height

    def upsert_table(self, name: str, frame: pl.DataFrame, keys: list[str]) -> int:
        """Insert ``frame`` into ``name``, first deleting rows matching on ``keys``."""
        import polars as pl_

        expected = self.column_names(name)
        aligned = frame.select(
            [
                pl_.col(c) if c in frame.columns else pl_.lit(None).alias(c)
                for c in expected
            ]
        )
        key_sql = " AND ".join(f't."{k}" = i."{k}"' for k in keys)
        with self.transaction() as conn:
            conn.register("_incoming", aligned)
            conn.execute(
                f'DELETE FROM "{name}" t WHERE EXISTS '
                f"(SELECT 1 FROM _incoming i WHERE {key_sql})"
            )
            conn.execute(f'INSERT INTO "{name}" SELECT * FROM _incoming')
            conn.unregister("_incoming")
        return aligned.height

    def column_names(self, name: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
            [name],
        ).fetchall()
        return [r[0] for r in rows]

    # --- refresh log --------------------------------------------------------------

    def log_refresh(
        self,
        source: str,
        table_name: str | None,
        status: str,
        rows: int | None = None,
        seasons: str | None = None,
        message: str | None = None,
        duration_seconds: float | None = None,
        source_updated_at: datetime | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO data_refresh_log
                (id, source, table_name, status, rows, seasons, message,
                 duration_seconds, source_updated_at, ingested_at)
            VALUES (nextval('seq_refresh_log_id'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [source, table_name, status, rows, seasons, message,
             duration_seconds, source_updated_at, datetime.now()],
        )

    def last_refresh(self, source: str) -> dict[str, Any] | None:
        """Most recent *successful* refresh for a source."""
        row = self.conn.execute(
            """
            SELECT source, table_name, status, rows, message, ingested_at, source_updated_at
            FROM data_refresh_log
            WHERE source = ? AND status = 'ok'
            ORDER BY ingested_at DESC LIMIT 1
            """,
            [source],
        ).fetchone()
        if row is None:
            return None
        return {
            "source": row[0], "table_name": row[1], "status": row[2], "rows": row[3],
            "message": row[4], "ingested_at": row[5], "source_updated_at": row[6],
        }


def connect(path: Path | str, read_only: bool = False, initialize: bool = True) -> Database:
    """Open the database, creating the schema unless told otherwise."""
    db = Database(path, read_only=read_only)
    if initialize and not read_only:
        db.initialize()
    return db
