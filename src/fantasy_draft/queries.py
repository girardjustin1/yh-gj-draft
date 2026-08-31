"""Read-side helpers over DuckDB.

Shared by the CLI, the future local API, and the MCP tool layer, so all three answer
the same question the same way. Everything here is read-only and returns Polars frames
or plain dicts — nothing in this module writes.
"""

from __future__ import annotations

import polars as pl

from .constants import DRAFTABLE_POSITIONS
from .database import Database
from .normalization.players import normalize_name, normalize_position


def search_players(
    db: Database,
    query: str,
    position: str | None = None,
    limit: int = 25,
    fantasy_only: bool = True,
) -> pl.DataFrame:
    """Find players by name.

    Matching is layered: exact normalized name, then prefix, then substring. Results are
    ordered by best available fantasy ranking so the relevant Josh Allen comes first.
    """
    normalized = normalize_name(query)
    canonical_position = normalize_position(position) if position else None

    filters = ["1 = 1"]
    params: list[object] = []
    if canonical_position:
        filters.append("p.position = ?")
        params.append(canonical_position)
    if fantasy_only:
        placeholders = ", ".join("?" for _ in DRAFTABLE_POSITIONS)
        filters.append(f"p.position IN ({placeholders})")
        params.extend(DRAFTABLE_POSITIONS)

    sql = f"""
        WITH ecr AS (
            SELECT player_key, min(ecr) AS ecr
            FROM rankings WHERE ranking_type = 'redraft-overall'
            GROUP BY player_key
        )
        SELECT p.player_key, p.full_name, p.position, p.team, p.status,
               p.years_experience, p.rookie_season, b.bye_week, e.ecr,
               CASE
                   WHEN p.normalized_name = ? THEN 0
                   WHEN p.normalized_name LIKE ? THEN 1
                   ELSE 2
               END AS match_rank
        FROM players p
        LEFT JOIN ecr e USING (player_key)
        LEFT JOIN byes b ON b.team = p.team
        WHERE {' AND '.join(filters)}
          AND p.normalized_name LIKE ?
        ORDER BY match_rank, e.ecr NULLS LAST, p.full_name
        LIMIT ?
    """
    return db.query(
        sql,
        [normalized, f"{normalized}%", *params, f"%{normalized}%", limit],
    )


def get_player(db: Database, player_key: str) -> dict | None:
    """Full identity record for one player, or None."""
    frame = db.query(
        """
        SELECT p.*, b.bye_week,
               i.sleeper_id, i.espn_id, i.yahoo_id, i.fantasypros_id, i.pfr_id
        FROM players p
        LEFT JOIN player_ids i USING (player_key)
        LEFT JOIN byes b ON b.team = p.team
        WHERE p.player_key = ?
        """,
        [player_key],
    )
    return frame.to_dicts()[0] if frame.height else None


def player_rankings(db: Database, player_key: str) -> pl.DataFrame:
    """Every ranking page this player appears on, with the ECR spread."""
    return db.query(
        """
        SELECT ranking_type, ecr, sd, best, worst, source, source_updated_at
        FROM rankings WHERE player_key = ? ORDER BY ranking_type
        """,
        [player_key],
    )


def player_season_stats(db: Database, player_key: str) -> pl.DataFrame:
    """Per-season regular-season totals from the ingested weekly box scores."""
    return db.query(
        """
        SELECT season,
               count(*)                      AS games,
               sum(carries)                  AS carries,
               sum(rushing_yards)            AS rush_yards,
               sum(rushing_tds)              AS rush_tds,
               sum(targets)                  AS targets,
               sum(receptions)               AS receptions,
               sum(receiving_yards)          AS rec_yards,
               sum(receiving_tds)            AS rec_tds,
               sum(passing_yards)            AS pass_yards,
               sum(passing_tds)              AS pass_tds,
               avg(target_share)             AS target_share,
               sum(fantasy_points)           AS fantasy_points_std
        FROM historical_player_stats
        WHERE player_key = ? AND season_type = 'REG'
        GROUP BY season ORDER BY season DESC
        """,
        [player_key],
    )


def player_opportunity(db: Database, player_key: str) -> pl.DataFrame:
    """Per-season expected fantasy points versus actual — the luck/role split."""
    return db.query(
        """
        SELECT season,
               count(*)                            AS games,
               sum(total_fantasy_points_exp)       AS expected_points,
               sum(total_fantasy_points)           AS actual_points,
               sum(rush_attempt)                   AS carries,
               sum(rec_attempt)                    AS targets
        FROM opportunity
        WHERE player_key = ?
        GROUP BY season ORDER BY season DESC
        """,
        [player_key],
    )


def player_snaps(db: Database, player_key: str) -> pl.DataFrame:
    """Per-season average offensive snap share, as a 0-1 fraction (nflverse units)."""
    return db.query(
        """
        SELECT season, count(*) AS games, avg(offense_pct) AS snap_share
        FROM snap_counts WHERE player_key = ? AND game_type = 'REG'
        GROUP BY season ORDER BY season DESC
        """,
        [player_key],
    )


def latest_injury(db: Database, player_key: str) -> dict | None:
    """Most recent injury report row, or None if the player has never been listed."""
    frame = db.query(
        """
        SELECT season, week, report_status, report_primary, practice_status,
               practice_primary, ingested_at
        FROM injuries WHERE player_key = ?
        ORDER BY season DESC, week DESC LIMIT 1
        """,
        [player_key],
    )
    return frame.to_dicts()[0] if frame.height else None


def depth_chart_slot(db: Database, player_key: str) -> dict | None:
    """Where the latest depth chart puts this player."""
    frame = db.query(
        """
        SELECT team, pos_abb, pos_name, pos_rank, as_of
        FROM depth_charts WHERE player_key = ?
        ORDER BY as_of DESC, pos_rank LIMIT 1
        """,
        [player_key],
    )
    return frame.to_dicts()[0] if frame.height else None


def resolve_one(db: Database, query: str, position: str | None = None) -> dict | None:
    """Resolve a user-typed name to exactly one player, or None if it is ambiguous."""
    matches = search_players(db, query, position=position, limit=5)
    if matches.is_empty():
        return None
    if matches.height > 1 and matches["match_rank"][0] == matches["match_rank"][1]:
        exact = matches.filter(pl.col("match_rank") == 0)
        if exact.height != 1:
            return None
    return matches.to_dicts()[0]
