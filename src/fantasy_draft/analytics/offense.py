"""Offensive environment: how good is the offence this player plays in?

A strong role on a bad offence is worth less than the same role on a good one — fewer
plays, fewer red-zone trips, fewer points to go around. This scores each team's offence
from recent seasons: scoring rate, play volume, and pass/run balance.

**Limitation, stated plainly:** this is backward-looking. It cannot know about a coaching
change, a new quarterback, or an offensive-line rebuild. It is weighted at 10% of the
Player Score for exactly that reason, and its confidence drops for teams with less
recent data.
"""

from __future__ import annotations

import polars as pl

from ..config import AppConfig
from ..database import Database


def team_offense_scores(db: Database, cfg: AppConfig) -> pl.DataFrame:
    """Score every team's offence 0-100 from recent seasons."""
    team_stats = db.query(
        """
        SELECT team, season,
               count(*)            AS games,
               sum(plays)          AS plays,
               sum(pass_attempts)  AS pass_attempts,
               sum(rush_attempts)  AS rush_attempts,
               sum(passing_yards)  AS pass_yards,
               sum(rushing_yards)  AS rush_yards,
               sum(passing_tds)    AS pass_tds,
               sum(rushing_tds)    AS rush_tds
        FROM historical_team_stats
        WHERE season_type = 'REG' AND team <> 'FA'
        GROUP BY team, season
        """
    )
    if team_stats.is_empty():
        return pl.DataFrame(
            schema={"team": pl.Utf8, "offense_score": pl.Float64, "confidence": pl.Float64}
        )

    # Points scored come from the schedule, which records final scores; the team-stats
    # feed does not carry a points column.
    scores = db.query(
        """
        SELECT team, season, sum(points) AS points, count(*) AS scored_games FROM (
            SELECT home_team AS team, season, home_score AS points
            FROM schedules WHERE game_type = 'REG' AND home_score IS NOT NULL
            UNION ALL
            SELECT away_team AS team, season, away_score AS points
            FROM schedules WHERE game_type = 'REG' AND away_score IS NOT NULL
        ) GROUP BY team, season
        """
    )
    frame = team_stats.join(scores, on=["team", "season"], how="left")

    latest = int(frame["season"].max())
    halflife = cfg.weights.season_recency_halflife
    frame = frame.with_columns(
        (0.5 ** ((latest - pl.col("season")) / halflife)).alias("weight"),
        (pl.col("points") / pl.col("scored_games")).alias("points_per_game"),
        (pl.col("plays") / pl.col("games")).alias("plays_per_game"),
        (
            pl.col("pass_attempts") / (pl.col("pass_attempts") + pl.col("rush_attempts"))
        ).alias("pass_rate"),
        (
            (pl.col("pass_yards") + pl.col("rush_yards")) / pl.col("plays")
        ).alias("yards_per_play"),
    )

    def weighted(column: str) -> pl.Expr:
        return (
            (pl.col(column) * pl.col("weight")).sum() / pl.col("weight").sum()
        ).alias(column)

    aggregated = frame.group_by("team").agg(
        weighted("points_per_game"),
        weighted("plays_per_game"),
        weighted("pass_rate"),
        weighted("yards_per_play"),
        pl.col("season").n_unique().alias("seasons"),
        pl.col("season").max().alias("last_season"),
    )

    def percentile(column: str) -> pl.Expr:
        return (pl.col(column).rank("average") / pl.len() * 100.0).alias(f"pct_{column}")

    aggregated = aggregated.with_columns(
        percentile("points_per_game"),
        percentile("plays_per_game"),
        percentile("yards_per_play"),
    )
    return aggregated.with_columns(
        (
            pl.col("pct_points_per_game") * 0.55
            + pl.col("pct_yards_per_play") * 0.25
            + pl.col("pct_plays_per_game") * 0.20
        ).clip(0, 100).alias("offense_score"),
        (
            pl.when(pl.col("last_season") >= latest).then(0.75).otherwise(0.4)
            * (pl.col("seasons") / 3.0).clip(0.3, 1.0)
        ).alias("confidence"),
    ).select(
        "team", "offense_score", "confidence", "points_per_game", "plays_per_game",
        "pass_rate", "yards_per_play", "seasons",
    )
