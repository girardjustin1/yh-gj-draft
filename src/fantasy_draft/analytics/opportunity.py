"""Opportunity Score: how good is the *role*, independent of how the ball bounced?

Fantasy points are noisy — touchdowns in particular are close to random given usage.
Opportunity measures what a player was *given*: snaps, carries, targets, air yards, and
above all expected fantasy points from the nflverse ff_opportunity model, which prices
each touch by its situation.

This is the component that catches a player whose box score understates his job, and
flags one whose box score was carried by touchdown luck that will not repeat.

Everything is measured **per game** and normalized **within position**, because a tight
end's target share is not comparable to a receiver's. Players with no history get zero
confidence rather than a zero score — a rookie is unknown, not bad.
"""

from __future__ import annotations

import polars as pl

from ..config import AppConfig
from ..database import Database


def _recency_weight(cfg: AppConfig, latest_season: int) -> pl.Expr:
    halflife = cfg.weights.season_recency_halflife
    return (0.5 ** ((latest_season - pl.col("season")) / halflife)).alias("weight")


def player_opportunity(db: Database, cfg: AppConfig) -> pl.DataFrame:
    """Per-player usage rates, recency weighted across the ingested seasons.

    Returns one row per player with per-game expected points, snap share, and the
    volume shares that make up a role, plus ``opportunity_confidence`` reflecting how
    much history backs the numbers.
    """
    opportunity = db.query(
        """
        SELECT player_key, season, position,
               count(*)                       AS games,
               sum(total_fantasy_points_exp)  AS exp_points,
               sum(total_fantasy_points)      AS actual_points,
               sum(rush_attempt)              AS carries,
               sum(rec_attempt)               AS targets,
               sum(rec_air_yards)             AS air_yards
        FROM opportunity WHERE player_key IS NOT NULL
        GROUP BY player_key, season, position
        """
    )
    if opportunity.is_empty():
        return pl.DataFrame(
            schema={"player_key": pl.Utf8, "opportunity_confidence": pl.Float64}
        )

    latest = int(opportunity["season"].max())
    opportunity = opportunity.with_columns(_recency_weight(cfg, latest))

    snaps = db.query(
        """
        SELECT player_key, season, avg(offense_pct) AS snap_share, count(*) AS snap_games
        FROM snap_counts WHERE player_key IS NOT NULL AND game_type = 'REG'
        GROUP BY player_key, season
        """
    )
    shares = db.query(
        """
        SELECT player_key, season,
               avg(target_share)     AS target_share,
               avg(air_yards_share)  AS air_yards_share,
               avg(wopr)             AS wopr
        FROM historical_player_stats
        WHERE player_key IS NOT NULL AND season_type = 'REG'
        GROUP BY player_key, season
        """
    )

    joined = opportunity.join(snaps, on=["player_key", "season"], how="left").join(
        shares, on=["player_key", "season"], how="left"
    )

    def weighted(column: str, per_game: bool = False) -> pl.Expr:
        value = (pl.col(column) / pl.col("games")) if per_game else pl.col(column)
        return (
            (value * pl.col("weight")).sum() / pl.col("weight").sum()
        ).alias(column if not per_game else f"{column}_per_game")

    aggregated = joined.group_by("player_key").agg(
        pl.col("position").last(),
        pl.col("games").sum().alias("career_games"),
        pl.col("season").n_unique().alias("seasons_played"),
        pl.col("season").max().alias("last_season"),
        weighted("exp_points", per_game=True),
        weighted("actual_points", per_game=True),
        weighted("carries", per_game=True),
        weighted("targets", per_game=True),
        weighted("air_yards", per_game=True),
        weighted("snap_share"),
        weighted("target_share"),
        weighted("air_yards_share"),
        weighted("wopr"),
    )

    # How much of the box score was role, and how much was fortune?
    aggregated = aggregated.with_columns(
        (pl.col("actual_points_per_game") - pl.col("exp_points_per_game"))
        .alias("points_over_expected_per_game")
    )

    # Confidence grows with sample and decays if the last real season is old.
    latest_overall = int(aggregated["last_season"].max())
    return aggregated.with_columns(
        (
            (pl.col("career_games") / 24.0).clip(0, 1) * 0.7
            + (pl.col("seasons_played") / 3.0).clip(0, 1) * 0.3
        ).mul(
            pl.when(pl.col("last_season") >= latest_overall).then(1.0)
            .when(pl.col("last_season") == latest_overall - 1).then(0.7)
            .otherwise(0.35)
        ).clip(0, 1).alias("opportunity_confidence")
    )


def opportunity_score(db: Database, cfg: AppConfig) -> pl.DataFrame:
    """Normalize opportunity into a 0-100 score, ranked within position.

    Expected points per game carries most of the weight because it already folds volume
    and situation together; snap share and the receiving-share family are included to
    separate two players with similar expected points but different role security.
    """
    frame = player_opportunity(db, cfg)
    if frame.is_empty():
        return frame

    def percentile(column: str) -> pl.Expr:
        return (
            pl.col(column).rank("average").over("position")
            / pl.len().over("position")
            * 100.0
        ).alias(f"pct_{column}")

    frame = frame.with_columns(
        percentile("exp_points_per_game"),
        percentile("snap_share"),
        percentile("target_share"),
        percentile("wopr"),
        percentile("carries_per_game"),
    )

    # Receiving-share signals are meaningless for a quarterback, so the blend is
    # position-aware rather than one formula applied everywhere.
    receiving = (
        pl.col("pct_exp_points_per_game") * 0.55
        + pl.col("pct_snap_share").fill_null(50.0) * 0.15
        + pl.col("pct_target_share").fill_null(50.0) * 0.15
        + pl.col("pct_wopr").fill_null(50.0) * 0.15
    )
    rushing = (
        pl.col("pct_exp_points_per_game") * 0.50
        + pl.col("pct_snap_share").fill_null(50.0) * 0.20
        + pl.col("pct_carries_per_game").fill_null(50.0) * 0.15
        + pl.col("pct_target_share").fill_null(50.0) * 0.15
    )
    passing = (
        pl.col("pct_exp_points_per_game") * 0.75
        + pl.col("pct_snap_share").fill_null(50.0) * 0.25
    )

    return frame.with_columns(
        pl.when(pl.col("position") == "QB").then(passing)
        .when(pl.col("position") == "RB").then(rushing)
        .otherwise(receiving)
        .clip(0, 100)
        .alias("opportunity_score")
    ).select(
        "player_key", "position", "opportunity_score", "opportunity_confidence",
        "exp_points_per_game", "actual_points_per_game", "points_over_expected_per_game",
        "snap_share", "target_share", "wopr", "career_games", "seasons_played",
        "last_season",
    )
