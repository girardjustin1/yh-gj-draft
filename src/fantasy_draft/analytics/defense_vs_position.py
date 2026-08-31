"""Defense vs Position: how much does each defence give up to each fantasy position?

Computed from our own weekly data in *our* league's scoring, rather than borrowed from a
site's PPR table — a defence that concedes catches matters more in full PPR than in
standard, and the calculation should reflect that.

**How much this is worth knowing, honestly.** Preseason defence-vs-position is weak
evidence. Personnel turns over, schemes change, and a defence's rank against tight ends
one year correlates poorly with the next. It is included because it is a real signal at
the margin, and it is weighted at 7.5% of the Player Score for exactly that reason.
Confidence is set low and reported, rather than the number being presented as known.

Recent seasons are weighted more heavily via ``season_recency_halflife``.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from ..config import AppConfig
from ..constants import OFFENSE_POSITIONS
from ..database import Database
from ..logging import get_logger
from .fantasy_points import score_weekly_stats

log = get_logger(__name__)

#: Minimum team-games behind a figure before it is treated as meaningful.
MIN_GAMES = 8


def defense_vs_position(db: Database, cfg: AppConfig) -> pl.DataFrame:
    """Fantasy points allowed per game, by defence and position, scored in our rules.

    Returns one row per (team, position) with a 0-100 ``score`` where **100 is the most
    favourable matchup** for the offensive player.
    """
    weekly = db.query(
        """
        SELECT * FROM historical_player_stats
        WHERE season_type = 'REG' AND opponent IS NOT NULL AND opponent <> 'FA'
          AND position IN ('QB', 'RB', 'WR', 'TE')
        """
    )
    if weekly.is_empty():
        return pl.DataFrame(
            schema={
                "team": pl.Utf8, "position": pl.Utf8, "season": pl.Int32,
                "fantasy_points_allowed_pg": pl.Float64, "score": pl.Float64,
                "confidence": pl.Float64,
            }
        )

    scored = score_weekly_stats(weekly, cfg.league.scoring)

    # Points conceded by the defence, per game, to each position as a group.
    per_game = (
        scored.group_by(["opponent", "position", "season", "week"])
        .agg(pl.col("fantasy_points_league").sum().alias("points_allowed"))
        .rename({"opponent": "team"})
    )

    latest = int(per_game["season"].max())
    halflife = cfg.weights.season_recency_halflife
    per_game = per_game.with_columns(
        (0.5 ** ((latest - pl.col("season")) / halflife)).alias("weight")
    )

    aggregated = per_game.group_by(["team", "position"]).agg(
        (
            (pl.col("points_allowed") * pl.col("weight")).sum() / pl.col("weight").sum()
        ).alias("fantasy_points_allowed_pg"),
        pl.len().alias("games"),
        pl.col("season").n_unique().alias("seasons"),
        pl.col("season").max().alias("last_season"),
        pl.col("points_allowed").std().alias("points_allowed_sd"),
    )

    # Rank within position: giving up 20 points a game to receivers means something very
    # different from giving up 20 to tight ends.
    scored_frame = aggregated.with_columns(
        (
            pl.col("fantasy_points_allowed_pg").rank("average").over("position")
            / pl.len().over("position")
            * 100.0
        ).alias("score")
    )

    return scored_frame.with_columns(
        # Deliberately capped well below 1.0: this is a weak preseason signal and the
        # score should never be treated as a known quantity.
        (
            pl.when(pl.col("games") < MIN_GAMES).then(0.1)
            .otherwise(0.45 * (pl.col("seasons") / 3.0).clip(0.4, 1.0))
        ).alias("confidence"),
        pl.lit(latest, dtype=pl.Int32).alias("season"),
        pl.lit(datetime.now(), dtype=pl.Datetime).alias("computed_at"),
    ).select(
        "team", "position", "season", "fantasy_points_allowed_pg", "points_allowed_sd",
        "games", "seasons", "score", "confidence", "computed_at",
    )


def store_defense_vs_position(db: Database, cfg: AppConfig) -> int:
    """Compute and persist defence-vs-position scores."""
    frame = defense_vs_position(db, cfg)
    if frame.is_empty():
        log.warning("no defense-vs-position data computed")
        return 0
    stored = frame.select(
        "team", "position", "season",
        pl.lit(None, dtype=pl.Float64).alias("fantasy_points_allowed"),
        pl.col("fantasy_points_allowed_pg"),
        pl.lit(None, dtype=pl.Float64).alias("epa_allowed"),
        pl.lit(None, dtype=pl.Float64).alias("success_rate_allowed"),
        pl.lit(None, dtype=pl.Float64).alias("explosive_rate_allowed"),
        pl.col("games").cast(pl.Int32),
        "score", "confidence", "computed_at",
    )
    return db.replace_table("defense_vs_position", stored)


def positions_covered(frame: pl.DataFrame) -> tuple[str, ...]:
    if frame.is_empty():
        return ()
    return tuple(p for p in OFFENSE_POSITIONS if p in set(frame["position"]))
