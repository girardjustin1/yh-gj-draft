"""Schedule strength, per position.

Every opponent on a player's schedule is scored by how much that defence concedes to his
position, then averaged over the windows that matter to a fantasy manager:

* **Weeks 1-4** — the start, where a fast beginning changes waiver and trade behaviour
* **Weeks 1-8** — the first half
* **Fantasy regular season** — the weeks our league actually plays, from `league.yaml`
* **Fantasy playoffs** — the weeks that decide the title

Bye weeks are excluded rather than scored as neutral.

**This is a supporting factor, not a driver, and the code says so.** Preseason schedule
strength is genuinely weak evidence: it rests on last year's defensive personnel, and it
assumes the schedule itself is the thing that varies rather than the teams. It carries
7.5% of the Player Score and a capped confidence, so it shifts close calls and nothing
more. The alternative — presenting a confident-looking schedule number — would be worse
than omitting it.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..config import AppConfig
from ..constants import OFFENSE_POSITIONS
from ..database import Database
from ..logging import get_logger

log = get_logger(__name__)

#: Windows reported for every team and position.
WINDOWS: dict[str, tuple[int, int]] = {
    "weeks_1_4": (1, 4),
    "weeks_1_8": (1, 8),
}


@dataclass(frozen=True, slots=True)
class ScheduleWindows:
    """The week ranges used, resolved from the league configuration."""

    regular_season: list[int]
    playoffs: list[int]

    @classmethod
    def from_config(cls, cfg: AppConfig) -> ScheduleWindows:
        return cls(
            regular_season=list(cfg.league.regular_season_weeks),
            playoffs=list(cfg.league.playoff_weeks),
        )


def team_opponents(db: Database, season: int) -> pl.DataFrame:
    """``team, week, opponent`` for one season's regular season."""
    return db.query(
        """
        SELECT home_team AS team, week, away_team AS opponent
        FROM schedules WHERE season = ? AND (game_type = 'REG' OR game_type IS NULL)
        UNION ALL
        SELECT away_team AS team, week, home_team AS opponent
        FROM schedules WHERE season = ? AND (game_type = 'REG' OR game_type IS NULL)
        """,
        [season, season],
    )


def schedule_scores(db: Database, cfg: AppConfig) -> pl.DataFrame:
    """Position-specific schedule strength for every team, 0-100 (100 = easiest).

    Returns one row per (team, position) with a score for each window.
    """
    from .defense_vs_position import defense_vs_position

    season = cfg.league.season
    opponents = team_opponents(db, season)
    if opponents.is_empty():
        log.warning("no schedule available", extra={"season": season})
        return pl.DataFrame(
            schema={"team": pl.Utf8, "position": pl.Utf8, "schedule_score": pl.Float64,
                    "schedule_confidence": pl.Float64}
        )

    dvp = defense_vs_position(db, cfg)
    if dvp.is_empty():
        log.warning("no defense-vs-position data; schedule scores unavailable")
        return pl.DataFrame(
            schema={"team": pl.Utf8, "position": pl.Utf8, "schedule_score": pl.Float64,
                    "schedule_confidence": pl.Float64}
        )

    matchups = opponents.join(
        dvp.select(
            pl.col("team").alias("opponent"),
            "position",
            pl.col("score").alias("matchup_score"),
            pl.col("confidence").alias("matchup_confidence"),
        ),
        on="opponent",
        how="inner",
    )
    if matchups.is_empty():
        return pl.DataFrame(
            schema={"team": pl.Utf8, "position": pl.Utf8, "schedule_score": pl.Float64,
                    "schedule_confidence": pl.Float64}
        )

    windows = ScheduleWindows.from_config(cfg)
    aggregates: list[pl.Expr] = []
    for name, (lo, hi) in WINDOWS.items():
        aggregates.append(
            pl.col("matchup_score")
            .filter((pl.col("week") >= lo) & (pl.col("week") <= hi))
            .mean()
            .alias(name)
        )
    aggregates.append(
        pl.col("matchup_score")
        .filter(pl.col("week").is_in(windows.regular_season))
        .mean()
        .alias("regular_season")
    )
    aggregates.append(
        pl.col("matchup_score")
        .filter(pl.col("week").is_in(windows.playoffs))
        .mean()
        .alias("playoffs")
    )
    aggregates.append(pl.col("matchup_score").mean().alias("full_season"))
    aggregates.append(pl.len().alias("games"))
    aggregates.append(pl.col("matchup_confidence").mean().alias("matchup_confidence"))

    grouped = matchups.group_by(["team", "position"]).agg(aggregates)

    # The headline score leans on the weeks our league actually plays, with the playoff
    # stretch carrying real weight because that is where a season is decided.
    grouped = grouped.with_columns(
        (
            0.6 * pl.col("regular_season").fill_null(pl.col("full_season"))
            + 0.25 * pl.col("playoffs").fill_null(pl.col("full_season"))
            + 0.15 * pl.col("weeks_1_4").fill_null(pl.col("full_season"))
        ).alias("schedule_score")
    )

    return grouped.with_columns(
        # Capped low on purpose: see the module docstring. Schedule is a tiebreaker.
        (pl.col("matchup_confidence") * 0.8).clip(0, 0.5).alias("schedule_confidence"),
        pl.col("schedule_score").alias("schedule_raw"),
    ).select(
        "team", "position", "schedule_score", "schedule_raw", "schedule_confidence",
        "weeks_1_4", "weeks_1_8", "regular_season", "playoffs", "full_season", "games",
    )


def attach_schedule(board: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    """Join schedule scores onto the board, leaving unknowns at zero confidence."""
    if board.is_empty():
        return board
    if scores.is_empty():
        return board.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("schedule_score"),
            pl.lit(None, dtype=pl.Float64).alias("schedule_raw"),
            pl.lit(0.0).alias("schedule_confidence"),
        )
    joined = board.join(
        scores.select(
            "team", "position", "schedule_score", "schedule_raw", "schedule_confidence",
            "playoffs", "weeks_1_4",
        ),
        on=["team", "position"],
        how="left",
    )
    return joined.with_columns(pl.col("schedule_confidence").fill_null(0.0))


def bye_week_conflicts(db: Database, cfg: AppConfig, player_keys: list[str]) -> dict[int, int]:
    """How many of these players share each bye week. Used by roster advice."""
    if not player_keys:
        return {}
    frame = db.query(
        f"""
        SELECT b.bye_week, count(*) AS n
        FROM players p JOIN byes b ON b.team = p.team AND b.season = ?
        WHERE p.player_key IN ({', '.join('?' for _ in player_keys)})
          AND b.bye_week IS NOT NULL
        GROUP BY b.bye_week
        """,
        [cfg.league.season, *player_keys],
    )
    return dict(zip(frame["bye_week"], frame["n"], strict=True)) if not frame.is_empty() else {}


def positions_with_schedule(scores: pl.DataFrame) -> tuple[str, ...]:
    if scores.is_empty():
        return ()
    present = set(scores["position"])
    return tuple(p for p in OFFENSE_POSITIONS if p in present)
