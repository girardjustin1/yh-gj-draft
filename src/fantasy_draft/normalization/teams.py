"""Team canonicalization and bye weeks."""

from __future__ import annotations

import polars as pl

from ..constants import FREE_AGENT_TEAM, REGULAR_SEASON_WEEKS, TEAM_ALIASES


def canonical_team(team: str | None) -> str:
    """Map any historical or vendor team code to the current nflverse abbreviation.

    >>> canonical_team("OAK")
    'LV'
    >>> canonical_team(None)
    'FA'
    """
    if team is None:
        return FREE_AGENT_TEAM
    code = str(team).strip().upper()
    if not code or code in {"NONE", "NULL", "NAN", "FA", "-"}:
        return FREE_AGENT_TEAM
    return TEAM_ALIASES.get(code, code)


def canonical_team_expr(column: str) -> pl.Expr:
    """Polars expression form of :func:`canonical_team`, for whole columns."""
    upper = pl.col(column).cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    normalized = (
        pl.when(upper.is_null() | upper.is_in(["", "NONE", "NULL", "NAN", "-"]))
        .then(pl.lit(FREE_AGENT_TEAM))
        .otherwise(upper)
    )
    return normalized.replace(TEAM_ALIASES).alias(column)


def bye_weeks(schedules: pl.DataFrame, season: int) -> pl.DataFrame:
    """Derive each team's bye week from the regular-season schedule.

    A team's bye is the regular-season week in which it does not appear. Teams with no
    missing week (or more than one) get a null bye rather than a guess — the schedule
    can legitimately be incomplete for a future season.
    """
    games = schedules.filter(
        (pl.col("season") == season)
        & (pl.col("week") <= REGULAR_SEASON_WEEKS)
        & (pl.col("game_type").is_null() | (pl.col("game_type") == "REG"))
    )
    if games.is_empty():
        return pl.DataFrame(
            schema={"season": pl.Int64, "team": pl.Utf8, "bye_week": pl.Int64}
        )

    appearances = pl.concat(
        [
            games.select(pl.col("home_team").alias("team"), "week"),
            games.select(pl.col("away_team").alias("team"), "week"),
        ]
    ).with_columns(canonical_team_expr("team"))

    weeks_played = int(games["week"].max())
    all_weeks = pl.DataFrame({"week": list(range(1, weeks_played + 1))})
    teams = appearances.select("team").unique()

    missing = (
        teams.join(all_weeks, how="cross")
        .join(appearances.unique(), on=["team", "week"], how="anti")
        .group_by("team")
        .agg(pl.col("week").min().alias("bye_week"), pl.len().alias("missing_weeks"))
    )
    # Every team gets a row. A team that plays every week, or that is missing more than
    # one week, gets a null bye rather than a fabricated one.
    return (
        teams.join(missing, on="team", how="left")
        .with_columns(
            pl.when(pl.col("missing_weeks") == 1)
            .then(pl.col("bye_week"))
            .otherwise(None)
            .alias("bye_week"),
            pl.lit(season, dtype=pl.Int64).alias("season"),
        )
        .select("season", "team", pl.col("bye_week").cast(pl.Int64))
        .sort("team")
    )
