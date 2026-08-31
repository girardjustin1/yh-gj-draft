"""Fantasy scoring of raw box-score stats, in *our* league's rules.

Borrowed "PPR points" from a vendor are not our points: pass-TD value, reception value,
and bonuses all differ league to league. Every fantasy-point number in this project is
recomputed from component stats through :func:`fantasy_points_expr`, so changing
``config/league.yaml`` changes every downstream score.
"""

from __future__ import annotations

import polars as pl

from ..config import ScoringRules


def _col(frame_columns: set[str], name: str) -> pl.Expr:
    """Column if present, else zero — historical tables vary by season."""
    return pl.col(name).fill_null(0.0) if name in frame_columns else pl.lit(0.0)


def fantasy_points_expr(scoring: ScoringRules, columns: set[str]) -> pl.Expr:
    """Polars expression computing fantasy points for one row of box-score stats.

    ``columns`` is the set of available column names, so the expression degrades
    gracefully on partial data rather than raising.
    """
    passing_yards = _col(columns, "passing_yards")
    rushing_yards = _col(columns, "rushing_yards")
    receiving_yards = _col(columns, "receiving_yards")

    total = (
        passing_yards / scoring.passing_yards_per_point
        + _col(columns, "passing_tds") * scoring.passing_td
        + _col(columns, "interceptions") * scoring.passing_interception
        + rushing_yards / scoring.rushing_yards_per_point
        + _col(columns, "rushing_tds") * scoring.rushing_td
        + receiving_yards / scoring.receiving_yards_per_point
        + _col(columns, "receiving_tds") * scoring.receiving_td
        + _col(columns, "receptions") * scoring.reception
        + _col(columns, "fumbles_lost") * scoring.fumble_lost
        + _col(columns, "two_point_conv") * scoring.rushing_2pt
    )

    bonuses = [
        (passing_yards >= 300, scoring.bonus_pass_300_yards),
        (passing_yards >= 400, scoring.bonus_pass_400_yards),
        (rushing_yards >= 100, scoring.bonus_rush_100_yards),
        (rushing_yards >= 200, scoring.bonus_rush_200_yards),
        (receiving_yards >= 100, scoring.bonus_rec_100_yards),
        (receiving_yards >= 200, scoring.bonus_rec_200_yards),
    ]
    for condition, points in bonuses:
        if points:
            total = total + pl.when(condition).then(points).otherwise(0.0)

    return total.alias("fantasy_points_league")


def score_weekly_stats(frame: pl.DataFrame, scoring: ScoringRules) -> pl.DataFrame:
    """Add a ``fantasy_points_league`` column to a weekly box-score frame."""
    if frame.is_empty():
        return frame.with_columns(pl.lit(0.0).alias("fantasy_points_league"))
    return frame.with_columns(fantasy_points_expr(scoring, set(frame.columns)))
