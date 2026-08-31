"""Risk: how likely is this projection to be wrong, and in which direction?

Risk is an *uncertainty* measure, not a prediction of failure. It enters the Player
Score inverted (low risk raises the score) at a deliberately small weight, because being
scared of every uncertain player is its own way of drafting badly.

Signals used, all derived from data we actually hold:

* **Expert disagreement** — the ECR best/worst spread relative to the player's own rank.
  A player ranked 30th whom someone ranks 12th and someone else 60th is a genuine coin
  flip.
* **Injury status** — the most recent injury report. Its *age* matters: a report from
  last December says little in August, and is treated as such.
* **Sample size** — a rookie or a player with 8 career games is unknown, not bad.
* **Age** — only where it is real. Running-back production falls off sharply after about
  27; receivers and quarterbacks age far more gently.
* **Role competition** — not being the clear first-teamer on the latest depth chart.

**We never invent injury information.** If we have no report, that is recorded as
*unknown* (lower confidence) rather than as *healthy*.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from ..config import AppConfig
from ..database import Database

#: Age past which production risk rises, by position. RBs age hardest.
AGE_CLIFF: dict[str, float] = {"RB": 27.0, "WR": 30.0, "TE": 31.0, "QB": 36.0}

#: Injury report statuses mapped to a 0-1 severity.
INJURY_SEVERITY: dict[str, float] = {
    "Out": 1.0, "Doubtful": 0.8, "Questionable": 0.45, "Limited": 0.3, "Full": 0.05,
}


def risk_score(db: Database, cfg: AppConfig, board: pl.DataFrame) -> pl.DataFrame:
    """Score risk 0-100 (higher = riskier) for every player on ``board``.

    ``board`` needs ``player_key``, ``position``, and — where available — ``ecr_best``,
    ``ecr_worst``, ``overall_ecr``, and ``projection_disagreement``.
    """
    if board.is_empty():
        return board

    bio = db.query(
        "SELECT player_key, birth_date, rookie_season, years_experience FROM players"
    )
    injuries = db.query(
        """
        SELECT player_key, report_status, report_primary, season, week, ingested_at
        FROM (
            SELECT *, row_number() OVER (
                PARTITION BY player_key ORDER BY season DESC, week DESC
            ) AS rn
            FROM injuries WHERE player_key IS NOT NULL AND report_status IS NOT NULL
        ) WHERE rn = 1
        """
    )
    depth = db.query(
        """
        SELECT player_key, pos_rank FROM (
            SELECT *, row_number() OVER (
                PARTITION BY player_key ORDER BY as_of DESC, pos_rank
            ) AS rn
            FROM depth_charts WHERE player_key IS NOT NULL
        ) WHERE rn = 1
        """
    )

    frame = (
        board.join(bio, on="player_key", how="left")
        .join(injuries, on="player_key", how="left")
        .join(depth, on="player_key", how="left")
    )

    today = datetime.now()
    season = cfg.league.season

    # 1. Expert disagreement, as a fraction of the player's own rank.
    disagreement = (
        (pl.col("ecr_worst") - pl.col("ecr_best")).cast(pl.Float64)
        / (pl.col("overall_ecr").fill_null(250.0) + 15.0)
    ).clip(0, 3) / 3.0

    # 2. Injury. Weighted down by how stale the report is: a week-18 note from two
    #    seasons ago is nearly meaningless by draft day.
    seasons_old = (pl.lit(season) - pl.col("season").fill_null(season - 3)).cast(pl.Float64)
    injury_recency = (0.5 ** seasons_old.clip(0, 6)).fill_null(0.0)
    injury = (
        pl.col("report_status").replace_strict(INJURY_SEVERITY, default=0.25).fill_null(0.0)
        * injury_recency
    )

    # 3. Sample size.
    experience = pl.col("years_experience").fill_null(0).cast(pl.Float64)
    sample = (1.0 - (experience / 3.0).clip(0, 1)).fill_null(0.6)

    # 4. Age, only past the position's cliff.
    age_years = (
        (pl.lit(today) - pl.col("birth_date").cast(pl.Datetime)).dt.total_days() / 365.25
    )
    cliff = pl.col("position").replace_strict(AGE_CLIFF, default=32.0).cast(pl.Float64)
    age = ((age_years - cliff) / 4.0).clip(0, 1).fill_null(0.0)

    # 5. Role competition on the latest depth chart.
    competition = (
        pl.when(pl.col("pos_rank").is_null()).then(0.35)
        .when(pl.col("pos_rank") <= 1).then(0.0)
        .when(pl.col("pos_rank") == 2).then(0.5)
        .otherwise(0.8)
    )

    # 6. Projection sources disagreeing with each other.
    projection_spread = (
        pl.col("projection_disagreement").fill_null(0.0)
        / (pl.col("projected_points").abs() + 25.0)
    ).clip(0, 1) if "projection_disagreement" in frame.columns else pl.lit(0.0)

    frame = frame.with_columns(
        disagreement.alias("risk_disagreement"),
        injury.alias("risk_injury"),
        sample.alias("risk_sample"),
        age.alias("risk_age"),
        competition.alias("risk_competition"),
        projection_spread.alias("risk_projection_spread"),
        age_years.alias("age"),
    )

    frame = frame.with_columns(
        (
            100.0
            * (
                pl.col("risk_disagreement") * 0.30
                + pl.col("risk_injury") * 0.20
                + pl.col("risk_sample") * 0.20
                + pl.col("risk_age") * 0.15
                + pl.col("risk_competition") * 0.10
                + pl.col("risk_projection_spread") * 0.05
            )
        ).clip(0, 100).alias("risk_score")
    )

    # Confidence: we know more about a player with an ECR spread, a depth-chart slot and
    # a birth date than about one with none of them.
    return frame.with_columns(
        (
            0.4
            + pl.when(pl.col("ecr_best").is_not_null()).then(0.25).otherwise(0.0)
            + pl.when(pl.col("pos_rank").is_not_null()).then(0.2).otherwise(0.0)
            + pl.when(pl.col("birth_date").is_not_null()).then(0.15).otherwise(0.0)
        ).clip(0, 1).alias("risk_confidence")
    ).select(
        "player_key", "risk_score", "risk_confidence", "age", "risk_disagreement",
        "risk_injury", "risk_sample", "risk_age", "risk_competition",
        pl.col("report_status").alias("injury_status"),
        pl.col("report_primary").alias("injury_detail"),
        pl.col("season").alias("injury_season"),
        pl.col("pos_rank").alias("depth_rank"),
    )
