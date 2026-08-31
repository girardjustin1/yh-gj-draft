"""Market signals: what the room thinks, and where we disagree with it.

Two distinct questions live here, and conflating them is a common way to draft badly:

**"Is he falling?"** — his consensus draft position versus the pick actually on the
clock. Pure market value, and only meaningful during a live draft.

Two corrections make this behave. First, the discount is **capped at the horizon we can
act over**: a player 90 picks past his ADP is not a bargain now, because we could simply
take him later and spend this pick on someone who will be gone. Second, and more
importantly, the discount is **weighted by the probability we would actually lose him**.

That second point is the one that makes the difference between a rankings viewer and a
decision engine: *a discount you can still capture at your next pick is not a discount*.
Without it, market value and next-pick urgency double-count the same fact with opposite
signs — a player available well past his ADP scores high on value and low on urgency,
and the two very nearly cancel, leaving the engine recommending players it simultaneously
reports an 81% chance of still being there. Scaling the discount by the probability he
disappears resolves it: value you would have got anyway is worth nothing now.

**"Is the market wrong about him?"** — his consensus rank versus *our* VBD ordering.
Meaningful at any time, and the thing that makes this engine more than a rankings viewer.

A falling player is not automatically a good pick. Falls carry information: the room may
know about a hamstring we do not. So a large unexplained fall raises the risk component
as well as the value component, and the two are reported separately rather than netted.

**Honesty about the source.** We use FantasyPros expert consensus rank as the ADP proxy,
because it is what nflverse publishes for 2026 with a best/worst spread attached. ECR is
*not* ADP — experts and drafters differ, systematically so at quarterback and tight end.
When a real ADP feed is imported it takes precedence, and ``adp_is_proxy`` records which
was used so the explanation can say so out loud.
"""

from __future__ import annotations

import polars as pl

from ..config import AppConfig
from ..database import Database


def adp_table(db: Database, cfg: AppConfig) -> pl.DataFrame:
    """Best available ADP per player: an imported feed if present, else the ECR proxy."""
    imported = db.query(
        """
        SELECT player_key, avg(adp) AS adp, avg(adp_sd) AS adp_sd,
               min(adp_min) AS adp_min, max(adp_max) AS adp_max,
               count(*) AS adp_sources, max(snapshot_at) AS adp_updated_at
        FROM adp_snapshots WHERE season = ? AND player_key IS NOT NULL
        GROUP BY player_key
        """,
        [cfg.league.season],
    )
    proxy = db.query(
        """
        SELECT player_key, ecr AS adp, sd AS adp_sd,
               CAST(best AS DOUBLE) AS adp_min, CAST(worst AS DOUBLE) AS adp_max,
               source_updated_at AS adp_updated_at
        FROM rankings
        WHERE season = ? AND ranking_type = 'redraft-overall' AND player_key IS NOT NULL
        """,
        [cfg.league.season],
    ).with_columns(pl.lit(1, dtype=pl.Int64).alias("adp_sources"))

    if imported.is_empty():
        return proxy.with_columns(pl.lit(True).alias("adp_is_proxy"))

    imported = imported.with_columns(pl.lit(False).alias("adp_is_proxy"))
    missing = proxy.join(imported.select("player_key"), on="player_key", how="anti")
    return pl.concat(
        [imported, missing.with_columns(pl.lit(True).alias("adp_is_proxy"))],
        how="diagonal_relaxed",
    )


def market_signals(
    board: pl.DataFrame,
    adp: pl.DataFrame,
    current_pick: int | None = None,
    picks_until_next: int | None = None,
    next_pick: int | None = None,
) -> pl.DataFrame:
    """Attach ADP, market value, and projection-versus-market disagreement.

    ``current_pick`` is the overall pick on the clock. Without it, market value is
    reported against the player's own VBD ordering instead, which is the right question
    for a static board. ``next_pick`` enables the survival weighting described above; it
    uses only the ADP distribution, so there is no circular dependency on the full
    roster-aware survival model.
    """
    if board.is_empty():
        return board

    frame = board.join(adp, on="player_key", how="left")

    # Our own ordering. VBD is the cross-positional currency, so this is what we think
    # the draft order should be.
    frame = frame.with_columns(
        pl.col("vbd").rank("ordinal", descending=True).cast(pl.Float64).alias("vbd_rank")
    )

    # Positive = the market is lower on him than we are, i.e. he is available later than
    # our board says he should be.
    frame = frame.with_columns(
        (pl.col("adp") - pl.col("vbd_rank")).alias("market_disagreement")
    )

    if current_pick is not None:
        # Positive = he has fallen past his ADP and is available at a discount.
        frame = frame.with_columns(
            (pl.col("adp") - float(current_pick)).alias("adp_delta")
        )
    else:
        frame = frame.with_columns(pl.col("market_disagreement").alias("adp_delta"))

    # A discount is only worth something if we could not have captured it by waiting.
    # A player whose ADP is 90 picks away is not a bargain at this pick -- we could take
    # him three rounds from now and spend this pick on someone who will be gone. So the
    # discount that counts is capped at the horizon we can actually act over: the picks
    # until our next selection, plus a round of slack. Without that cap, every deep
    # player scores a perfect 100 on market value and floats to the top of the board.
    horizon = float((picks_until_next or 12) + 12)
    frame = frame.with_columns(
        pl.min_horizontal(pl.col("adp_delta"), pl.lit(horizon)).alias("capturable_delta")
    )

    # Weight the discount by the chance we would actually lose him. A reach (negative
    # delta) is a reach regardless of who else wants him, so only the positive side is
    # scaled.
    if next_pick is not None:
        from ..draft.availability import adp_survival

        gone = [
            1.0 - adp_survival(a, sd, next_pick)
            for a, sd in zip(
                frame["adp"].to_list(),
                frame["adp_sd"].to_list() if "adp_sd" in frame.columns
                else [None] * frame.height,
                strict=True,
            )
        ]
        frame = frame.with_columns(pl.Series("adp_probability_gone", gone, dtype=pl.Float64))
        frame = frame.with_columns(
            pl.when(pl.col("capturable_delta") > 0)
            .then(pl.col("capturable_delta") * pl.col("adp_probability_gone"))
            .otherwise(pl.col("capturable_delta"))
            .alias("capturable_delta")
        )
    else:
        frame = frame.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("adp_probability_gone")
        )

    # Normalize the discount against the ADP uncertainty we actually observe, so a
    # 10-pick fall for a tightly ranked player counts for more than for a volatile one.
    spread = pl.col("adp_sd").fill_null(
        (pl.col("adp_max") - pl.col("adp_min")) / 4.0
    ).fill_null(12.0).clip(2.0, 60.0)

    frame = frame.with_columns(
        (50.0 + 50.0 * (pl.col("capturable_delta") / (2.0 * spread)).clip(-1, 1))
        .fill_null(50.0)
        .alias("market_value_score"),
        (
            50.0 + 50.0 * (pl.col("market_disagreement") / 60.0).clip(-1, 1)
        ).fill_null(50.0).alias("projection_vs_market_score"),
        (
            pl.when(pl.col("adp").is_null()).then(0.0)
            .when(pl.col("adp_is_proxy")).then(0.6)
            .otherwise(1.0)
        ).alias("market_confidence"),
    )
    return frame
