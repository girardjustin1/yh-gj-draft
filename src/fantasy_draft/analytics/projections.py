"""Projected fantasy points for the upcoming season.

We deliberately do **not** build a machine-learned projection system. What we build is a
transparent bridge from the market's *ordering* to fantasy *points*, because VBD,
scarcity, and tier cliffs all need points, not ranks — the whole question "how much do I
lose by waiting?" is meaningless in rank space.

**Method: historical positional value curve.**

1. Score every past season's weekly box scores in *our* league's rules.
2. For each position, take the points scored by the player who finished 1st, 2nd, 3rd...
   Average those across recent seasons, weighting recent seasons more heavily, and
   smooth across neighbouring ranks.
3. Read each current player's positional rank off the consensus board, and look up what
   that finishing position has historically been worth.

So "the RB the market ranks 8th" is projected to score what RB8 has typically scored.

**What this assumes, and where it breaks.**

* The consensus ordering is approximately right. Where it is wrong, the projection is
  wrong in the same direction — which is why Opportunity, Risk, and market-disagreement
  enter the Player Score as *separate* components rather than being folded in here.
* The shape of the curve is stable year to year. It is far more stable than any
  individual player's output, which is exactly why this is the safer thing to model.
* It projects the *rank*, not the player: it cannot know that RB8 this year is a rookie
  on a bad offence. That is what the other components are for.

Imported projections (``ff import projections``) always take precedence, and are kept as
separate sources rather than overwriting each other.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl

from ..config import AppConfig
from ..constants import DRAFTABLE_POSITIONS, SOURCE_MANUAL
from ..database import Database
from ..logging import get_logger
from ..normalization.ids import IdentityMap, resolve_column
from ..normalization.players import normalize_position
from ..normalization.teams import canonical_team_expr
from .fantasy_points import score_weekly_stats

log = get_logger(__name__)

#: Source label for curve-derived projections.
SOURCE_DERIVED = "derived_ecr_curve"

#: Ranks beyond this per position are treated as replacement-level noise.
CURVE_DEPTH: dict[str, int] = {"QB": 48, "RB": 90, "WR": 110, "TE": 48, "K": 36, "DST": 36}


def season_totals(db: Database, cfg: AppConfig, seasons: list[int] | None = None) -> pl.DataFrame:
    """Per player-season regular-season fantasy points in our league's scoring."""
    where = "season_type = 'REG'"
    params: list[object] = []
    if seasons:
        placeholders = ", ".join("?" for _ in seasons)
        where += f" AND season IN ({placeholders})"
        params = list(seasons)

    weekly = db.query(f"SELECT * FROM historical_player_stats WHERE {where}", params)
    if weekly.is_empty():
        return pl.DataFrame(
            schema={
                "player_key": pl.Utf8, "season": pl.Int32, "position": pl.Utf8,
                "games": pl.UInt32, "fantasy_points": pl.Float64,
            }
        )
    scored = score_weekly_stats(weekly, cfg.league.scoring)
    return (
        scored.filter(pl.col("player_key").is_not_null())
        .group_by(["player_key", "season", "position"])
        .agg(
            pl.len().alias("games"),
            pl.col("fantasy_points_league").sum().alias("fantasy_points"),
        )
        .filter(pl.col("position").is_in(list(DRAFTABLE_POSITIONS)))
    )


def positional_value_curve(
    db: Database, cfg: AppConfig, smoothing: int = 2
) -> pl.DataFrame:
    """What each positional finish has historically been worth, in our scoring.

    Returns ``position, rank, points, seasons_used, points_sd``. Recent seasons are
    weighted more heavily via ``season_recency_halflife``; ``points_sd`` across seasons
    is carried through so the projection can report an honest confidence.
    """
    totals = season_totals(db, cfg)
    if totals.is_empty():
        return pl.DataFrame(
            schema={
                "position": pl.Utf8, "rank": pl.Int64, "points": pl.Float64,
                "points_sd": pl.Float64, "seasons_used": pl.UInt32,
            }
        )

    latest = int(totals["season"].max())
    halflife = cfg.weights.season_recency_halflife
    ranked = (
        totals.sort(["position", "season", "fantasy_points"], descending=[False, False, True])
        .with_columns(
            pl.col("fantasy_points")
            .rank("ordinal", descending=True)
            .over(["position", "season"])
            .cast(pl.Int64)
            .alias("rank"),
            (0.5 ** ((latest - pl.col("season")) / halflife)).alias("weight"),
        )
    )

    depth = pl.DataFrame(
        {"position": list(CURVE_DEPTH), "max_rank": [CURVE_DEPTH[p] for p in CURVE_DEPTH]}
    )
    ranked = ranked.join(depth, on="position", how="left").filter(
        pl.col("rank") <= pl.col("max_rank").fill_null(60)
    )

    curve = (
        ranked.group_by(["position", "rank"])
        .agg(
            (
                (pl.col("fantasy_points") * pl.col("weight")).sum() / pl.col("weight").sum()
            ).alias("points"),
            pl.col("fantasy_points").std().alias("points_sd"),
            pl.len().alias("seasons_used"),
        )
        .sort(["position", "rank"])
    )

    # Smooth across neighbouring ranks: the difference between the 31st and 32nd RB in
    # any single season is noise, and an unsmoothed curve puts that noise straight into
    # every VBD number.
    window = 2 * smoothing + 1
    curve = curve.with_columns(
        pl.col("points")
        .rolling_mean(window_size=window, min_samples=1, center=True)
        .over("position")
        .alias("points")
    )
    # The curve must be monotonically non-increasing: rank 20 cannot be worth more than
    # rank 19. Smoothing can violate this at the shoulders.
    return curve.with_columns(
        pl.col("points").cum_min().over("position").alias("points")
    )


def consensus_board(db: Database, cfg: AppConfig) -> pl.DataFrame:
    """The market's ordering, with a positional rank per player.

    Built from the overall redraft board, falling back to per-position boards for
    players deep enough that the overall page does not list them.
    """
    rankings = db.query(
        """
        SELECT player_key, player_name, position, team, ranking_type, ecr, sd, best,
               worst, bye_week, source, source_updated_at
        FROM rankings WHERE season = ?
        """,
        [cfg.league.season],
    )
    if rankings.is_empty():
        return pl.DataFrame(
            schema={
                "player_key": pl.Utf8, "player_name": pl.Utf8, "position": pl.Utf8,
                "team": pl.Utf8, "overall_ecr": pl.Float64, "ecr_sd": pl.Float64,
                "positional_rank": pl.Int64,
            }
        )

    overall = rankings.filter(pl.col("ranking_type") == "redraft-overall").select(
        "player_key", "player_name", "position", "team", "bye_week",
        pl.col("ecr").alias("overall_ecr"), pl.col("sd").alias("ecr_sd"),
        pl.col("best").alias("ecr_best"), pl.col("worst").alias("ecr_worst"),
        "source_updated_at",
    )
    positional = (
        rankings.filter(
            pl.col("ranking_type").str.starts_with("redraft-")
            & (pl.col("ranking_type") != "redraft-overall")
            & (pl.col("ranking_type") != "redraft-op")
            & (pl.col("ranking_type") != "redraft-idp")
        )
        .select(
            "player_key", "player_name", "position", "team", "bye_week",
            pl.col("ecr").alias("pos_ecr"), pl.col("sd").alias("pos_sd"),
            pl.col("best").alias("pos_best"), pl.col("worst").alias("pos_worst"),
            "source_updated_at",
        )
        .unique(subset=["player_key"], keep="first")
    )

    board = overall.join(
        positional.select("player_key", "pos_ecr", "pos_sd", "pos_best", "pos_worst"),
        on="player_key", how="full", coalesce=True,
    )
    # Players missing from the overall page (deep bench) still get an identity and a
    # position from the positional page.
    board = board.join(
        positional.select("player_key", "player_name", "position", "team", "bye_week",
                          "source_updated_at").rename(
            {
                "player_name": "pos_name", "position": "pos_position", "team": "pos_team",
                "bye_week": "pos_bye", "source_updated_at": "pos_updated",
            }
        ),
        on="player_key", how="left",
    ).with_columns(
        pl.coalesce("player_name", "pos_name").alias("player_name"),
        pl.coalesce("position", "pos_position").alias("position"),
        pl.coalesce("team", "pos_team").alias("team"),
        pl.coalesce("bye_week", "pos_bye").alias("bye_week"),
        pl.coalesce("ecr_sd", "pos_sd").alias("ecr_sd"),
        pl.coalesce("ecr_best", "pos_best").alias("ecr_best"),
        pl.coalesce("ecr_worst", "pos_worst").alias("ecr_worst"),
        pl.coalesce("source_updated_at", "pos_updated").alias("source_updated_at"),
        pl.col("overall_ecr").is_not_null().alias("on_overall_board"),
    ).filter(pl.col("position").is_in(list(DRAFTABLE_POSITIONS)))

    # Rank within position: the overall board first (it is the sharper signal), then the
    # positional-only tail, ordered by their positional ECR.
    board = board.sort(
        ["position", "on_overall_board", "overall_ecr", "pos_ecr"],
        descending=[False, True, False, False],
        nulls_last=True,
    ).with_columns(
        pl.int_range(1, pl.len() + 1).over("position").cast(pl.Int64).alias("positional_rank")
    )
    return board.select(
        "player_key", "player_name", "position", "team", "bye_week", "overall_ecr",
        "pos_ecr", "ecr_sd", "ecr_best", "ecr_worst", "on_overall_board",
        "positional_rank", "source_updated_at",
    )


def derive_projections(db: Database, cfg: AppConfig) -> pl.DataFrame:
    """Project fantasy points by mapping each player's positional rank onto the curve.

    Confidence reflects how much we should trust the number: it falls when the experts
    disagree (wide ECR spread), when the player sits below the overall board, and when
    the historical curve at that rank was itself volatile across seasons.
    """
    board = consensus_board(db, cfg)
    curve = positional_value_curve(db, cfg)
    if board.is_empty() or curve.is_empty():
        log.warning("cannot derive projections", extra={
            "board_rows": board.height, "curve_rows": curve.height
        })
        return pl.DataFrame()

    # Beyond the curve's depth, hold the last (lowest) value: deep bench players are all
    # worth roughly replacement level, and pretending otherwise invents precision.
    tail = curve.group_by("position").agg(
        pl.col("points").min().alias("tail_points"),
        pl.col("rank").max().alias("curve_depth"),
    )

    projected = (
        board.join(curve, left_on=["position", "positional_rank"], right_on=["position", "rank"],
                   how="left")
        .join(tail, on="position", how="left")
        .with_columns(
            pl.coalesce("points", "tail_points").alias("fantasy_points"),
        )
    )

    # Confidence: start high, deduct for each source of uncertainty.
    spread = (pl.col("ecr_worst") - pl.col("ecr_best")).cast(pl.Float64)
    projected = projected.with_columns(
        (
            pl.lit(1.0)
            # Expert disagreement, as a fraction of the player's own rank.
            - (spread / (pl.col("overall_ecr").fill_null(300.0) + 20.0)).fill_null(0.4).clip(0, 0.45)
            # Not on the overall board: we inferred the rank rather than reading it.
            - pl.when(pl.col("on_overall_board")).then(0.0).otherwise(0.25)
            # The curve itself was noisy at this rank.
            - (pl.col("points_sd") / (pl.col("fantasy_points") + 25.0)).fill_null(0.15).clip(0, 0.2)
        ).clip(0.05, 1.0).alias("confidence")
    )

    now = datetime.now()
    return projected.select(
        "player_key",
        pl.lit(SOURCE_DERIVED, dtype=pl.Utf8).alias("source"),
        pl.lit(None, dtype=pl.Utf8).alias("source_player_id"),
        "player_name", "position", "team",
        pl.lit(cfg.league.season, dtype=pl.Int32).alias("season"),
        pl.lit(None, dtype=pl.Int32).alias("week"),
        pl.lit(None, dtype=pl.Float64).alias("games"),
        *[pl.lit(None, dtype=pl.Float64).alias(c) for c in
          ("pass_yards", "pass_tds", "interceptions", "rush_yards", "rush_tds",
           "receptions", "rec_yards", "rec_tds", "fumbles_lost")],
        pl.col("fantasy_points").cast(pl.Float64),
        pl.col("source_updated_at").cast(pl.Datetime),
        pl.lit(now, dtype=pl.Datetime).alias("ingested_at"),
    ).filter(pl.col("fantasy_points").is_not_null())


# --- imported projections --------------------------------------------------------------

#: Column aliases accepted by the import adapter, mapping to our schema.
IMPORT_ALIASES: dict[str, tuple[str, ...]] = {
    "player_name": ("player_name", "player", "name", "full_name", "playername"),
    "position": ("position", "pos"),
    "team": ("team", "tm", "nfl_team"),
    "fantasy_points": ("fantasy_points", "points", "fpts", "proj_points", "projection"),
    "games": ("games", "g", "gp"),
    "pass_yards": ("pass_yards", "passing_yards", "pass_yds", "py"),
    "pass_tds": ("pass_tds", "passing_tds", "pass_td"),
    "interceptions": ("interceptions", "int", "ints", "pass_int"),
    "rush_yards": ("rush_yards", "rushing_yards", "rush_yds"),
    "rush_tds": ("rush_tds", "rushing_tds", "rush_td"),
    "receptions": ("receptions", "rec", "catches"),
    "rec_yards": ("rec_yards", "receiving_yards", "rec_yds"),
    "rec_tds": ("rec_tds", "receiving_tds", "rec_td"),
    "fumbles_lost": ("fumbles_lost", "fl", "fum_lost"),
    "source_player_id": ("player_id", "id", "source_player_id"),
}


class ProjectionImportError(RuntimeError):
    """Raised with a human-readable message when an import file cannot be used."""


def _read_any(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path, infer_schema_length=2000, ignore_errors=True)
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix == ".json":
        text = path.read_text()
        data = json.loads(text)
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    data = value
                    break
        if not isinstance(data, list):
            raise ProjectionImportError(
                f"{path.name}: expected a JSON array of objects, or an object containing one."
            )
        return pl.DataFrame(data)
    raise ProjectionImportError(
        f"{path.name}: unsupported format {suffix!r}. Use .csv, .parquet, or .json."
    )


def import_projections(
    db: Database, cfg: AppConfig, path: Path, source: str | None = None
) -> tuple[pl.DataFrame, int]:
    """Load a projection file into the ``projections`` table.

    Accepts either a ``fantasy_points`` column, or component stats which are then scored
    in *our* league's rules — a vendor's PPR total is not our total. Returns the frame
    written and the count of unresolved players.
    """
    if not path.is_file():
        raise ProjectionImportError(f"{path} does not exist.")
    raw = _read_any(path)
    if raw.is_empty():
        raise ProjectionImportError(f"{path.name} is empty.")

    lower = {c.lower().strip().replace(" ", "_"): c for c in raw.columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in IMPORT_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                resolved[canonical] = lower[alias]
                break

    if "player_name" not in resolved:
        raise ProjectionImportError(
            f"{path.name}: no player-name column found. "
            f"Expected one of: {', '.join(IMPORT_ALIASES['player_name'])}.\n"
            f"Found: {', '.join(raw.columns[:12])}"
        )

    frame = raw.select(
        [
            pl.col(source_col).alias(canonical)
            for canonical, source_col in resolved.items()
        ]
    )
    numeric = [c for c in frame.columns if c not in {"player_name", "position", "team",
                                                     "source_player_id"}]
    frame = frame.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric])
    for canonical in ("position", "team", "source_player_id"):
        if canonical not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias(canonical))
    frame = frame.with_columns(
        pl.col("position").map_elements(normalize_position, return_dtype=pl.Utf8),
        canonical_team_expr("team"),
    )

    if "fantasy_points" not in frame.columns:
        component_map = {
            "passing_yards": "pass_yards", "passing_tds": "pass_tds",
            "interceptions": "interceptions", "rushing_yards": "rush_yards",
            "rushing_tds": "rush_tds", "receptions": "receptions",
            "receiving_yards": "rec_yards", "receiving_tds": "rec_tds",
            "fumbles_lost": "fumbles_lost",
        }
        available = {ours for stat, ours in component_map.items() if ours in frame.columns}
        if not available:
            raise ProjectionImportError(
                f"{path.name}: no 'fantasy_points' column and no component stats to score "
                f"one from. Provide points, or yardage/TD/reception columns."
            )
        scoring_frame = frame.rename({v: k for k, v in component_map.items() if v in frame.columns})
        from .fantasy_points import fantasy_points_expr

        frame = frame.with_columns(
            fantasy_points_expr(cfg.league.scoring, set(scoring_frame.columns))
            .alias("fantasy_points")
        )
        log.info("scored imported projections from components", extra={"file": path.name})

    identity = IdentityMap(
        db.query(
            """
            SELECT p.player_key, p.normalized_name, p.position,
                   i.gsis_id, i.sleeper_id, i.espn_id, i.yahoo_id, i.fantasypros_id, i.pfr_id
            FROM players p LEFT JOIN player_ids i USING (player_key)
            """
        )
    )
    frame = resolve_column(
        frame, identity, source=source or path.stem,
        name_column="player_name", position_column="position", team_column="team",
    )

    label = source or f"{SOURCE_MANUAL}:{path.stem}"
    now = datetime.now()
    out = frame.with_columns(
        pl.lit(label, dtype=pl.Utf8).alias("source"),
        pl.lit(cfg.league.season, dtype=pl.Int32).alias("season"),
        pl.lit(None, dtype=pl.Int32).alias("week"),
        pl.lit(datetime.fromtimestamp(path.stat().st_mtime), dtype=pl.Datetime)
        .alias("source_updated_at"),
        pl.lit(now, dtype=pl.Datetime).alias("ingested_at"),
    )
    db.upsert_table("projections", out, keys=["source", "season"])
    if identity.unresolved:
        db.upsert_table(
            "unresolved_players", identity.unresolved_frame(), keys=["source", "raw_name"]
        )
    return out, len(identity.unresolved)


def refresh_derived_projections(db: Database, cfg: AppConfig) -> int:
    """Recompute and store the curve-derived projections."""
    frame = derive_projections(db, cfg)
    if frame.is_empty():
        log.warning("no derived projections produced")
        return 0
    db.upsert_table("projections", frame, keys=["source", "season"])
    return frame.height


def consensus_projections(db: Database, cfg: AppConfig) -> pl.DataFrame:
    """Blend every stored projection source into one number per player.

    Imported sources are weighted by ``config/data_sources.yaml``. The derived curve is
    always present as a floor, so a partial import never silently drops players off the
    board.
    """
    frame = db.query(
        "SELECT player_key, source, player_name, position, team, fantasy_points, "
        "source_updated_at FROM projections WHERE season = ? AND week IS NULL "
        "AND fantasy_points IS NOT NULL",
        [cfg.league.season],
    )
    if frame.is_empty():
        return frame

    weights = {
        source: cfg.data_sources.spec(source).weight if source != SOURCE_DERIVED else 1.0
        for source in frame["source"].unique().to_list()
    }
    frame = frame.with_columns(
        pl.col("source").replace_strict(weights, default=1.0).alias("weight")
    )
    return (
        frame.group_by("player_key")
        .agg(
            pl.col("player_name").first(),
            pl.col("position").first(),
            pl.col("team").first(),
            (
                (pl.col("fantasy_points") * pl.col("weight")).sum() / pl.col("weight").sum()
            ).alias("projected_points"),
            pl.col("fantasy_points").std().alias("projection_disagreement"),
            pl.len().alias("projection_sources"),
            pl.col("source_updated_at").max().alias("source_updated_at"),
        )
        .sort("projected_points", descending=True)
    )
