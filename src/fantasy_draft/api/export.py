"""Static export: bake the Python-scored board into JSON a plain web host can serve.

GitHub Pages runs no Python. That is a hard constraint, and pretending otherwise would
ship a page that loads and then fails on every request.

So the split is explicit. **Everything that does not depend on live draft state is
computed here, in Python, at build time** — projections, VBD, tiers, replacement levels,
floor/median/ceiling, opportunity, risk, schedule, ADP and its spread. The exported file
is the engine's output, not a reimplementation of it.

What the static page then does client-side is limited on purpose to things that are
either pure arithmetic or already published by the engine as a documented baseline:

* snake pick maths — deterministic, and the same formulae as ``draft/snake.py``
* the ADP-only survival curve — exactly ``availability.adp_survival``, which the live
  engine already computes and reports beside its roster-aware figure as
  ``probability_available_adp_only``
* marginal lineup value — a pure function of your roster and the baked VBD

What it deliberately does **not** attempt is the roster-aware survival model, the Monte
Carlo two-pick expected value, the draft-room read, or the adaptive strategy state. Those
need the live engine, and a second approximate implementation of them in JavaScript would
drift from the real one and quietly disagree with it mid-draft. The page says so rather
than guessing.

No league-private data is written: draft state lives only in the browser.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from ..analytics.board import build_board
from ..config import AppConfig
from ..constants import FLEX_ELIGIBILITY
from ..database import Database
from ..logging import get_logger
from ..service import _board_row, lineup_slots

log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

#: Columns carried into the export. Everything the offline page can legitimately show.
EXPORT_COLUMNS: tuple[str, ...] = (
    "player_key", "player_name", "position", "team", "bye_week",
    "projected_points", "vbd", "positional_rank", "tier", "tier_rank",
    "adp", "adp_sd", "player_score", "value_score", "draft_now_score",
    "floor_points", "median_points", "ceiling_points",
    "opportunity_score", "risk_score", "offense_score", "schedule_score",
    "scarcity_score", "tier_cliff_score", "market_value_score",
    "player_score_confidence",
)


def export_board(db: Database, cfg: AppConfig, limit: int = 400) -> dict[str, Any]:
    """Build the payload the offline page consumes."""
    board = build_board(db, cfg)
    if board.frame.is_empty():
        raise RuntimeError(
            "No board to export. Run `ff data refresh` first."
        )

    frame = board.frame
    available = [c for c in EXPORT_COLUMNS if c in frame.columns]
    rows = frame.head(limit).select(available).to_dicts()

    players = []
    for index, row in enumerate(rows, start=1):
        record = _board_row(row, index)
        # Static scores only: anything draft-dependent is recomputed in the browser.
        for key in ("probability_gone", "probability_available",
                    "two_pick_expected_value", "lineup_upgrade"):
            record.pop(key, None)
        record["adp_sd"] = (
            round(float(row["adp_sd"]), 2) if row.get("adp_sd") is not None else None
        )
        record["positional_rank"] = row.get("positional_rank")
        players.append(record)

    league = cfg.league
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "season": league.season,
        "engine_version": "0.2.0",
        "league": {
            "name": league.name,
            "label": league.label,
            "teams": league.teams,
            "rounds": league.draft.rounds,
            "draft_type": league.draft.type,
            "third_round_reversal": league.draft.third_round_reversal,
            "scoring_type": league.scoring_type,
            "is_superflex": league.roster.is_superflex,
            "lineup_slots": lineup_slots(league.roster),
            "flex_eligibility": {
                name: list(FLEX_ELIGIBILITY[name]) for name in league.roster.flex_counts
            },
            "starters": league.roster.starters,
            "bench": league.roster.bench,
        },
        "replacement": {
            position: {
                "rank": level.rank,
                "points": round(level.points, 1),
                "explanation": level.explanation,
            }
            for position, level in board.replacement.items()
            if level.players_available
        },
        "warnings": board.warnings,
        "players": players,
        "capabilities": {
            "computed_in_python": [
                "projections", "VBD", "replacement level", "tiers",
                "floor/median/ceiling", "opportunity", "risk", "schedule",
                "offensive environment", "ADP and spread", "Player Score", "Value Score",
            ],
            "computed_in_browser": [
                "snake pick maths",
                "ADP-only survival probability",
                "marginal lineup value",
                "available pool and roster tracking",
            ],
            "requires_local_engine": [
                "roster-aware survival probability",
                "Monte Carlo two-pick expected value",
                "draft-room behaviour and position runs",
                "adaptive strategy state",
                "live Sleeper sync",
            ],
        },
    }


def write_static_site(
    db: Database, cfg: AppConfig, out_dir: Path, limit: int = 400
) -> dict[str, Any]:
    """Write ``board.json`` plus the page into ``out_dir``. Returns a small summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = export_board(db, cfg, limit=limit)

    board_path = out_dir / "board.json"
    board_path.write_text(json.dumps(payload, separators=(",", ":")))

    index_source = STATIC_DIR / "index.html"
    shutil.copyfile(index_source, out_dir / "index.html")

    # Stop Jekyll mangling anything on GitHub Pages.
    (out_dir / ".nojekyll").write_text("")

    return {
        "out_dir": str(out_dir),
        "players": len(payload["players"]),
        "board_bytes": board_path.stat().st_size,
        "generated_at": payload["generated_at"],
    }


def board_frame_preview(db: Database, cfg: AppConfig) -> pl.DataFrame:
    """Small helper used by tests to assert the export matches the live board."""
    return build_board(db, cfg).frame
