"""Persisting draft state to DuckDB.

Saving every sync serves two purposes. It keeps the last known board available when
Sleeper is unreachable — clearly labelled stale rather than silently missing — and it
leaves a record that later makes backtesting possible.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import polars as pl

from ..database import Database
from ..models import DraftPick
from .snake import SnakeBoard
from .state import DraftState


def save_state(db: Database, state: DraftState, settings: dict[str, Any] | None = None) -> None:
    """Write a draft, its picks, and its roster/slot map."""
    drafts = pl.DataFrame(
        [
            {
                "draft_id": state.draft_id,
                "platform": state.platform,
                "league_id": (settings or {}).get("league_id"),
                "season": state.season,
                "draft_type": state.board.draft_type,
                "teams": state.board.teams,
                "rounds": state.board.rounds,
                "my_slot": state.my_slot,
                "my_team_id": state.my_team_id,
                "status": state.status,
                "settings": json.dumps(
                    {
                        "third_round_reversal": state.board.third_round_reversal,
                        **(settings or {}),
                    }
                ),
                "synced_at": state.synced_at,
            }
        ]
    )
    db.upsert_table("drafts", drafts, keys=["draft_id"])

    if state.picks:
        picks = pl.DataFrame(
            [
                {
                    "draft_id": state.draft_id,
                    "overall": p.overall,
                    "round": p.round,
                    "slot": p.slot,
                    "team_id": p.team_id,
                    "player_key": p.player_key,
                    "source_player_id": None,
                    "player_name": p.player_name,
                    "position": p.position,
                    "nfl_team": p.nfl_team,
                    "is_keeper": p.is_keeper,
                    "picked_at": p.picked_at,
                    "synced_at": state.synced_at,
                }
                for p in state.picks
            ]
        )
        db.upsert_table("draft_picks", picks, keys=["draft_id"])

    if state.slot_to_team:
        rosters = pl.DataFrame(
            [
                {
                    "draft_id": state.draft_id,
                    "team_id": team_id,
                    "slot": slot,
                    "is_me": team_id == state.my_team_id,
                    "display_name": state.team_names.get(team_id),
                    "synced_at": state.synced_at,
                }
                for slot, team_id in state.slot_to_team.items()
            ]
        )
        db.upsert_table("draft_rosters", rosters, keys=["draft_id"])


def load_state(db: Database, draft_id: str | None = None) -> DraftState | None:
    """Load the most recently synced draft, or a specific one."""
    if draft_id:
        header = db.query("SELECT * FROM drafts WHERE draft_id = ?", [draft_id])
    else:
        header = db.query("SELECT * FROM drafts ORDER BY synced_at DESC LIMIT 1")
    if header.is_empty():
        return None

    row = header.to_dicts()[0]
    settings = json.loads(row.get("settings") or "{}")
    board = SnakeBoard(
        teams=int(row["teams"] or 12),
        rounds=int(row["rounds"] or 15),
        draft_type=row["draft_type"] or "snake",
        third_round_reversal=bool(settings.get("third_round_reversal")),
    )

    picks_frame = db.query(
        "SELECT * FROM draft_picks WHERE draft_id = ? ORDER BY overall", [row["draft_id"]]
    )
    picks = [
        DraftPick(
            overall=int(p["overall"]),
            round=int(p["round"]),
            slot=int(p["slot"]),
            team_id=p["team_id"] or str(p["slot"]),
            player_key=p["player_key"],
            player_name=p["player_name"],
            position=p["position"],
            nfl_team=p["nfl_team"],
            is_keeper=bool(p["is_keeper"]),
            picked_at=p["picked_at"],
        )
        for p in picks_frame.to_dicts()
    ]

    roster_frame = db.query(
        "SELECT * FROM draft_rosters WHERE draft_id = ?", [row["draft_id"]]
    )
    slot_to_team = {
        int(r["slot"]): r["team_id"] for r in roster_frame.to_dicts() if r["slot"]
    }
    team_names = {
        r["team_id"]: r["display_name"]
        for r in roster_frame.to_dicts()
        if r.get("display_name")
    }

    return DraftState(
        draft_id=row["draft_id"],
        platform=row["platform"],
        season=int(row["season"] or 0),
        board=board,
        picks=picks,
        slot_to_team=slot_to_team,
        team_names=team_names,
        my_slot=int(row["my_slot"]) if row["my_slot"] else None,
        my_team_id=row["my_team_id"],
        status=row["status"] or "unknown",
        synced_at=row["synced_at"] or datetime.now(),
    )
