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


def record_pick(
    db: Database,
    state: DraftState,
    player_key: str,
    player_name: str | None = None,
    position: str | None = None,
    nfl_team: str | None = None,
    mine: bool | None = None,
) -> DraftState:
    """Append one selection to a stored draft by hand.

    The fallback for any platform we cannot read live — Yahoo today, a league drafting in
    a room, or an API outage mid-draft. The engine cannot tell the difference: a manually
    entered pick produces exactly the same ``DraftState`` a synced one does.

    ``mine`` says whose pick this was. Left as None it is inferred from the snake, which
    is right whenever every pick has been entered in order. Passing it explicitly makes
    your own roster correct even if the order slipped — and your roster is the input that
    drives what you still need, so it is the one worth being certain about.
    """
    overall = state.picks_made + 1
    if overall > state.board.total_picks:
        raise ValueError(
            f"The draft is complete ({state.board.total_picks} picks). Nothing to record."
        )
    if player_key in state.drafted_keys:
        raise ValueError(f"{player_name or player_key} has already been drafted.")

    slot = state.board.slot_for(overall)
    inferred_team = state.slot_to_team.get(slot, f"slot-{slot:02d}")
    if mine is True:
        team_id = state.my_team_id or inferred_team
    elif mine is False and state.my_team_id and inferred_team == state.my_team_id:
        # Explicitly not ours, but the snake says it is our turn — the order has drifted.
        # Park it on a neutral team so it never lands in our roster.
        team_id = f"slot-{slot:02d}-other"
    else:
        team_id = inferred_team

    pick = DraftPick(
        overall=overall,
        round=state.board.round_for(overall),
        slot=slot,
        team_id=team_id,
        player_key=player_key,
        player_name=player_name,
        position=position,
        nfl_team=nfl_team,
        picked_at=datetime.now(),
    )
    state.picks.append(pick)
    state.synced_at = datetime.now()
    save_state(db, state)
    return state


def undo_pick(db: Database, state: DraftState) -> DraftPick | None:
    """Remove the most recent selection. Returns it, or None if the draft was empty."""
    if not state.picks:
        return None
    removed = state.picks.pop()
    state.synced_at = datetime.now()
    with db.transaction() as conn:
        conn.execute(
            "DELETE FROM draft_picks WHERE draft_id = ? AND overall = ?",
            [state.draft_id, removed.overall],
        )
    save_state(db, state)
    return removed


def create_manual_draft(
    db: Database, league: Any, draft_id: str = "manual-draft"
) -> DraftState:
    """Start an empty draft we will fill in by hand."""
    state = DraftState(
        draft_id=draft_id,
        platform="manual",
        season=league.season,
        board=SnakeBoard.from_league(league),
        picks=[],
        slot_to_team={s: f"slot-{s:02d}" for s in range(1, league.teams + 1)},
        my_slot=league.draft.slot,
        my_team_id=f"slot-{league.draft.slot:02d}" if league.draft.slot else None,
        status="drafting",
        synced_at=datetime.now(),
    )
    save_state(db, state)
    return state


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
