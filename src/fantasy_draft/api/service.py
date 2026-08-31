"""Local JSON API and the draft dashboard it serves.

**This layer computes nothing.** Every endpoint is a thin wrapper over
:func:`fantasy_draft.service.analyze_current_pick`, which is the same function the CLI
and Claude call. Business logic in an HTTP handler is how a GUI and an assistant end up
disagreeing mid-draft, so there is none here.

It binds to localhost by default. The data is your league's; nothing leaves the machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import AppConfig, load_config
from ..database import connect
from ..logging import configure_logging, get_logger
from ..service import (
    NoDraftError,
    ServiceError,
    UnknownSlotError,
    analyze_current_pick,
    compare_picks,
)

log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class OnClockRequest(BaseModel):
    refresh: bool = Field(True, description="Pull the live draft before analysing.")
    simulate: bool = Field(True, description="Run the Monte Carlo two-pick model.")
    iterations: int | None = Field(None, ge=100, le=100_000)
    limit: int = Field(60, ge=1, le=300, description="Rows of BEST AVAILABLE to return.")
    draft_id: str | None = None


class PickRequest(BaseModel):
    """Record a selection from the interface — the swipe-to-draft path."""

    player_key: str | None = None
    name: str | None = None
    draft_id: str | None = None


class StartDraftRequest(BaseModel):
    slot: int | None = Field(None, ge=1, le=32)
    draft_id: str = "manual-draft"


class CompareRequest(BaseModel):
    player_keys: list[str] = Field(min_length=2, max_length=6)
    refresh: bool = False
    draft_id: str | None = None


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    """Build the application. Accepts a config so tests can point at a temp database."""
    configuration = cfg or load_config()
    configuration.paths.ensure_dirs()
    configure_logging(log_dir=configuration.paths.log_dir)

    app = FastAPI(
        title="Fantasy Draft AI",
        description="Local snake-draft decision engine. All computation happens in Python.",
        version="0.2.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.cfg = configuration

    def config_of() -> AppConfig:
        return app.state.cfg

    # --- error handling ---------------------------------------------------------------
    #
    # Service errors carry a message written for a human mid-draft, with the specific fix
    # in it. They are surfaced verbatim rather than flattened to "500 Internal Error".

    @app.exception_handler(ServiceError)
    async def service_error_handler(_request: Any, exc: ServiceError) -> JSONResponse:
        status = 409 if isinstance(exc, NoDraftError | UnknownSlotError) else 400
        return JSONResponse(
            status_code=status,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    # --- endpoints --------------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        cfg = config_of()
        with connect(cfg.paths.db_path, read_only=False) as db:
            counts = {
                table: db.row_count(table)
                for table in ("players", "rankings", "projections", "draft_picks")
            }
        return {"status": "ok", "database": str(cfg.paths.db_path), "rows": counts}

    @app.get("/api/config")
    def configuration_endpoint() -> dict[str, Any]:
        """League settings and the state of the active draft provider."""
        from ..draft.providers import provider_status

        cfg = config_of()
        league = cfg.league
        with connect(cfg.paths.db_path) as db:
            provider = provider_status(cfg, db)
        return {
            "league": {
                "name": league.name,
                "season": league.season,
                "label": league.label,
                "teams": league.teams,
                "scoring_type": league.scoring_type,
                "reception": league.scoring.reception,
                "passing_td": league.scoring.passing_td,
                "draft_type": league.draft.type,
                "rounds": league.draft.rounds,
                "slot": league.draft.slot,
                "third_round_reversal": league.draft.third_round_reversal,
                "is_superflex": league.roster.is_superflex,
                "roster": {
                    "starters": league.roster.starters,
                    "bench": league.roster.bench,
                    "total": league.roster.total,
                    "slots": {
                        **league.roster.dedicated,
                        **league.roster.flex_counts,
                    },
                },
            },
            "provider": provider,
            "weights": {
                "player_score": cfg.weights.player_score.as_dict(),
                "value_score": cfg.weights.value_score.as_dict(),
                "draft_now": cfg.weights.draft_now.as_dict(),
            },
        }

    @app.post("/api/on-clock")
    def on_clock(request: OnClockRequest) -> dict[str, Any]:
        """The primary action. Everything the five-area interface needs, in one call."""
        cfg = config_of()
        with connect(cfg.paths.db_path) as db:
            analysis = analyze_current_pick(
                cfg, db,
                draft_id=request.draft_id,
                refresh=request.refresh,
                simulate=request.simulate,
                iterations=request.iterations,
            )
            return analysis.to_dict(board_limit=request.limit)

    @app.post("/api/compare")
    def compare(request: CompareRequest) -> dict[str, Any]:
        """Head-to-head: what happens to our next pick under each choice?"""
        cfg = config_of()
        with connect(cfg.paths.db_path) as db:
            analysis = analyze_current_pick(
                cfg, db, draft_id=request.draft_id, refresh=request.refresh
            )
            return compare_picks(analysis, request.player_keys)

    @app.post("/api/pick")
    def record(request: PickRequest) -> dict[str, Any]:
        """Mark a player drafted by hand.

        The offline path: leagues on platforms we cannot read live, in-person drafts, or
        an API outage. A pick recorded here is indistinguishable downstream from a synced
        one.
        """
        from .. import queries
        from ..draft.store import load_state, record_pick

        cfg = config_of()
        with connect(cfg.paths.db_path) as db:
            state = load_state(db, request.draft_id)
            if state is None:
                raise HTTPException(
                    status_code=409,
                    detail="No draft in progress. Start one first.",
                )
            player_key, name, position, team = request.player_key, request.name, None, None
            if player_key is None:
                if not request.name:
                    raise HTTPException(status_code=422, detail="Give a player_key or a name.")
                match = queries.resolve_one(db, request.name)
                if match is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Could not resolve {request.name!r} to exactly one player.",
                    )
                player_key, name = match["player_key"], match["full_name"]
                position, team = match["position"], match["team"]
            else:
                row = queries.get_player(db, player_key)
                if row is None:
                    raise HTTPException(status_code=404, detail="Unknown player_key.")
                name, position, team = row["full_name"], row["position"], row["team"]

            try:
                state = record_pick(db, state, player_key, name, position, team)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

            last = state.picks[-1]
            return {
                "recorded": {
                    "overall": last.overall,
                    "pick_label": state.board.label(last.overall),
                    "slot": last.slot,
                    "name": last.player_name,
                    "position": last.position,
                    "was_ours": last.slot == state.my_slot,
                },
                "picks_made": state.picks_made,
                "is_my_pick": state.is_my_pick,
                "picks_until_my_turn": state.picks_until_my_turn,
                "pick_label": state.pick_label,
            }

    @app.post("/api/undo")
    def undo(request: PickRequest) -> dict[str, Any]:
        """Remove the most recently recorded selection."""
        from ..draft.store import load_state, undo_pick

        cfg = config_of()
        with connect(cfg.paths.db_path) as db:
            state = load_state(db, request.draft_id)
            if state is None:
                raise HTTPException(status_code=409, detail="No draft in progress.")
            removed = undo_pick(db, state)
            return {
                "removed": (
                    {"name": removed.player_name, "overall": removed.overall}
                    if removed else None
                ),
                "picks_made": state.picks_made,
                "pick_label": state.pick_label,
                "is_my_pick": state.is_my_pick,
            }

    @app.post("/api/draft/start")
    def start_draft(request: StartDraftRequest) -> dict[str, Any]:
        """Begin an empty draft to fill in by hand."""
        from ..draft.store import create_manual_draft

        cfg = config_of()
        league = cfg.league
        if request.slot is not None:
            league = league.model_copy(
                update={"draft": league.draft.model_copy(update={"slot": request.slot})}
            )
        if league.draft.slot is None:
            raise HTTPException(
                status_code=422,
                detail="A draft slot is required. Pass slot, or set draft.slot in league.yaml.",
            )
        if league.draft.slot > league.teams:
            raise HTTPException(
                status_code=422,
                detail=f"Slot {league.draft.slot} exceeds {league.teams} teams.",
            )
        with connect(cfg.paths.db_path) as db:
            state = create_manual_draft(db, league, draft_id=request.draft_id)
        return {
            "draft_id": state.draft_id,
            "teams": state.board.teams,
            "rounds": state.board.rounds,
            "my_slot": state.my_slot,
            "pick_label": state.pick_label,
        }

    @app.get("/api/board")
    def board(limit: int = 60, position: str | None = None) -> dict[str, Any]:
        """The static board, for when no draft is running."""
        from ..analytics.board import build_board
        from ..service import _board_row

        cfg = config_of()
        with connect(cfg.paths.db_path) as db:
            built = build_board(db, cfg)
        if built.frame.is_empty():
            raise HTTPException(
                status_code=409,
                detail="No board available. Run `ff data refresh` first.",
            )
        frame = built.frame
        if position and position.upper() != "ALL":
            frame = frame.filter(frame["position"] == position.upper())
        rows = frame.head(limit).to_dicts()
        return {
            "players": [_board_row(row, i) for i, row in enumerate(rows, start=1)],
            "warnings": built.warnings,
        }

    # --- the page ----------------------------------------------------------------------

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = None  # populated by `ff serve`; import-time config loading would fight the tests


def run(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Start the server."""
    import uvicorn

    uvicorn.run(
        "fantasy_draft.api.service:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="warning",
    )
