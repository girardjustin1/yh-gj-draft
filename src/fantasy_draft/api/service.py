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
