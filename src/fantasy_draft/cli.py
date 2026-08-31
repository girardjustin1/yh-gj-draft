"""``ff`` command line interface.

The CLI is the primary interface and the contract the MCP/Claude layer will wrap later.
Every command here must be usable by a human mid-draft: fast, readable, and honest
about what it does not know.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import AppConfig, ConfigError, load_config
from .database import SCHEMA_VERSION, Database, connect
from .logging import configure_logging

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="ff",
    help="Fantasy Draft AI — a local-first NFL snake-draft decision engine.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
config_app = typer.Typer(help="Inspect and validate league configuration.", no_args_is_help=True)
db_app = typer.Typer(help="Database maintenance.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(db_app, name="db")

OK = "[green]OK[/green]"
WARN = "[yellow]WARN[/yellow]"
FAIL = "[red]FAIL[/red]"


@app.callback()
def main(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log at INFO.")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Log at DEBUG.")] = False,
) -> None:
    """Shared setup for every command."""
    level = "DEBUG" if debug else ("INFO" if verbose else None)
    try:
        cfg = load_config()
    except ConfigError as exc:
        configure_logging(level)
        err_console.print(Panel(str(exc), title="Configuration error", border_style="red"))
        raise typer.Exit(code=2) from exc
    cfg.paths.ensure_dirs()
    configure_logging(level, log_dir=cfg.paths.log_dir)
    ctx.obj = cfg


def get_config(ctx: typer.Context) -> AppConfig:
    cfg = ctx.obj
    if cfg is None:  # invoked programmatically without the callback
        cfg = load_config()
        cfg.paths.ensure_dirs()
    return cfg


# --- version -------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"fantasy-draft-ai [bold cyan]{__version__}[/bold cyan]")


# --- doctor --------------------------------------------------------------------------


def _check_python() -> tuple[str, str]:
    v = sys.version_info
    text = f"{v.major}.{v.minor}.{v.micro}"
    return (OK, text) if v >= (3, 12) else (FAIL, f"{text} (need >= 3.12)")


def _check_imports() -> tuple[str, str]:
    required = ["polars", "duckdb", "pyarrow", "pydantic", "httpx", "yaml",
                "typer", "rich", "numpy", "nflreadpy"]
    missing = []
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        return FAIL, "missing: " + ", ".join(missing)
    return OK, f"{len(required)} packages present"


def _check_league(cfg: AppConfig) -> tuple[str, str]:
    league = cfg.league
    if not cfg.league_file_exists:
        return WARN, (
            "using config/league.example.yaml — "
            "run [bold]cp config/league.example.yaml config/league.yaml[/bold]"
        )
    if league.draft.slot is None:
        return WARN, f"{league.label}, draft slot not set yet"
    return OK, f"{league.label}, slot {league.draft.slot}/{league.teams}"


def _check_roster(cfg: AppConfig) -> tuple[str, str]:
    roster = cfg.league.roster
    total = roster.total
    picks = cfg.league.draft.rounds
    detail = f"{roster.starters} starters + {roster.bench} bench = {total} over {picks} rounds"
    if total != picks:
        return WARN, detail + " — roster size and rounds disagree"
    return OK, detail


def _check_weights(cfg: AppConfig) -> tuple[str, str]:
    # Sums are enforced at load time; reaching here means they validated.
    blocks = ["player_score", "value_score", "draft_now"]
    return OK, f"{len(blocks)} weight blocks sum to 1.0"


def _check_db(cfg: AppConfig) -> tuple[str, str]:
    try:
        with connect(cfg.paths.db_path) as db:
            tables = db.table_names()
            version_ = db.schema_version
    except Exception as exc:  # noqa: BLE001 — doctor must never crash
        return FAIL, f"{type(exc).__name__}: {exc}"
    if version_ != SCHEMA_VERSION:
        return WARN, f"schema v{version_} (code expects v{SCHEMA_VERSION}); run [bold]ff db init[/bold]"
    return OK, f"{len(tables)} tables, schema v{version_}"


def _check_data(cfg: AppConfig) -> tuple[str, str]:
    key_tables = ("players", "player_ids", "rankings", "schedules")
    try:
        with connect(cfg.paths.db_path) as db:
            counts = {t: db.row_count(t) for t in key_tables}
    except Exception as exc:  # noqa: BLE001
        return FAIL, f"{type(exc).__name__}: {exc}"
    empty = [t for t, n in counts.items() if n == 0]
    if len(empty) == len(key_tables):
        return WARN, "no data ingested yet — run [bold]ff data refresh[/bold]"
    if empty:
        return WARN, "empty: " + ", ".join(empty)
    return OK, ", ".join(f"{t}={n:,}" for t, n in counts.items())


def _check_network(cfg: AppConfig) -> tuple[str, str]:
    try:
        import nflreadpy as nfl

        teams = nfl.load_teams()
    except Exception as exc:  # noqa: BLE001 — offline is a warning, not a failure
        return WARN, f"nflverse unreachable ({type(exc).__name__}); cached data still usable"
    return OK, f"nflverse reachable ({teams.height} team rows)"


def _check_paths(cfg: AppConfig) -> tuple[str, str]:
    missing = [
        str(p)
        for p in (cfg.paths.data_dir, cfg.paths.raw_dir, cfg.paths.cache_dir, cfg.paths.log_dir)
        if not p.is_dir()
    ]
    if missing:
        return FAIL, "missing: " + ", ".join(missing)
    return OK, str(cfg.paths.data_dir)


@app.command()
def doctor(
    ctx: typer.Context,
    skip_network: Annotated[
        bool, typer.Option("--skip-network", help="Do not contact nflverse.")
    ] = False,
) -> None:
    """Validate environment, configuration, database and data freshness."""
    cfg = get_config(ctx)

    checks: list[tuple[str, tuple[str, str]]] = [
        ("Python", _check_python()),
        ("Dependencies", _check_imports()),
        ("Paths", _check_paths(cfg)),
        ("League config", _check_league(cfg)),
        ("Roster / rounds", _check_roster(cfg)),
        ("Scoring weights", _check_weights(cfg)),
        ("Database", _check_db(cfg)),
        ("Ingested data", _check_data(cfg)),
    ]
    if not skip_network:
        checks.append(("Network", _check_network(cfg)))

    table = Table(title="ff doctor", title_style="bold", header_style="bold", expand=False)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail", overflow="fold")
    for name, (status, detail) in checks:
        table.add_row(name, status, detail)
    console.print(table)

    failures = [n for n, (s, _) in checks if s == FAIL]
    warnings = [n for n, (s, _) in checks if s == WARN]

    if failures:
        console.print(f"\n[red]{len(failures)} check(s) failed:[/red] {', '.join(failures)}")
        raise typer.Exit(code=1)
    if warnings:
        console.print(
            f"\n[yellow]{len(warnings)} warning(s):[/yellow] {', '.join(warnings)}"
            "\nThe engine will run, but see [bold]HUMAN_TODO.md[/bold]."
        )
    else:
        console.print("\n[green]All checks passed.[/green]")


# --- config --------------------------------------------------------------------------


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Show the resolved league configuration."""
    cfg = get_config(ctx)
    league = cfg.league
    roster = league.roster

    source = "config/league.yaml" if cfg.league_file_exists else "config/league.example.yaml"
    header = Text.assemble(
        (league.name, "bold"), "  ", (league.label, "cyan"), "\n",
        ("source: ", "dim"), (source, "dim"),
    )
    console.print(Panel(header, border_style="cyan"))

    basics = Table(show_header=False, box=None, pad_edge=False)
    basics.add_column(style="dim", no_wrap=True)
    basics.add_column()
    basics.add_row("Season", str(league.season))
    basics.add_row("Platform", league.platform)
    basics.add_row("League ID", league.league_id or "[dim]not set[/dim]")
    basics.add_row("Teams", str(league.teams))
    basics.add_row("Draft", f"{league.draft.type}, {league.draft.rounds} rounds")
    basics.add_row("Draft slot", str(league.draft.slot) if league.draft.slot else "[dim]not set[/dim]")
    basics.add_row("Scoring", league.scoring_type)
    basics.add_row("Regular season", f"weeks {min(league.regular_season_weeks)}-{max(league.regular_season_weeks)}")
    basics.add_row("Playoffs", ", ".join(str(w) for w in league.playoff_weeks))
    console.print(basics)

    lineup = Table(title="Starting lineup", header_style="bold", title_justify="left")
    lineup.add_column("Slot", style="cyan")
    lineup.add_column("Count", justify="right")
    lineup.add_column("League-wide starters", justify="right")
    for label, count in [("QB", roster.qb), ("RB", roster.rb), ("WR", roster.wr), ("TE", roster.te)]:
        lineup.add_row(label, str(count), f"{league.starter_demand(label):.1f}")
    for name, count in roster.flex_counts.items():
        lineup.add_row(name, str(count), "[dim]shared[/dim]")
    for label, count in [("K", roster.k), ("DST", roster.dst)]:
        if count:
            lineup.add_row(label, str(count), f"{league.starter_demand(label):.1f}")
    lineup.add_row("Bench", str(roster.bench), "")
    lineup.add_row("[bold]Total[/bold]", f"[bold]{roster.total}[/bold]", "")
    console.print(lineup)

    scoring = Table(title="Scoring", header_style="bold", title_justify="left")
    scoring.add_column("Rule", style="cyan")
    scoring.add_column("Value", justify="right")
    s = league.scoring
    scoring.add_row("Reception", f"{s.reception:g}")
    scoring.add_row("Passing yards / point", f"{s.passing_yards_per_point:g}")
    scoring.add_row("Passing TD", f"{s.passing_td:g}")
    scoring.add_row("Interception", f"{s.passing_interception:g}")
    scoring.add_row("Rushing yards / point", f"{s.rushing_yards_per_point:g}")
    scoring.add_row("Rushing TD", f"{s.rushing_td:g}")
    scoring.add_row("Receiving yards / point", f"{s.receiving_yards_per_point:g}")
    scoring.add_row("Receiving TD", f"{s.receiving_td:g}")
    scoring.add_row("Fumble lost", f"{s.fumble_lost:g}")
    console.print(scoring)


@config_app.command("weights")
def config_weights(ctx: typer.Context) -> None:
    """Show the scoring weights currently in effect."""
    cfg = get_config(ctx)
    w = cfg.weights
    for title, block in (
        ("Player Score", w.player_score),
        ("Value Score", w.value_score),
        ("Draft Now Score", w.draft_now),
    ):
        table = Table(title=title, header_style="bold", title_justify="left")
        table.add_column("Component", style="cyan")
        table.add_column("Weight", justify="right")
        for key, value in block.as_dict().items():
            table.add_row(key, f"{value:.3f}")
        table.add_row("[bold]Total[/bold]", f"[bold]{sum(block.as_dict().values()):.3f}[/bold]")
        console.print(table)

    strat = Table(title="Strategy priors", header_style="bold", title_justify="left")
    strat.add_column("Strategy", style="cyan")
    strat.add_column("Prior", justify="right")
    for key, value in w.strategy_priors.as_dict().items():
        strat.add_row(key, f"{value:.2f}")
    console.print(strat)


@config_app.command("validate")
def config_validate(ctx: typer.Context) -> None:
    """Validate configuration files and exit non-zero if anything is wrong."""
    cfg = get_config(ctx)
    console.print(f"{OK} config/scoring_weights.yaml")
    console.print(f"{OK} config/data_sources.yaml ({len(cfg.data_sources.enabled_sources())} sources enabled)")
    if cfg.league_file_exists:
        console.print(f"{OK} config/league.yaml — {cfg.league.label}")
    else:
        console.print(f"{WARN} config/league.yaml missing; using the example file")
    if cfg.league.draft.slot is None:
        console.print(f"{WARN} draft.slot is not set — required before [bold]ff on-clock[/bold]")


# --- db ------------------------------------------------------------------------------


@db_app.command("init")
def db_init(ctx: typer.Context) -> None:
    """Create or update the DuckDB schema."""
    cfg = get_config(ctx)
    with connect(cfg.paths.db_path) as db:
        console.print(
            f"{OK} {cfg.paths.db_path} — {len(db.table_names())} tables, schema v{db.schema_version}"
        )


@db_app.command("tables")
def db_tables(ctx: typer.Context) -> None:
    """List tables and row counts."""
    cfg = get_config(ctx)
    with connect(cfg.paths.db_path) as db:
        counts = db.table_counts()
    table = Table(title=str(cfg.paths.db_path), header_style="bold", title_justify="left")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right")
    for name, count in counts.items():
        style = "dim" if count == 0 else ""
        table.add_row(f"[{style}]{name}[/{style}]" if style else name, f"{count:,}")
    console.print(table)


@db_app.command("reset")
def db_reset(
    ctx: typer.Context,
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation.")] = False,
) -> None:
    """Delete the database file and recreate an empty schema."""
    cfg = get_config(ctx)
    path = cfg.paths.db_path
    if not yes:
        typer.confirm(f"Delete {path} and all ingested data?", abort=True)
    if path.exists():
        path.unlink()
    for sidecar in path.parent.glob(path.name + ".wal"):
        sidecar.unlink()
    with connect(path) as db:
        console.print(f"{OK} recreated {path} ({len(db.table_names())} tables)")


__all__ = ["app", "console", "get_config", "Database"]
