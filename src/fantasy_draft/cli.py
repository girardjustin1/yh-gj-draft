"""``ff`` command line interface.

The CLI is the primary interface and the contract the MCP/Claude layer will wrap later.
Every command here must be usable by a human mid-draft: fast, readable, and honest
about what it does not know.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import polars as pl
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
data_app = typer.Typer(help="Ingest and inspect NFL/fantasy data.", no_args_is_help=True)
import_app = typer.Typer(help="Import your own projections or ADP.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(db_app, name="db")
app.add_typer(data_app, name="data")
app.add_typer(import_app, name="import")

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


# --- data ----------------------------------------------------------------------------


def _humanize_age(hours: float | None) -> str:
    if hours is None:
        return "[red]never[/red]"
    if hours < 1 / 60:
        return "just now"
    if hours < 1:
        return f"{hours * 60:.0f} min ago"
    if hours < 48:
        return f"{hours:.1f}h ago"
    return f"{hours / 24:.1f}d ago"


@data_app.command("refresh")
def data_refresh(
    ctx: typer.Context,
    only: Annotated[
        list[str] | None,
        typer.Option("--only", help="Refresh just these sources. Repeatable."),
    ] = None,
) -> None:
    """Refresh NFL and fantasy datasets from nflverse.

    Sources are refreshed independently: one failing is reported and the rest continue.
    """
    from .data.nflverse import NflverseIngest

    cfg = get_config(ctx)
    with connect(cfg.paths.db_path) as db:
        ingest = NflverseIngest(cfg, db)
        known = set(ingest.datasets())
        if only:
            unknown = set(only) - known
            if unknown:
                err_console.print(
                    f"[red]Unknown source(s):[/red] {', '.join(sorted(unknown))}\n"
                    f"Available: {', '.join(sorted(known))}"
                )
                raise typer.Exit(code=2)

        table = Table(title="ff data refresh", header_style="bold", title_justify="left")
        table.add_column("Source", style="cyan", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Rows", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("Detail", overflow="fold")

        with console.status("[cyan]Refreshing...", spinner="dots"):
            results = ingest.refresh(only=list(only) if only else None)

        for result in results:
            status = {"ok": OK, "failed": FAIL, "skipped": "[dim]SKIP[/dim]"}[result.status]
            table.add_row(
                result.source,
                status,
                f"{result.rows:,}" if result.rows else "",
                f"{result.duration:.1f}s" if result.duration else "",
                result.message,
            )
        console.print(table)

        # Projections are derived from the rankings we just loaded, so they are stale
        # the moment a refresh lands.
        projection_rows = 0
        if not only or "fantasypros_ecr" in only:
            from .analytics.projections import refresh_derived_projections

            try:
                projection_rows = refresh_derived_projections(db, cfg)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Could not derive projections: {exc}[/yellow]")

        unresolved = db.row_count("unresolved_players")

    if projection_rows:
        console.print(
            f"[green]OK[/green] derived {projection_rows:,} projections from the "
            f"consensus board and the historical positional value curve"
        )

    failed = [r.source for r in results if r.status == "failed"]
    if failed:
        console.print(
            f"\n[yellow]{len(failed)} source(s) failed:[/yellow] {', '.join(failed)}"
            "\nThe engine still runs; affected components will report lower confidence."
        )
    if unresolved:
        console.print(
            f"[yellow]{unresolved} player(s) unresolved[/yellow] — "
            "inspect with [bold]ff data unresolved-players[/bold]"
        )


@data_app.command("status")
def data_status(ctx: typer.Context) -> None:
    """Show what data we hold and how stale it is."""
    from .models import DataFreshness

    cfg = get_config(ctx)
    from .data.nflverse import NflverseIngest

    with connect(cfg.paths.db_path) as db:
        sources = list(NflverseIngest(cfg, db).datasets())
        rows: list[DataFreshness] = []
        for source in sources:
            spec = cfg.data_sources.spec(source)
            entry = db.last_refresh(source)
            rows.append(
                DataFreshness(
                    source=source,
                    updated_at=entry["ingested_at"] if entry else None,
                    rows=entry["rows"] if entry else None,
                    max_age_hours=spec.max_age_hours,
                    ok=entry is not None,
                )
            )
        counts = db.table_counts()

    table = Table(title="Data freshness", header_style="bold", title_justify="left")
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Updated", no_wrap=True)
    table.add_column("Rows", justify="right")
    table.add_column("Stale after", justify="right")
    for row in rows:
        age = _humanize_age(row.age_hours)
        marker = f"[yellow]{age}[/yellow]" if row.is_stale and row.updated_at else age
        table.add_row(
            row.source,
            marker,
            f"{row.rows:,}" if row.rows else "",
            f"{row.max_age_hours:g}h",
        )
    console.print(table)

    stale = [r.source for r in rows if r.is_stale]
    if stale:
        console.print(
            f"[yellow]{len(stale)} source(s) stale or never loaded:[/yellow] {', '.join(stale)}"
        )
    else:
        console.print("[green]All sources fresh.[/green]")

    populated = {k: v for k, v in counts.items() if v}
    console.print(
        f"\n[dim]{len(populated)} populated tables, "
        f"{sum(populated.values()):,} rows total.[/dim]"
    )


@data_app.command("unresolved-players")
def data_unresolved(
    ctx: typer.Context,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Rows to show.")] = 40,
) -> None:
    """Show player records we could not confidently map to a canonical identity."""
    cfg = get_config(ctx)
    with connect(cfg.paths.db_path) as db:
        frame = db.query(
            """
            SELECT source, raw_name, position, team, reason, candidates, seen_at
            FROM unresolved_players ORDER BY seen_at DESC, raw_name LIMIT ?
            """,
            [limit],
        )
        total = db.row_count("unresolved_players")

    if not total:
        console.print("[green]No unresolved players.[/green]")
        return

    table = Table(
        title=f"Unresolved players ({total} total)", header_style="bold", title_justify="left"
    )
    for column in ("Source", "Name", "Pos", "Team", "Reason", "Candidates"):
        table.add_column(column, style="cyan" if column == "Source" else "")
    for row in frame.iter_rows(named=True):
        table.add_row(
            row["source"], row["raw_name"], row["position"] or "",
            row["team"] or "", row["reason"], (row["candidates"] or "")[:40],
        )
    console.print(table)
    console.print(
        "\n[dim]These are never merged into another player. Ambiguous rows are excluded\n"
        "from the board; unmatched rows keep a synthetic identity with lower confidence.[/dim]"
    )


@data_app.command("sources")
def data_sources(ctx: typer.Context) -> None:
    """List configured data sources."""
    cfg = get_config(ctx)
    table = Table(title="Configured sources", header_style="bold", title_justify="left")
    table.add_column("Source", style="cyan")
    table.add_column("Enabled", no_wrap=True)
    table.add_column("Stale after", justify="right")
    table.add_column("Notes", overflow="fold")
    for name, spec in cfg.data_sources.sources.items():
        table.add_row(
            name,
            "[green]yes[/green]" if spec.enabled else "[dim]no[/dim]",
            f"{spec.max_age_hours:g}h",
            spec.notes,
        )
    console.print(table)


# --- import --------------------------------------------------------------------------


@import_app.command("projections")
def import_projections_cmd(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="CSV, Parquet, or JSON file.")],
    source: Annotated[
        str | None, typer.Option("--source", help="Label for this source. Defaults to the filename.")
    ] = None,
) -> None:
    """Import a projection file. Sources are kept separately, never overwritten."""
    from .analytics.projections import ProjectionImportError, import_projections

    cfg = get_config(ctx)
    with connect(cfg.paths.db_path) as db:
        if db.row_count("players") == 0:
            err_console.print("Run [bold]ff data refresh[/bold] first so names can be matched.")
            raise typer.Exit(code=1)
        try:
            frame, unresolved = import_projections(db, cfg, path, source=source)
        except ProjectionImportError as exc:
            err_console.print(Panel(str(exc), title="Import failed", border_style="red"))
            raise typer.Exit(code=1) from exc

        # A synthetic key ("nm-", "slp-", "fp-") means we kept the player but could
        # not tie them to our canonical universe. That is not a match.
        known = set(db.query("SELECT player_key FROM players")["player_key"])
        matched = int(frame.filter(pl.col("player_key").is_in(list(known))).height)

    console.print(
        f"{OK} imported [bold]{frame.height:,}[/bold] projections from {path.name} "
        f"([bold]{matched:,}[/bold] matched to a canonical player)"
    )
    if unresolved:
        console.print(
            f"[yellow]{unresolved} name(s) could not be resolved[/yellow] — "
            "see [bold]ff data unresolved-players[/bold]"
        )
    console.print("[dim]Run [bold]ff board[/bold] to see the blended result.[/dim]")


@import_app.command("list")
def import_list(ctx: typer.Context) -> None:
    """Show every projection source currently stored."""
    cfg = get_config(ctx)
    with connect(cfg.paths.db_path) as db:
        frame = db.query(
            """
            SELECT source, count(*) AS players, avg(fantasy_points) AS avg_points,
                   max(source_updated_at) AS updated, max(ingested_at) AS imported
            FROM projections WHERE season = ? GROUP BY source ORDER BY source
            """,
            [cfg.league.season],
        )
    if frame.is_empty():
        console.print("No projections stored. Run [bold]ff data refresh[/bold].")
        return
    table = Table(title="Projection sources", header_style="bold", title_justify="left")
    for column in ("Source", "Players", "Avg points", "Imported"):
        table.add_column(column, justify="right" if column != "Source" else "left")
    for row in frame.iter_rows(named=True):
        table.add_row(
            row["source"], f"{row['players']:,}", _num(row["avg_points"], 1),
            str(row["imported"])[:19],
        )
    console.print(table)
    console.print(
        "[dim]Sources are blended by the weights in config/data_sources.yaml. "
        "The derived curve is always kept as a floor so no player drops off the board.[/dim]"
    )


# --- players -------------------------------------------------------------------------


@app.command("players")
def players_cmd(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Player name, or part of one.")],
    position: Annotated[
        str | None, typer.Option("--position", "-p", help="Filter by position.")
    ] = None,
) -> None:
    """Inspect a player: identity, market rank, usage, and role."""
    from . import queries

    cfg = get_config(ctx)
    with connect(cfg.paths.db_path) as db:
        if db.row_count("players") == 0:
            err_console.print("No data ingested. Run [bold]ff data refresh[/bold] first.")
            raise typer.Exit(code=1)

        matches = queries.search_players(db, name, position=position, limit=10)
        if matches.is_empty():
            err_console.print(f"No player matching [bold]{name}[/bold].")
            raise typer.Exit(code=1)

        if matches.height > 1 and matches["match_rank"][0] != 0:
            table = Table(title=f"Matches for '{name}'", header_style="bold", title_justify="left")
            for column in ("Name", "Pos", "Team", "ECR"):
                table.add_column(column)
            for row in matches.iter_rows(named=True):
                table.add_row(
                    row["full_name"], row["position"] or "", row["team"] or "",
                    f"{row['ecr']:.1f}" if row["ecr"] is not None else "",
                )
            console.print(table)
            console.print("[dim]Be more specific to see a single player.[/dim]")
            return

        player = matches.to_dicts()[0]
        key = player["player_key"]

        bye = f"bye {player['bye_week']}" if player["bye_week"] else "bye unknown"
        console.print(
            Panel(
                Text.assemble(
                    (player["full_name"], "bold"), "  ",
                    (f"{player['position']} · {player['team']} · {bye}", "cyan"), "\n",
                    (f"player_key: {key}", "dim"),
                ),
                border_style="cyan",
            )
        )

        rankings = queries.player_rankings(db, key)
        overall = rankings.filter(pl.col("ranking_type") == "redraft-overall")
        positional = rankings.filter(
            pl.col("ranking_type").str.starts_with("redraft-")
            & (pl.col("ranking_type") != "redraft-overall")
        )
        if overall.height:
            row = overall.to_dicts()[0]
            market = Table(title="Market (FantasyPros ECR)", header_style="bold", title_justify="left")
            for column in ("Board", "ECR", "SD", "Best", "Worst"):
                market.add_column(column, justify="right" if column != "Board" else "left")
            market.add_row(
                "Overall", f"{row['ecr']:.1f}",
                f"{row['sd']:.1f}" if row["sd"] is not None else "-",
                str(row["best"] or "-"), str(row["worst"] or "-"),
            )
            for pos_row in positional.to_dicts():
                market.add_row(
                    pos_row["ranking_type"].removeprefix("redraft-").upper(),
                    f"{pos_row['ecr']:.1f}",
                    f"{pos_row['sd']:.1f}" if pos_row["sd"] is not None else "-",
                    str(pos_row["best"] or "-"), str(pos_row["worst"] or "-"),
                )
            console.print(market)
        else:
            console.print("[yellow]No consensus ranking for this player.[/yellow]")

        stats = queries.player_season_stats(db, key)
        if stats.height:
            usage = Table(title="Usage by season", header_style="bold", title_justify="left")
            for column in ("Season", "G", "Car", "RuYd", "RuTD", "Tgt", "Rec", "ReYd", "ReTD", "Tgt%"):
                usage.add_column(column, justify="right" if column != "Season" else "left")
            for row in stats.to_dicts():
                usage.add_row(
                    str(row["season"]), str(row["games"]),
                    _num(row["carries"]), _num(row["rush_yards"]), _num(row["rush_tds"]),
                    _num(row["targets"]), _num(row["receptions"]), _num(row["rec_yards"]),
                    _num(row["rec_tds"]),
                    f"{row['target_share'] * 100:.1f}%" if row["target_share"] else "-",
                )
            console.print(usage)

        opportunity = queries.player_opportunity(db, key)
        snaps = queries.player_snaps(db, key)
        if opportunity.height:
            merged = opportunity.join(snaps.select("season", "snap_share"), on="season", how="left")
            opp = Table(
                title="Opportunity: expected vs actual points",
                header_style="bold", title_justify="left",
            )
            for column in ("Season", "G", "Expected", "Actual", "Diff", "Snap%"):
                opp.add_column(column, justify="right" if column != "Season" else "left")
            for row in merged.to_dicts():
                expected, actual = row["expected_points"], row["actual_points"]
                diff = (actual - expected) if (expected is not None and actual is not None) else None
                opp.add_row(
                    str(row["season"]), str(row["games"]),
                    _num(expected, 1), _num(actual, 1),
                    f"[green]+{diff:.1f}[/green]" if diff and diff > 0
                    else (f"[red]{diff:.1f}[/red]" if diff else "-"),
                    f"{row['snap_share'] * 100:.0f}%" if row.get("snap_share") is not None else "-",
                )
            console.print(opp)
            console.print(
                "[dim]Expected points come from the nflverse ff_opportunity model: what a\n"
                "player's usage was worth, independent of whether the ball bounced his way.[/dim]"
            )

        depth = queries.depth_chart_slot(db, key)
        injury = queries.latest_injury(db, key)
        notes = []
        if depth:
            notes.append(f"Depth chart: {depth['pos_abb']} #{depth['pos_rank']} ({depth['team']})")
        if injury and injury.get("report_status"):
            notes.append(
                f"Injury report: {injury['report_status']} "
                f"({injury.get('report_primary') or 'unspecified'}) "
                f"— {injury['season']} week {injury['week']}"
            )
        if notes:
            console.print(Panel("\n".join(notes), title="Role & health", border_style="dim"))


# --- board ---------------------------------------------------------------------------


def _load_board(cfg, current_pick: int | None = None, quiet: bool = False):
    """Build the scored board, surfacing any degraded signals as warnings."""
    from .analytics.board import build_board

    with connect(cfg.paths.db_path) as db:
        if db.row_count("projections") == 0:
            from .analytics.projections import refresh_derived_projections

            if db.row_count("rankings") == 0:
                err_console.print(
                    "No rankings ingested. Run [bold]ff data refresh[/bold] first."
                )
                raise typer.Exit(code=1)
            refresh_derived_projections(db, cfg)
        board = build_board(db, cfg, current_pick=current_pick)

    if board.frame.is_empty():
        for warning in board.warnings:
            err_console.print(f"[red]{warning}[/red]")
        raise typer.Exit(code=1)
    if board.warnings and not quiet:
        for warning in board.warnings:
            console.print(f"[yellow]note:[/yellow] [dim]{warning}[/dim]")
        console.print()
    return board


def _tier_style(tier: int | None) -> str:
    return "" if tier is None else ["bold cyan", "cyan", "green", "yellow"][min(tier - 1, 3)]


@app.command("board")
def board_cmd(
    ctx: typer.Context,
    position: Annotated[
        str | None, typer.Option("--position", "-p", help="Limit to one position.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Players to show.")] = 30,
    sort_by: Annotated[
        str, typer.Option("--sort", help="player|value|vbd|points|adp")
    ] = "player",
    show_replacement: Annotated[
        bool, typer.Option("--replacement", help="Show replacement-level derivation.")
    ] = False,
) -> None:
    """Show the ranked draft board."""
    from .normalization.players import normalize_position

    cfg = get_config(ctx)
    board = _load_board(cfg)

    canonical = normalize_position(position) if position else None
    sort_column = {
        "player": "player_score", "value": "value_score", "vbd": "vbd",
        "points": "projected_points", "adp": "adp",
    }.get(sort_by)
    if sort_column is None:
        err_console.print(f"[red]Unknown --sort {sort_by!r}.[/red] Use player|value|vbd|points|adp")
        raise typer.Exit(code=2)

    frame = board.frame
    if canonical:
        frame = frame.filter(pl.col("position") == canonical)
        if frame.is_empty():
            err_console.print(f"No players at position {canonical}.")
            raise typer.Exit(code=1)
    frame = frame.sort(sort_column, descending=(sort_column != "adp"), nulls_last=True).head(limit)

    title = f"{cfg.league.label} draft board"
    if canonical:
        title += f" — {canonical}"
    table = Table(title=title, header_style="bold", title_justify="left")
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Player", no_wrap=True, min_width=18)
    table.add_column("Pos", justify="center", no_wrap=True)
    table.add_column("Tm", justify="center", style="dim", no_wrap=True)
    table.add_column("Bye", justify="right", style="dim", no_wrap=True)
    table.add_column("Proj", justify="right", no_wrap=True)
    table.add_column("VBD", justify="right", no_wrap=True)
    table.add_column("Tier", justify="center", no_wrap=True)
    table.add_column("ADP", justify="right", no_wrap=True)
    table.add_column("PLYR", justify="right", no_wrap=True)
    table.add_column("VAL", justify="right", no_wrap=True)
    table.add_column("Conf", justify="right", style="dim", no_wrap=True)

    for i, row in enumerate(frame.iter_rows(named=True), start=1):
        tier = row.get("tier")
        table.add_row(
            str(i),
            row["player_name"],
            row["position"],
            row.get("team") or "",
            str(row.get("bye_week") or ""),
            _num(row.get("projected_points"), 1),
            _num(row.get("vbd"), 1),
            f"[{_tier_style(tier)}]{tier}[/{_tier_style(tier)}]" if tier else "",
            _num(row.get("adp"), 1),
            f"[bold]{row['player_score']:.1f}[/bold]",
            f"{row['value_score']:.1f}",
            f"{row.get('player_score_confidence', 0) * 100:.0f}%",
        )
    console.print(table)

    if show_replacement:
        rep = Table(title="Replacement level", header_style="bold", title_justify="left")
        rep.add_column("Pos", style="cyan")
        rep.add_column("Derivation", overflow="fold")
        for level in board.replacement.values():
            if level.players_available:
                rep.add_row(level.position, level.explanation)
        console.print(rep)

    scarcity = Table(title="Positional scarcity", header_style="bold", title_justify="left")
    scarcity.add_column("Pos", style="cyan")
    scarcity.add_column("Score", justify="right")
    scarcity.add_column("Startable left", justify="right")
    scarcity.add_column("Starter slots left", justify="right")
    scarcity.add_column("Note", overflow="fold", style="dim")
    for entry in sorted(board.scarcity.values(), key=lambda e: -e.score):
        if entry.available == 0:
            continue
        scarcity.add_row(
            entry.position, f"{entry.score:.0f}", str(entry.startable_available),
            f"{entry.remaining_demand:.0f}", "; ".join(entry.notes),
        )
    console.print(scarcity)


@app.command("compare")
def compare_cmd(
    ctx: typer.Context,
    names: Annotated[list[str], typer.Argument(help="Two or more player names.")],
) -> None:
    """Compare players side by side across every score component."""
    from . import queries

    cfg = get_config(ctx)
    if len(names) < 2:
        err_console.print("[red]Give at least two players to compare.[/red]")
        raise typer.Exit(code=2)

    board = _load_board(cfg, quiet=True)
    with connect(cfg.paths.db_path) as db:
        resolved = []
        for name in names:
            match = queries.resolve_one(db, name)
            if match is None:
                err_console.print(f"[red]Could not resolve[/red] {name!r} to one player.")
                raise typer.Exit(code=1)
            row = board.row(match["player_key"])
            if row is None:
                err_console.print(
                    f"[yellow]{match['full_name']} is not on the modelled board[/yellow] "
                    f"(position {match['position']})."
                )
                raise typer.Exit(code=1)
            resolved.append(row)

    table = Table(title="Comparison", header_style="bold", title_justify="left")
    table.add_column("Metric", style="cyan")
    for row in resolved:
        table.add_column(f"{row['player_name']}\n{row['position']} · {row.get('team') or ''}",
                         justify="right")

    def add(label: str, key: str, digits: int = 1, suffix: str = "") -> None:
        table.add_row(label, *[f"{_num(r.get(key), digits)}{suffix}" for r in resolved])

    add("Board rank", "board_rank", 0)
    add("Projected points", "projected_points")
    add("VBD", "vbd")
    add("Positional rank", "positional_rank", 0)
    add("Tier", "tier", 0)
    add("Points to next tier", "points_to_next_tier")
    add("ADP", "adp")
    table.add_section()
    add("Player Score", "player_score")
    add("Value Score", "value_score")
    table.add_section()
    add("  projection", "projection_score")
    add("  vbd", "vbd_score")
    add("  opportunity", "opportunity_score")
    add("  offense", "offense_score")
    add("  risk (inverted)", "risk_inverted_score")
    add("  market value", "market_value_score")
    add("  tier cliff", "tier_cliff_score")
    add("  scarcity", "scarcity_score")
    table.add_section()
    add("Confidence", "player_score_confidence", 2)
    console.print(table)

    best = max(resolved, key=lambda r: r["player_score"])
    margin = best["player_score"] - sorted(
        (r["player_score"] for r in resolved), reverse=True
    )[1]
    verdict = (
        f"[bold]{best['player_name']}[/bold] rates highest on season-long value "
        f"(+{margin:.1f} Player Score)."
    )
    if margin < 2:
        verdict += " That margin is inside the noise — treat them as equivalent."
    console.print(Panel(verdict, border_style="cyan"))


@app.command("explain")
def explain_cmd(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Player name.")],
) -> None:
    """Show every score component for one player, and how it was derived."""
    from . import queries
    from .scoring.player_score import player_score_bundle
    from .scoring.value_score import value_score_bundle

    cfg = get_config(ctx)
    board = _load_board(cfg, quiet=True)
    with connect(cfg.paths.db_path) as db:
        match = queries.resolve_one(db, name)
    if match is None:
        err_console.print(f"[red]Could not resolve[/red] {name!r} to one player.")
        raise typer.Exit(code=1)
    row = board.row(match["player_key"])
    if row is None:
        err_console.print(f"[yellow]{match['full_name']} is not on the modelled board.[/yellow]")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            Text.assemble(
                (row["player_name"], "bold"), "  ",
                (f"{row['position']} · {row.get('team') or ''} · board #{row['board_rank']}", "cyan"),
            ),
            border_style="cyan",
        )
    )

    for bundle in (player_score_bundle(cfg, row), value_score_bundle(cfg, row)):
        table = Table(
            title=f"{bundle.name.replace('_', ' ').title()}  =  {bundle.value:.1f}"
                  f"   (confidence {bundle.confidence * 100:.0f}%)",
            header_style="bold", title_justify="left",
        )
        table.add_column("Component", style="cyan")
        table.add_column("Raw", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Conf", justify="right")
        table.add_column("Weight", justify="right")
        table.add_column("Contributes", justify="right")
        table.add_column("Method", overflow="fold", style="dim")
        for key, comp in bundle.components.items():
            weight = bundle.weights.get(key, 0.0)
            known = comp.confidence > 0
            table.add_row(
                key,
                _num(comp.raw_value, 1),
                f"{comp.normalized:.1f}" if known else "[dim]unknown[/dim]",
                f"{comp.confidence * 100:.0f}%",
                f"{weight * 100:.1f}%" if weight else "[dim]—[/dim]",
                f"{comp.normalized * weight:.1f}" if weight else "[dim]—[/dim]",
                comp.method if known else comp.notes,
            )
        console.print(table)

    dropped = [
        key for key, comp in player_score_bundle(cfg, row).components.items()
        if comp.confidence <= 0
    ]
    if dropped:
        console.print(
            f"[dim]Unknown components ({', '.join(dropped)}) had their weight "
            f"redistributed across the rest rather than scored as average.[/dim]"
        )

    level = board.replacement.get(row["position"])
    if level:
        console.print(Panel(level.explanation, title="Replacement level", border_style="dim"))
    entry = board.scarcity.get(row["position"])
    if entry:
        console.print(Panel(entry.explanation, title="Positional scarcity", border_style="dim"))


def _num(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


__all__ = ["app", "console", "get_config", "Database"]
