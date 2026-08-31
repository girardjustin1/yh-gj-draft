# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

A local-first NFL fantasy snake-draft decision engine. The deliverable is a *decision*,
not a ranking: given the live board, my roster, and my next snake pick, who do I take?

## The rule that matters most

**Python computes; Claude explains.**

Never invent a fantasy score, projection, survival probability, or expected value in
prose. If a number is needed, it comes from the engine. If the engine cannot produce it,
say so and lower the stated confidence — do not estimate it yourself.

## Working agreements

- **Verify APIs against the installed version.** Do not assume a `nflreadpy`,
  `duckdb`, or MCP SDK signature; inspect it. Schemas from vendors change.
- **Weights live in YAML.** `config/scoring_weights.yaml`. A magic number in a `.py`
  file is a bug. Weight blocks must sum to 1.0 or the app refuses to start.
- **Never claim something works without running it.** Run the tests, run the command,
  paste the real output.
- **Missing data lowers confidence.** It never becomes a neutral default presented with
  certainty.
- **Ambiguous player matches are logged, never merged.** See `unresolved_players`.
- **Nothing leaves the machine.** No telemetry, no uploads, no third-party calls beyond
  fetching public NFL data. Never commit `config/league.yaml`, `.env`, or `data/`.

## Layout

`config.py` config models · `database.py` DuckDB · `data/` vendor adapters ·
`normalization/` identity · `analytics/` football math · `draft/` pick math + simulation ·
`scoring/` composites · `recommendation/` ranking + explanation · `cli.py` the interface.

Dependencies flow downward only. `analytics/` must not import from `data/`.

## Commands

```bash
uv pip install -e ".[dev]"      # install
.venv/bin/python -m pytest      # tests
.venv/bin/ff doctor             # health check
.venv/bin/ruff check src tests  # lint
```

## Conventions

- Overall picks and rounds are **1-indexed**. Draft slots are 1..teams, left to right in
  round 1. `draft/snake.py` is the only place that computes pick arithmetic.
- Bulk numeric work uses Polars. Pydantic models are for boundaries and serialization.
- Every scoring component returns `raw_value`, `normalized` (0–100), and `confidence`.

## When you need something from the human

Add it to `HUMAN_TODO.md` under REQUIRED / LATER / OPTIONAL and keep building with a
sensible default or fixture. Do not block.
