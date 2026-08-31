# Architecture

## Pipeline

```
  nflverse   FantasyPros   Sleeper   your CSVs
      │           │           │          │
      └───────────┴─────┬─────┴──────────┘
                        │            src/fantasy_draft/data/*
                   INGESTION           one adapter per source, independent,
                        │              failure-isolated
                        ▼
                  NORMALIZATION        src/fantasy_draft/normalization/*
                        │              vendor rows → player_key, canonical teams
                        ▼
                     DUCKDB            data/fantasy.duckdb
                        │              every row keeps source + timestamps
                        ▼
                   ANALYTICS           src/fantasy_draft/analytics/*
                        │              projections, replacement, VBD, tiers,
                        │              scarcity, opportunity, schedule, risk, market
                        ▼
                    SCORING            src/fantasy_draft/scoring/*
                        │              Player Score, Value Score, Draft Now Score
                        ▼
          DRAFT STATE ─────► SIMULATOR src/fantasy_draft/draft/*
                        │              snake math, availability, opponent needs,
                        │              Monte Carlo survival, two-pick EV
                        ▼
                 RECOMMENDATION        src/fantasy_draft/recommendation/*
                        │              ranked candidates + explanation + confidence
                        ▼
             analyze_current_pick()    src/fantasy_draft/service.py
                        │              THE single orchestration entry point
            ┌───────────┼───────────┐
            ▼           ▼           ▼
          CLI      JSON API      CLAUDE
       (cli.py)  (api/service)   (skill)
                        │
                        ▼
                  DASHBOARD            api/static/index.html — renders only
```

## Module responsibilities

| Module | Owns | Must not |
|---|---|---|
| `data/` | Fetching, caching, and raw-shape validation per vendor | Contain scoring logic |
| `normalization/` | `player_key` assignment, team canonicalization, name matching | Silently merge ambiguous players |
| `analytics/` | Football math on normalized tables | Know about Sleeper/ESPN/vendors |
| `draft/` | Pick arithmetic, board state, opponent modelling, simulation | Rank players by quality |
| `scoring/` | Combining components into 0–100 composites using YAML weights | Hard-code a weight |
| `recommendation/` | Ordering candidates and writing the explanation | Recompute analytics |
| `service.py` | Orchestrating a pick end to end | Contain football maths |
| `cli.py` | Human interface | Contain business logic worth testing |
| `api/` | HTTP transport and rendering | Compute *anything* |

The dependency direction is strictly downward. `analytics/` importing from `data/` is a
bug; ingestion writes to DuckDB and analytics reads from it.

## Data lineage

Every ingested table carries `source`, `source_updated_at`, and `ingested_at`.
Normalization *adds* a `player_key`; it never overwrites vendor identifiers. This is
what makes `ff data status` honest and backtesting possible later.

`data_refresh_log` records every refresh attempt — success or failure — with row counts
and duration, so staleness is a queryable fact rather than a guess.

## Player identity

`player_key` is the canonical internal ID. Resolution order:

1. `gsis_id` (nflverse canonical) — preferred, stable across seasons.
2. Platform ID via the ffverse/DynastyProcess map (`load_ff_playerids`): Sleeper, ESPN,
   Yahoo, FantasyPros, PFR, MFL, Sportradar.
3. Normalized name + position + team, as a last resort.

Normalization strips suffixes, punctuation, and case, and applies a small explicit
nickname table. **Ambiguous matches are never merged** — they land in
`unresolved_players` and surface via `ff data unresolved-players`.

## One entry point, three consumers

`analyze_current_pick()` in `service.py` is the only route to a recommendation. The CLI,
the JSON API and the Claude skill all call it and differ solely in how they render the
result.

This is deliberate rather than tidy. If the dashboard and the assistant computed answers
by different routes they would eventually disagree mid-draft, and two confident,
conflicting recommendations on a 90-second clock are worse than one imperfect answer.
The page contains no scoring logic at all — it reads keys off a JSON payload.

## Failure isolation

One source failing must degrade a component's confidence, not crash a recommendation:

| Failure | Behaviour |
|---|---|
| Schedule data missing | Schedule component confidence → 0, weight redistributed |
| ADP unavailable | Value Score falls back to projection/VBD; market confidence flagged |
| Sleeper unreachable | Last synced draft state is used and **labelled stale** |
| Injury data stale | Risk confidence lowered; never treated as "healthy" |
| Player unresolvable | Excluded from the board, logged, counted in `ff doctor` |

Missing information lowers confidence. It is never rounded to certainty.

## Score composition

A `ScoreBundle` holds the composite value plus every `ComponentScore` that produced it
(`raw_value`, `normalized`, `confidence`, `method`, `source`, `source_updated_at`). This
is what lets `ff explain PLAYER` show the full derivation and what gets persisted to
`player_scores.components` for future backtesting.

When a component has zero confidence, its weight is redistributed across the known
components rather than being scored as a neutral 50 — otherwise unknown data would drag
every player toward the mean and quietly compress the board.

## Extension points

- **New platform**: implement the `DraftProvider` interface in `draft/`. Sleeper is the
  reference implementation; ESPN and Yahoo can follow without touching scoring.
- **New projection source**: drop a CSV through `ff import projections`. Sources are
  stored separately and combined; none overwrites another.
- **New score component**: add it to `analytics/`, register a weight in
  `scoring_weights.yaml`, and it participates automatically — the weight blocks
  validate that everything still sums to 1.0.

## In-season future

The schema is intentionally wider than the draft needs (weekly stats, snap counts,
injuries, defense-vs-position, per-week projections) so start/sit, waivers, and trade
evaluation can be added later without a migration.
