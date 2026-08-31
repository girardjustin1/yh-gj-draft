# Roadmap

Status legend: ✅ done · 🔄 in progress · ⬜ not started

---

## ✅ Phase 0 — Foundation

Project structure, config system, database, CLI shell, snake math, tests.

**Built**
- `pyproject.toml` (Python 3.12, uv/hatchling), `.venv`, git repo, `.gitignore`, `.env.example`
- `config.py` — full Pydantic tree: `Paths`, `ScoringRules`, `RosterSlots`, `DraftSettings`,
  `LeagueConfig`, `PlayerScoreWeights` / `ValueScoreWeights` / `DraftNowWeights`,
  `ReplacementConfig`, `TierConfig`, `SimulationConfig`, `StrategyPriors`, `DataSourcesConfig`
- `config/league.example.yaml`, `league.yaml`, `scoring_weights.yaml`, `data_sources.yaml`
- `database.py` — 24-table DuckDB schema with lineage columns, transactional
  `replace_table`/`upsert_table`, `data_refresh_log`
- `models.py` — `PlayerIdentity`, `ComponentScore`, `ScoreBundle`, `DraftPick`,
  `RosterSnapshot`, `SurvivalEstimate`, `Candidate`, `DataFreshness`, `Recommendation`
- `draft/snake.py` — pick arithmetic incl. linear and third-round reversal, `SnakeBoard`
- `logging.py` — JSON structured logging, quiet by default
- `cli.py` — `ff doctor`, `ff version`, `ff config show|weights|validate`, `ff db init|tables|reset`

**Acceptance** — project installs ✅ · 114 tests pass ✅ · CLI responds ✅ · DuckDB
initializes (24 tables, schema v1) ✅

**Decisions**
- Weight blocks refuse to load unless they sum to 1.0 — a bad edit fails at startup, not mid-draft.
- Config models reject unknown keys (a typo in `league.yaml` is an error, not a silent no-op)
  while staying round-trippable via `StrictModel`.
- Third-round reversal implemented as "rounds 3+ invert from standard", which correctly
  moves the 2/3 turn advantage from slot 1 to slot N.

---

## 🔄 Phase 1 — NFL data

Ingest nflverse into DuckDB with lineage and freshness tracking.

- ⬜ `data/nflverse.py` adapters: players, ff_playerids, teams, schedules, rosters,
  player_stats, ff_opportunity, snap_counts, injuries, depth_charts, ff_rankings
- ⬜ `normalization/players.py` — `player_key` assignment, name normalization
- ⬜ `normalization/ids.py` — cross-platform ID map, unresolved logging
- ⬜ `normalization/teams.py` — team canonicalization, bye weeks from the 2026 schedule
- ⬜ `ff data refresh`, `ff data status`, `ff data unresolved-players`
- ⬜ Tests: identity resolution, normalization, freshness

**Acceptance** — I can query a current NFL player locally.

---

## ⬜ Phase 2 — Base fantasy engine

- Fantasy-point scoring of historical stats in *our* league's rules
- Projection adapters (CSV/Parquet/JSON import) + consensus
- Replacement level, VBD, tiers, scarcity, basic opportunity
- Player Score, Value Score
- `ff board`, `ff board --position RB`, `ff players NAME`, `ff compare A B`, `ff explain NAME`

**Acceptance** — the local board ranks players and explains every score.

---

## ⬜ Phase 3 — Sleeper

- `data/sleeper.py` — public read-only API: user, leagues, league, rosters, drafts,
  draft, picks, players
- `DraftProvider` interface + `SleeperDraftProvider`
- `draft/state.py` — canonical `DraftState`; `draft/availability.py`
- `ff sleeper connect|leagues|use-league`, `ff draft sync`, `ff draft status`

**Acceptance** — the live Sleeper board is represented accurately.

---

## ⬜ Phase 4 — Draft Now

- Roster fit, position runs, draft-room behaviour, opponent roster needs
- Next-pick survival model (interpretable, ADP-distribution based)
- Draft Now Score, `ff on-clock`

**Acceptance** — the tool recommends a *contextual* pick, not the highest-ranked
available player.

---

## ⬜ Phase 5 — Simulation

- Monte Carlo draft simulation to our next pick, seeded and reproducible
- `probability_available_next_pick`, two-pick expected value, strategy adaptation
- Simulation caching keyed on draft state; `ff simulate`

**Acceptance** — every top candidate carries a survival probability and a two-pick EV,
returned fast enough for a 30–90 second clock.

---

## ⬜ Phase 6 — Advanced data

- Defense-vs-position, position-specific schedule scores (weeks 1–4, 1–8, full, playoffs)
- Injury/risk model, offensive environment
- ADP history, trends, multi-source consensus

**Acceptance** — advanced signals improve scores without dominating them.

---

## ⬜ Phase 7 — Claude interface

- Narrow tool surface: `get_draft_state`, `get_top_recommendations`, `compare_players`,
  `simulate_until_next_pick`, `recommend_pick`, …
- Local MCP server (verify the current official SDK before depending on it)
- `/draft` skill

**Acceptance** — I say "I'm on the clock" and get a live, contextual recommendation.

---

## Later, deliberately not now

Backtesting harness · in-season start/sit, waivers, trades · ESPN/Yahoo · web UI ·
ML projections.
