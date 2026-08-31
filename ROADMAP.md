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

## ✅ Phase 1 — NFL data

Ingest nflverse into DuckDB with lineage and freshness tracking.

**Built**
- `data/nflverse.py` — 11 dataset adapters, each failure-isolated and defensive about
  vendor schema changes (`pick()` degrades a renamed column to null instead of raising)
- `normalization/players.py` — name normal form: accent folding, punctuation stripping,
  suffix removal, initial-run joining
- `normalization/ids.py` — `IdentityMap`: gsis_id → platform ID → name+position, with
  ambiguity reported rather than merged
- `normalization/teams.py` — team canonicalization, bye weeks derived from the schedule
- `queries.py` — shared read layer for CLI/API/MCP
- `ff data refresh|status|sources|unresolved-players`, `ff players NAME`

**Acceptance** ✅ — a full refresh loads **292,552 rows in ~7s**:
players 27,262 · rankings 1,849 (2026 ECR scraped 2026-08-28) · player_stats 75,879
(2022-2025) · snap_counts 106,148 · opportunity 24,178 · injuries 23,564 ·
schedules 1,411 · depth_charts 2,635 · all 32 2026 bye weeks derived.
`ff players "Bijan Robinson"` returns identity, ECR spread, usage, expected-vs-actual
points, and depth-chart slot.

**Findings**
- **515/517** of the FantasyPros overall board joins to a canonical player. The 6
  unresolved are ECR 250+ camp bodies, all listed in `ff data unresolved-players`.
- The speculative nickname table was **deleted**: measured against the live board, the
  general normalization rule matches 100% of the top-300 non-DST names with no lookups.
  Every entry in such a table asserts two feeds mean the same person, so entries are now
  added only when a real unmatched name proves one is needed.
- `normalize_position` preserves IDP positions (LB/DL/DB) because they disambiguate
  identity; draftability is filtered at ingest via `DRAFTABLE_POSITIONS`.

---

## ✅ Phase 2 — Base fantasy engine

**Built**
- `analytics/fantasy_points.py` — every point recomputed in *our* scoring rules, never
  borrowed from a vendor's PPR column
- `analytics/projections.py` — the **historical positional value curve**: score past
  seasons in our rules, learn what each positional finish is worth, map the consensus
  board's ordering onto it. Plus CSV/Parquet/JSON import with column-alias detection and
  component-stat scoring
- `analytics/replacement.py` · `vbd.py` · `tiers.py` · `scarcity.py` · `opportunity.py` ·
  `offense.py` · `risk.py` · `market.py` · `board.py`
- `scoring/compose.py` — confidence-weighted composition; `normalize.py`;
  `player_score.py`; `value_score.py`
- `ff board [--position --sort --replacement]`, `ff compare A B`, `ff explain NAME`,
  `ff import projections|list`

**Acceptance** ✅ — the board ranks 775 players and explains every component.
Verified against 2025: CMC 365.6 and Josh Allen 364.6 half-PPR points.
RB replacement lands at **RB41**, not RB24 — 28 league-wide starting slots × 1.55 for
bench hoarding — which is exactly the point the spec makes.

**Findings**
- **A global gap threshold cannot tier a position.** Projection gaps decay monotonically
  (RB gaps run 17, 13, 14, 9 at the top and a flat 4 by RB20), so one threshold fired on
  each of the top 8 backs — making each his own tier — then never fired again, dumping
  130 into one. Gaps are now standardized against their local neighbourhood, which
  produces football-sensible tiers: Gibbs/Bijan/CMC as tier 1, then a real 11-player
  plateau at RB7-17.
- **Raw projected points put six QBs above every RB**, confirming VBD is not optional
  bookkeeping but the thing that makes the board usable.
- Confidence varies as designed: a true rookie with no snaps scores 0.70 Player Score
  confidence against 0.85 for veterans, and his opportunity weight is redistributed
  rather than faked at 50.
- K and DST are **not modelled** and say so on every board: nflverse carries no
  team-defence scoring and we do not ingest kicking stats. Their value over replacement
  is near zero regardless.

---

## ✅ Phase 3 — Sleeper

**Built**
- `data/sleeper.py` — cached public read-only client; `infer_scoring` / `infer_roster` /
  `infer_draft_settings` translate a Sleeper league into our config
- `draft/providers.py` — `DraftProvider` protocol, `SleeperDraftProvider`,
  `FixtureDraftProvider`, and `PlayerKeyResolver` (ffverse map → Sleeper's own gsis_id →
  DST team code → unique name+position)
- `draft/state.py` — canonical `DraftState`; `draft/store.py` — DuckDB persistence;
  `draft/fixtures.py` — deterministic synthetic drafts
- `ff sleeper connect|leagues|use-league|status`, `ff draft sync|status`

**Acceptance** ✅ — verified end to end against the fixture: a 12-team half-PPR snake at
slot 7, 40 picks made, correctly reports on the clock 4.05, our next pick 4.06, the one
after at 5.07, 12 intervening managers, our reconstructed roster, and the last-12-pick
position mix. Round-trips through DuckDB unchanged.

**Findings**
- **Sleeper answers an unknown username with HTTP 200 and a body of `null`**, not a 404.
  Treating a 200 as success would silently "connect" you to nothing, so the client
  raises `SleeperNotFound` on a null body.
- `/players/nfl` is a **14.6 MB** document that Sleeper asks be fetched at most daily; it
  is cached on disk so a live draft never pays for it. It carries `gsis_id` directly,
  which makes pick resolution exact rather than name-based.
- `ff sleeper use-league` derives scoring, roster slots and draft format from the
  platform, so replacement level reflects the real league. It confirms before
  overwriting `config/league.yaml`, and calls out superflex explicitly.
- A pick we cannot resolve is still recorded as off the board and counted in
  `unresolved_pick_count`, so a recommendation can say the board is imperfect rather
  than quietly treating that player as available.

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
