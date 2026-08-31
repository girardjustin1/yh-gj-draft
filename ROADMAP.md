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

## ✅ Phase 4 — Draft Now

**Built**
- `draft/opponent_needs.py` — per-pick positional intent for every manager ahead of our
  next turn. The market prior is read off the consensus board's own ADP mix rather than
  hardcoded, then modulated by unfilled starting slots. No personal profiling.
- `draft/availability.py` — the roster-aware survival model, plus the ADP-only baseline
  for comparison
- `analytics/draft_room.py` — demand, run intensity, and **value created at the
  positions the room is skipping**
- `analytics/roster_fit.py` · `draft/strategies.py` (Balanced / Hero RB / Robust RB /
  Zero RB as soft, shifting probabilities)
- `scoring/draft_now.py` · `recommendation/ranker.py` · `recommendation/explanation.py`
- `ff on-clock [--sync --json]`, `ff draft mock` (practice over the real board)

**Acceptance** ✅ — at 4.06 with 1 RB / 2 WR rostered, it recommends Kyren Williams
(Player Score 91.5, **53% gone before our next pick**) over safer, higher-ADP-value
options, and explains the trade in the spec's own terms.

**Four real bugs, all found by looking at output that was obviously wrong**
1. **Replacement level was computed on the shrinking available pool**, so every VBD
   inflated as the draft progressed — and inflated *unevenly by position* (+37 phantom
   points to WR against +7 to TE), corrupting the one comparison VBD exists to make.
   Replacement is now derived from the full universe and applied to what is left.
2. **Unknown survival was rewarding players.** The model only covered the top 120 by
   Player Score; everyone else got a null urgency, whose weight the composition step
   then redistributed — so a player 90 picks past his ADP carried *no* urgency penalty
   and topped the board. Every player now gets a probability, falling back to the ADP
   curve at lower confidence.
3. **Market value and urgency double-counted with opposite signs**, very nearly
   cancelling, so the engine recommended players it simultaneously reported an 81%
   chance of still being available. The discount is now capped at the horizon we can act
   over and weighted by the probability we would actually lose him: *a discount you can
   capture at your next pick is not a discount*.
4. **Every player in a tier inherited the cliff at the tier's bottom edge.** The first of
   fourteen available tier-3 QBs scored a maximum cliff when the next QB was 10 points
   away. Replaced with the quantity it was approximating — *if I skip him, who do I
   actually get instead?* — sliding down his position by the number expected to go
   before our next turn. Tier cliffs fall out of it naturally and it is comparable
   across positions because it is measured in points.

---

## ✅ Phase 5 — Simulation

**Built**
- `draft/simulator.py` — vectorized NumPy Monte Carlo of the picks between our turn and
  our next, seeded and reproducible; two-pick expected value with an 80% interval and
  the likely position of our next selection; `blend_survival`
- `ff simulate`, and `ff on-clock --simulate/--no-simulate --iterations`

**Acceptance** ✅ — **0.17s for the full recommendation with 2,000 simulations**, 0.43s
at 10,000, against a 30-90 second clock. Two-pick EV is stable across iteration counts
(180.57 at 2k vs 180.61 at 10k), so 2,000 is enough.

**The simulation cross-validates the analytic model.** Built from the same inputs but
making different approximations, they agree within 1-5 points across the top candidates,
and the analytic model is consistently the *optimistic* one — exactly the direction its
independence assumption predicts, since it cannot see that the managers ahead of us
compete for the same players. Both figures are reported side by side in `ff simulate`;
where they diverge sharply, that is worth knowing.

**One approximation, stated rather than hidden.** The intervening picks are simulated
once over the full pool, and each candidate's next-pick value is computed by excluding
him from those same survivors. Simulating separately per candidate would multiply the
cost by the candidate count for a second-order correction — removing one player from a
pool of hundreds barely moves the other managers. The note is printed with the output.

Two-pick EV overrides the Draft Now ranking only on a margin larger than simulation
noise (1.5 points), and says so in the explanation when it does.

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
