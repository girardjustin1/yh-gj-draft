# Implementation Plan

## The one thing this system must do

During the 2026 draft, on the clock, answer **"who should I take right now?"** — using
this league's rules, this board's state, and the probability each candidate survives to
my next snake pick. Deterministic Python computes it; Claude explains it.

## Non-negotiable architecture

```
DATA SOURCES → INGESTION → NORMALIZATION → DUCKDB → ANALYTICS
             → SCORING → SIMULATION → RECOMMENDATION → CLI / MCP → CLAUDE
```

Three separations are load-bearing:

1. **Data vs. calculation vs. interpretation.** Claude never invents a number the
   engine can compute. Every score the CLI prints came out of tested Python.
2. **Scoring logic vs. vendor.** Adapters normalize into our schema; nothing in
   `analytics/` or `scoring/` knows what a Sleeper ID looks like.
3. **Weights vs. code.** Every tunable lives in `config/*.yaml`. The loader refuses to
   start if a weight block doesn't sum to 1.0.

## Three scores, not one magic number

| Score | Question | Inputs |
|---|---|---|
| **Player Score** | How good is he, ignoring the draft? | projection, VBD, opportunity, offense, schedule, risk |
| **Value Score** | What does taking him *here* capture? | market/ADP, tier cliff, scarcity, projection-vs-market |
| **Draft Now Score** | What do I take *at this pick*? | Player, Value, next-pick urgency, tier scarcity, roster fit, draft room, strategy |

Each component returns `raw_value`, `normalized` (0–100), and a **confidence**, so
"average" is distinguishable from "we don't know". Missing data lowers confidence; it
never silently becomes a neutral 50 with full certainty.

## Key modelling decisions

- **Replacement level is estimated, not hard-coded.** Starter demand (dedicated slots +
  a share of flex, × teams) scaled by a per-position bench-hoarding multiplier, blended
  with a fixed-rank anchor. RB replacement lands well below RB24 because leagues roster
  ~1.55× the RBs they start. Configurable in `scoring_weights.yaml`.
- **Survival probability drives urgency.** ADP alone is not enough: we combine the ADP
  distribution (FantasyPros ECR ships `sd`/`best`/`worst`) with the specific rosters and
  needs of the managers picking before our next turn.
- **Two-pick expected value beats single-pick maximization.** Take the player whose
  `value(now) + E[best available at our next pick]` is highest, not the highest raw score.
- **Strategy is descriptive, not prescriptive.** Balanced / Hero RB / Robust RB / Zero RB
  carry soft probabilities that shift with what actually falls. Never hard-coded.

## Phase order

Phase 0 foundation → 1 nflverse data → 2 base fantasy engine (VBD/tiers/scarcity/board)
→ 3 Sleeper live draft → 4 Draft Now → 5 Monte Carlo + two-pick EV → 6 advanced signals
→ 7 Claude/MCP interface. Detail and status live in [ROADMAP.md](ROADMAP.md).

## Explicitly out of scope for the MVP

Frontend, ML projections, ESPN/Yahoo, cloud anything, auth, MCP-before-CLI. Backtesting
is not built, but nothing in the schema or scoring API forecloses it: every score is
stored with its components, its confidence, and a timestamp.

## Verified environment facts (checked against installed versions, not assumed)

- Python 3.12.13 via `uv`; polars 1.44.1, duckdb 1.5.5, nflreadpy 0.1.5.
- `nflreadpy.load_ff_rankings(type="draft")` returns live 2026 FantasyPros ECR with
  `ecr`, `sd`, `best`, `worst` — the backbone of the ADP-uncertainty model.
- `nflreadpy.load_schedules([2026])` returns all 272 games of the 2026 season, so bye
  weeks and schedule strength are computable for the upcoming season.
- `nflreadpy.load_ff_opportunity()` provides expected fantasy points — the core
  Opportunity Score input.
- `get_current_season()` returns 2025 (last completed season) until the 2026 opener;
  `get_current_season(roster=True)` returns 2026. Ingestion uses each appropriately.
