# Fantasy Draft AI

A **local-first** 2026 NFL fantasy-football snake-draft decision engine.

Not a rankings viewer. It answers one question well:

> *I am on the clock. Given who's gone, my roster, the other rosters, my next snake pick,
> ADP, projections, scarcity, opportunity, schedule, risk, and the probability players
> survive until my next pick — who should I draft?*

Deterministic Python does the math. Claude explains it.

## Quick start

```bash
uv venv --python 3.12
uv pip install -e ".[dev,web]"
source .venv/bin/activate

ff doctor            # validate environment + data
ff data refresh      # ~290k rows in ~8s. Do this before draft day, not during.
ff draft mock        # a practice draft over the real board
ff serve             # dashboard at http://127.0.0.1:8000
```

See [ROADMAP.md](ROADMAP.md) for build phases and [HUMAN_TODO.md](HUMAN_TODO.md) for
anything the engine needs from you.

## The thing it does

```
$ ff on-clock

──────────────────────── ON THE CLOCK ────────────────────────
League        Home League — 12-team Half-PPR
Pick          4.06 (overall 42)
Next pick     5.07 (overall 55) — 12 picks away

Current roster:  WR Amon-Ra St. Brown · WR DeVonta Smith · RB Kenneth Walker III

 Pos   Demand              Run   Value falling to us   Expected gone before our pick
 QB       27  ███░░░░░░░░░   0                    70                             1.3
 RB       76  █████████░░░   7                    55                             4.3
 WR       75  █████████░░░   7                    56                             5.8
 TE       22  ███░░░░░░░░░   0                   100                             0.6

 #  Player              Pos  Tier  Draft Now  Player  Value    ADP  Gone by next  2-pick EV
 1  Kyren Williams      RB     1        74.2    91.5   77.5   41.5          57%      180.6
 2  Javonte Williams    RB     1        72.9    90.1   81.7   44.6          46%      179.5
 3  Joe Burrow          QB     1        72.4    89.7   71.8   45.8          36%      179.0

──────────────────── TAKE KYREN WILLIAMS ────────────────────

Take Kyren Williams (RB, LAR). He projects 207 points, 101 above RB replacement.
He is more likely than not gone before our next pick (57%). Across both picks,
simulated expected value: Kyren Williams 180.6; Javonte Williams 179.5.
Recommendation confidence: 43%.
```

## Commands

**Setup**

| Command | Description |
|---|---|
| `ff doctor` | Validate environment, config, database, data freshness |
| `ff config show` / `weights` / `validate` | Inspect league settings and scoring weights |
| `ff data refresh` | Pull nflverse/FantasyPros data (~290k rows in ~8s) |
| `ff data status` / `sources` / `unresolved-players` | Freshness, sources, identity failures |
| `ff db init` / `tables` / `reset` | Database maintenance |

**Analysis**

| Command | Description |
|---|---|
| `ff board [--position RB] [--sort value] [--replacement]` | The ranked draft board |
| `ff players NAME` | Identity, ECR spread, usage, expected-vs-actual points, depth chart |
| `ff compare A B [C…]` | Side-by-side across every score component |
| `ff explain NAME` | Every component, its raw value, weight, confidence, and method |
| `ff import projections FILE` / `ff import list` | Bring your own CSV/Parquet/JSON |

**Drafting**

| Command | Description |
|---|---|
| `ff sleeper connect USERNAME` / `leagues` / `use-league ID` / `status` | Link Sleeper (public read-only) |
| `ff draft sync` / `status [--rosters]` | Sync and inspect the live board |
| `ff draft start` / `pick NAME` / `undo` | Enter a draft by hand (any platform) |
| `ff draft mock [--picks N --slot N]` | Practice draft over the real board |
| **`ff on-clock [--sync] [--json]`** | **The recommendation, in five areas** |
| `ff compare-picks A B` | "What if I take A instead of B?" |
| `ff simulate [--iterations N]` | Survival + two-pick EV in detail |
| `ff serve [--port N]` | The local dashboard |

Add `--verbose` or `--debug` to any command for structured logs.

## The dashboard

`ff serve` opens a local page at **http://127.0.0.1:8000** with one primary action —
**I'M ON THE CLOCK** (or press `C`) — and five areas:

1. **On the clock** — the pick, the next pick, the recommendation, and why
2. **Best available** — the pool, filterable by position and sortable by any column
3. **My roster** — actual starting slots, with the holes visible
4. **Who makes it back to me?** — survival probability at your next pick
5. **What if I take…** — each candidate priced across *both* picks

One button press calls `analyze_current_pick()` once; the page only renders. No build
step, no framework, no external requests — it binds to localhost and nothing leaves the
machine.

## With Claude

A project skill lives at `.claude/skills/draft/SKILL.md`. In Claude Code, say:

> I'm on the clock.

Claude runs `ff on-clock --json` — **the same `analyze_current_pick()` the dashboard
calls** — and interprets the result. It is instructed never to invent a number the engine
can compute. The GUI, the CLI and Claude cannot disagree, because there is only one
code path.

## How it decides

Three scores, not one opaque number — all weights in `config/scoring_weights.yaml`:

- **Player Score** — how good is he? projection, VBD, opportunity, offensive
  environment, schedule, risk
- **Value Score** — what does taking him *here* capture? market/ADP, tier cliff,
  scarcity, projection-vs-market
- **Draft Now Score** — what do I take *at this pick*? the two above plus next-pick
  urgency, tier scarcity, roster fit, draft-room behaviour, strategy fit

Plus, for every player, a **floor / median / ceiling** range and a shape label (safe,
upside, floor play, boom or bust), so a 210-point certainty is distinguishable from a
120-or-300 coin flip.

## Draft platforms

| Platform | Status | What it needs |
|---|---|---|
| **Sleeper** | ✅ working | Your **public username** only — no password, no token |
| **Manual entry** | ✅ working | Nothing. `ff draft start` then `ff draft pick "Name"` |
| **Yahoo** | ⚠️ stub | An approved Yahoo developer application (manual review, read-only). Raises with instructions rather than failing confusingly. |

All three normalize into the same `DraftState`; the recommendation engine never learns
which one supplied it.

Every component reports a **confidence**. When one cannot be computed, its weight is
redistributed across the components we do have — never scored as a confident-looking
average. `ff explain` shows this happening.

See [docs/scoring.md](docs/scoring.md) for the methodology and its limitations.

## Layout

```
config/     league + weights + data-source YAML
data/       raw / processed / cache / fantasy.duckdb   (gitignored)
src/fantasy_draft/
  config.py      Pydantic league + weights models
  database.py    DuckDB connection + schema
  data/          source adapters (nflverse, sleeper, adp, projections)
  analytics/     vbd, tiers, scarcity, opportunity, schedule, risk, market
  draft/         snake math, draft state, availability, simulator
  scoring/       player_score, value_score, draft_now
  recommendation/ ranker + explanation
tests/
```

## Privacy

Everything runs on your machine. No telemetry, no uploads, no third-party transmission
beyond fetching public NFL/fantasy data. Secrets, caches, and the DuckDB file are
gitignored.
