---
name: draft
description: Give a live NFL fantasy snake-draft recommendation for the pick on the clock. Use when the user says "I'm on the clock", "who should I draft", "/draft", asks who to take at a pick, or asks to compare two draft candidates. Runs the local deterministic engine and interprets its output.
---

# Fantasy draft recommendation

The user is drafting and may have 30–90 seconds. Run the engine, then **interpret and
communicate** its output. Lead with the pick.

## The one rule

**Python computes; you explain.**

Never invent a projection, score, survival probability, or expected value. Every number
you state must come from the engine. If the engine did not produce one, say so and lower
your stated confidence — do not estimate it. If you disagree with a recommendation, argue
in terms of the engine's own components ("the risk component is only 36% confident here
— he has one season of data"), never from general football opinion.

## Procedure

Run from the project root; `ff` is at `.venv/bin/ff`.

```
.venv/bin/ff on-clock --json
```

That single command runs `analyze_current_pick()`, which is **the same function the web
dashboard calls**. It syncs the live draft, rebuilds `DraftState`, recomputes the
available pool, scores it, estimates survival, simulates, prices both picks, and ranks —
all in one call, typically under a second. Do not stitch this together from other
commands; if the GUI and you compute answers by different routes you will eventually
disagree mid-draft.

Add `--no-sync` if the user says the board is already current, and `--no-simulate` only
if speed is critical.

## Reading the JSON

Five areas, mirroring the dashboard:

- **`on_the_clock`** — `pick_label`, `next_pick_label`, `picks_until_next`,
  `recommendation`, `alternatives`, `confidence`, `reason`, `strategy`
- **`best_available`** — the pool, with `floor` / `median` / `ceiling` and `outcome`
  (safe / balanced / upside / floor play / volatile / boom or bust)
- **`my_roster`** — starting slots with holes visible, plus `unfilled_starters`
- **`who_makes_it_back`** — `probability_available` per player at our next pick
- **`what_if`** — `value_now` + `expected_next_value` = `combined` per candidate

Plus `draft_environment`, `simulation`, `staleness`, `warnings`, `sync_error`.

## Digging in

```
.venv/bin/ff compare-picks "Player A" "Player B"   # both picks priced head to head
.venv/bin/ff explain "Player Name"                 # every component and its derivation
.venv/bin/ff simulate                              # survival + two-pick EV in detail
```

## What to tell the user

1. **One primary recommendation**, named, first.
2. **The single fact that drove it** — usually `probability_gone` or the two-pick value.
3. **Two alternatives**, with the reason each lost.
4. **Confidence**, as the engine reported it.
5. **Stale data or warnings**, explicitly.

Fifteen seconds to read. Offer detail rather than front-loading it.

Shape to aim for:

> **TAKE KYREN WILLIAMS.** Draft Now 74.2, confidence 50%.
> He is 57% likely to be gone before your next pick at 5.07, 12 picks away; Joe Burrow
> has a 64% chance of surviving it, so taking Kyren now is the only way to have both.
> Across both picks: Kyren 178.4 vs Burrow 176.9.
> Alternatives: Javonte Williams (72.6), Joe Burrow (72.2).

## What matters most

- **`probability_gone`** is what separates the right pick from the highest-ranked one. A
  player 90% gone and one 20% gone are different decisions at identical scores.
- **`what_if` / `two_pick_expected_value`** prices *both* picks. When it disagrees with
  the Draft Now ordering the engine says so in `reason` — pass that reasoning on.
- **`confidence` below ~45%** means the candidates are genuinely bunched. Say so; offer
  the alternatives as real options rather than projecting false certainty.
- **`warnings`** are not boilerplate. Kickers and defences are unmodelled; schedule
  strength is deliberately low-confidence. Surface anything bearing on this pick.
- **`sync_error`** means the board may have moved. Say it plainly.

## Never

- Recommend from generic positional strategy ("take a RB early") when live board data
  exists — the engine already knows the room.
- Restate engine numbers with different values.
- Run `ff data refresh` mid-draft; it is slow, and the draft board syncs separately.

## If it cannot answer

| Message | Fix |
|---|---|
| `NoDraftError` | `ff draft sync` (Sleeper), `ff draft start` (manual entry), or `ff draft mock` to practise |
| `UnknownSlotError` | `ff draft sync` once the order is set, or set `draft.slot` in `config/league.yaml` |
| "No rankings ingested" | `ff data refresh` — before draft day, not during |
| Yahoo not implemented | Yahoo needs approved OAuth credentials; use `ff draft start` + `ff draft pick` to enter picks by hand |
