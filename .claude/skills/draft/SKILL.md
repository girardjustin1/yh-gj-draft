---
name: draft
description: Give a live NFL fantasy draft recommendation for the pick on the clock. Use when the user says "I'm on the clock", "who should I draft", "/draft", asks who to take at a pick, or asks to compare draft candidates. Runs the local deterministic engine and interprets its output.
---

# Fantasy draft recommendation

The user is drafting. Your job is to run the local engine, then **interpret and
communicate** its output.

## The one rule

**Python computes; you explain.**

Never invent a projection, score, survival probability, or expected value. Every number
you state must come from the engine's output. If the engine did not produce a number,
say that it did not, and lower your stated confidence — do not estimate it yourself. If
you disagree with a recommendation, say why *in terms of the engine's own components*
(for example, "the risk component is only 36% confident here because he has one season
of data"), not from general football opinion.

## Procedure

Run these from the project root. The `ff` command is at `.venv/bin/ff`.

1. **Sync the live board first.** Stale board data is the single biggest source of a
   wrong recommendation.
   ```
   .venv/bin/ff draft sync
   ```
   If this fails, continue — the engine falls back to the last synced state and labels
   it stale. Report that staleness to the user.

2. **Get the recommendation as JSON.**
   ```
   .venv/bin/ff on-clock --json
   ```
   If the user wants the full formatted table, run `.venv/bin/ff on-clock` instead and
   let its output stand; do not re-render it yourself.

3. **Read the JSON.** It contains: `pick`, `next_pick`, `picks_until_next`, `strategy`,
   `confidence`, `position_demand`, `simulation`, `recommendation`, `alternatives`,
   `board`, `warnings`, and a written `explanation`.

4. **For a close call, look deeper.**
   ```
   .venv/bin/ff explain "Player Name"      # every component and how it was derived
   .venv/bin/ff compare "Player A" "Player B"
   .venv/bin/ff simulate                    # survival and two-pick EV in detail
   ```

## What to tell the user

Lead with the pick. They are on a clock.

1. **One primary recommendation**, named.
2. **The single fact that drove it** — usually the survival probability or the
   two-pick expected value.
3. **Two alternatives**, with the reason each lost.
4. **Confidence**, as the engine reported it.
5. **Any stale data or warnings**, explicitly.

Keep it short enough to read in fifteen seconds. Detail is available on request.

## What matters in the output

- `probability_gone` is the number that most often separates the right pick from the
  highest-ranked one. A player 90% gone and a player 20% gone are different decisions
  even at identical scores.
- `two_pick_ev` prices *both* picks — this player plus the simulated best available at
  the next turn. When it disagrees with the Draft Now ordering, the engine says so in
  `explanation`; pass that reasoning on.
- `confidence` below about 45% means the top candidates are genuinely bunched. Say so
  rather than projecting false certainty; offer the alternatives as real options.
- `warnings` are not boilerplate. Kickers and defences are not modelled; schedule
  strength is deliberately low-confidence. Surface anything that bears on this pick.

## Never do

- Do not recommend from generic positional strategy ("take a RB early") when live board
  information exists. The engine already knows the room.
- Do not restate the engine's numbers with different values.
- Do not run `ff data refresh` mid-draft; it is slow and the draft board is synced
  separately.

## If the engine cannot answer

| Symptom | Fix |
|---|---|
| "No draft available" | `ff draft sync`, or `ff draft mock` to practise |
| "Your draft slot is unknown" | `ff draft sync` once the order is set, or set `draft.slot` in `config/league.yaml` |
| "No rankings ingested" | `ff data refresh` (do this before draft day, not during) |
| Sleeper unreachable | The last synced board is used and labelled stale — say so |
