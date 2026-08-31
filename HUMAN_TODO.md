# Human TODO

Things the engine needs from you. Nothing here blocks development — defaults and
fixtures keep everything running until you fill these in.

Last updated: 2026-08-31 (end of Phase 0)

---

## REQUIRED — before `ff on-clock` can be trusted

These are the settings that drive replacement level, VBD, scarcity, and roster fit. If
they are wrong, every number downstream is wrong.

- [ ] **Copy the example config**: `cp config/league.example.yaml config/league.yaml`
      (`config/league.yaml` is gitignored, so your league details stay local.)
- [ ] **Number of teams** — currently `12`
- [ ] **Scoring format** — currently half-PPR (`scoring.reception: 0.5`).
      Set `0.0` for standard, `1.0` for full PPR.
- [ ] **Passing TD value** — currently `4`. Some leagues use `6`.
- [ ] **Starting roster requirements** — currently 1QB / 2RB / 2WR / 1TE / 1FLEX / 1K /
      1DST / 6 bench. **Set `superflex: 1` if this is a superflex or 2QB league** — it
      changes QB valuation more than any other single setting.
- [ ] **Draft rounds** — currently `15`. Must equal starters + bench.
- [ ] **Your draft slot** — currently unset. Required for snake pick math.
      `ff draft sync` can fill this in automatically once Sleeper is connected (Phase 3).

Check your work with `ff config show` and `ff doctor`.

## LATER — needed for live draft sync (Phase 3)

- [ ] **Sleeper username** — then run `ff sleeper connect <username>`
- [ ] **Select your league** — `ff sleeper leagues`, then `ff sleeper use-league <id>`
- [ ] **Confirm the draft ID** — `ff draft sync` will detect it; verify it matches the
      draft you are actually in if the league has multiple.
- [ ] **Confirm third-round reversal** — if your Sleeper league uses a reversal round,
      set `draft.third_round_reversal: true`. This changes which picks are yours.

## OPTIONAL — improves accuracy, not required

- [ ] **Import your own projections** — `ff import projections ./my-projections.csv`.
      Any CSV/Parquet/JSON with a name column plus either `fantasy_points` or the
      component stat columns. Multiple sources are kept separately and combined into a
      consensus; nothing is overwritten.
- [ ] **Import an additional ADP source** — `ff import adp ./adp.csv`. More independent
      ADP sources tighten the survival model. FantasyPros ECR is already built in.
- [ ] **Tell me your keepers**, if any — they change the available pool and every
      opponent's roster needs.
- [ ] **Tune scoring weights** — `config/scoring_weights.yaml`. The defaults are
      defensible starting points, not settled truth. `ff config weights` shows them.

## NOT NEEDED — do not send these

- ❌ Passwords for any platform. Sleeper's public read API needs none.
- ❌ Session cookies or auth tokens.
- ❌ ESPN/Yahoo credentials. Those platforms are not implemented and won't be for MVP.

Nothing in this project transmits your league data anywhere. There is no telemetry.
