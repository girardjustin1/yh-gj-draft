# Human TODO

Things the engine needs from you. Nothing here blocks development — defaults and
fixtures keep everything running until you fill these in.

Last updated: 2026-08-31 (end of Phase 6)

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

## FASTEST PATH — if your league is on Sleeper, do this instead

`ff sleeper use-league` reads your scoring rules, roster slots and draft format straight
off the platform and writes them into `config/league.yaml`, which settles almost
everything in the REQUIRED list above:

```bash
ff sleeper connect <your-sleeper-username>
ff sleeper leagues
ff sleeper use-league <league_id>     # confirms before overwriting league.yaml
ff config show                        # check it looks right
ff draft sync                         # once the draft order is set, fills in your slot
```

- [ ] **Sleeper username** — it is the username, not the display name
- [ ] **Select your league**
- [ ] **Confirm the draft ID** if the league has more than one draft
- [ ] **Confirm third-round reversal** — `use-league` detects Sleeper's reversal round,
      but verify it. It changes which picks are yours.

## BEFORE DRAFT DAY

- [ ] Run `ff data refresh` (takes ~8 seconds; do **not** run it mid-draft)
- [ ] Run `ff doctor` and clear any warnings
- [ ] Rehearse with `ff draft mock --picks 40 --slot <yours>` then `ff on-clock`, so the
      output is familiar before you are on a 90-second clock

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
      If you think the board over- or under-values a factor, the weight is the knob.

## NOT NEEDED — do not send these

- ❌ Passwords for any platform. Sleeper's public read API needs none.
- ❌ Session cookies or auth tokens.
- ❌ ESPN/Yahoo credentials. Those platforms are not implemented and won't be for MVP.

## KNOWN LIMITATIONS — no action needed, but worth knowing

These are stated plainly here rather than hidden behind a confident-looking number:

- **Kickers and team defences are not modelled.** nflverse carries no team-defence
  scoring and we do not ingest kicking stats, so there is no honest projection for them.
  Draft them by consensus rank in the last two rounds; their value over replacement is
  near zero either way.
- **Projections come from a historical positional value curve**, not a per-player model:
  the market's ordering is mapped onto what each positional finish has historically been
  worth in your scoring. It models the rank, not the player. Import your own projections
  if you have better ones — they take precedence.
- **Schedule strength is deliberately low-confidence** (~0.36). Preseason
  defence-vs-position rests on last year's personnel and is weak evidence.
- **Bench multipliers in the replacement model are priors, not measurements.** Once
  historical draft data is ingested they should be estimated rather than assumed.
- **ADP is FantasyPros expert consensus rank**, used as a proxy. ECR is not ADP —
  experts and drafters differ systematically, especially at QB and TE. Import a real ADP
  feed with `ff import` and it takes precedence.

Nothing in this project transmits your league data anywhere. There is no telemetry.
