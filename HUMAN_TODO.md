# Human TODO

Only what the engine genuinely needs from you. Nothing here blocks development — mock
configuration and `ff draft mock` keep everything running until you fill these in.

Last updated: 2026-08-31

---

## 1. CHOOSE HOW THE DRAFT BOARD REACHES THE APP

Three paths. Pick one — you can change later.

### A. Sleeper — works today, needs **no credentials at all** ✅

Sleeper's read API is fully public. I need **only your public username**. No password,
no token, no OAuth, nothing secret.

- [ ] **Sleeper username** (the handle, not the display name), then:

```bash
ff sleeper connect <username>     # resolves username -> public user_id
ff sleeper leagues
ff sleeper use-league <league_id> # reads scoring, roster slots, draft format
ff draft sync                     # fills in your draft slot once the order is set
```

`use-league` writes your real settings into `config/league.yaml`, which settles most of
section 2 below automatically. It asks before overwriting.

> **Never send a password for any service.** Nothing in this project ever needs one.

### B. Yahoo — needs an approved application, and that is no longer instant ⚠️

Yahoo changed this: creating a developer app is **no longer sufficient by itself**.
Fantasy API access now requires a manual review by Yahoo's team, and it is **read-only**.
You submit your organisation, product, and use case — including saying it is personal or
single-league use — and incomplete submissions are closed without correspondence.

- [ ] Apply at <https://sports.yahoo.com/developer/access/> and wait for approval
- [ ] Once approved, give me:
  - [ ] **Client ID** and **Client Secret** — I will put them in `.env`, which is
        gitignored. Do not paste them into a chat you would not want logged.
  - [ ] The **redirect URI** you registered (e.g. `https://localhost:8000/callback`)
  - [ ] Your **league key**, in Yahoo's `<game_key>.l.<league_id>` form
  - [ ] One **OAuth consent**: I generate a URL, you approve it in a browser and paste
        the code back once. The refresh token is then stored locally.

`YahooDraftProvider` exists as a stub that raises with these instructions rather than
failing confusingly. Tell me when approval lands and I will implement it — the work
itself is small; the approval is the slow part.

**Given the approval delay, do not plan on Yahoo being ready for your draft.** Use C.

### C. Manual entry — works with any platform, no integration at all ✅

The pragmatic fallback for Yahoo, an in-person draft, or an API outage mid-draft. A
manually entered pick produces exactly the same state a synced one does.

```bash
ff draft start --slot 7           # your draft slot
ff draft pick "Ja'Marr Chase"     # after each selection, including other managers'
ff draft undo                     # fix a mistake
ff on-clock                       # when it is your turn
```

---

## 2. LEAGUE SETTINGS

These drive replacement level, VBD, scarcity and roster fit. If they are wrong, every
number downstream is wrong. `ff sleeper use-league` fills them in automatically; set them
by hand otherwise.

- [ ] **Platform** — `sleeper`, `yahoo`, or `manual` in `config/league.yaml`
- [ ] **Number of teams** — currently `12`
- [ ] **Scoring** — currently half-PPR (`scoring.reception: 0.5`). Use `0.0` for
      standard, `1.0` for full PPR
- [ ] **Passing TD value** — currently `4`; some leagues use `6`
- [ ] **Starting roster** — currently 1QB / 2RB / 2WR / 1TE / 1FLEX / 1K / 1DST
- [ ] **FLEX / SUPERFLEX** — **set `superflex: 1` if this is a superflex or 2QB league.**
      This changes QB valuation more than any other single setting.
- [ ] **Bench size** — currently `6`. Starters + bench must equal `draft.rounds`
- [ ] **Your draft slot** — currently unset. Required for the snake maths;
      `ff draft sync` fills it in once the order exists

Check with `ff config show` and `ff doctor`.

---

## 3. BEFORE DRAFT DAY

- [ ] `ff data refresh` — about 8 seconds. **Do not run this mid-draft.**
- [ ] `ff doctor` and clear any warnings
- [ ] Rehearse: `ff draft mock --picks 40 --slot <yours>` then `ff on-clock`, so the
      output is familiar before you are on a clock
- [ ] Open the dashboard once: `ff serve` → <http://127.0.0.1:8000>

---

## 4. OPTIONAL — improves accuracy

- [ ] **Your own projections** — `ff import projections ./file.csv`. Any CSV/Parquet/JSON
      with a name column plus either `fantasy_points` or component stats. Imported
      sources take precedence and are blended, never overwritten.
- [ ] **A real ADP feed** — `ff import adp ./adp.csv`. We currently use FantasyPros
      expert consensus rank as a proxy, and ECR is not ADP.
- [ ] **Keepers**, if any — they change the pool and every opponent's roster needs.
- [ ] **Tune weights** — `config/scoring_weights.yaml`, shown by `ff config weights`. The
      defaults are defensible starting points, not settled truth.

---

## KNOWN LIMITATIONS

Stated plainly rather than hidden behind a confident-looking number.

- **Kickers and team defences are not modelled.** nflverse carries no team-defence
  scoring and we do not ingest kicking stats. Draft them by consensus rank in the last
  two rounds; their value over replacement is near zero regardless.
- **Projections come from a historical positional value curve**, not a per-player model:
  the market's ordering is mapped onto what each positional finish has historically been
  worth in your scoring. It models the rank, not the player.
- **Floor / ceiling is an honest width, not a calibrated interval.** It combines expert
  disagreement with the season-to-season variance of that positional finish, skewed by
  injury and age risk. Read it as "roughly how wrong could this be".
- **Schedule strength is deliberately low-confidence** (~0.36). Preseason
  defence-vs-position rests on last year's personnel.
- **Bench multipliers in the replacement model are priors, not measurements.**
- **ADP is FantasyPros ECR used as a proxy.** Experts and drafters differ systematically,
  especially at QB and TE.

Nothing here transmits your league data anywhere. There is no telemetry. The dashboard
binds to localhost.
