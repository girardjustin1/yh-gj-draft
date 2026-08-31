# Scoring methodology

Why each component exists, what it consumes, how it is normalized, and where its weight
lives. No number in this system should be unexplainable.

Status: components are documented as they are implemented. Phase 0 defines the contract;
Phases 2–6 fill in the analytics.

## The contract every component honours

```python
ComponentScore(
    name="opportunity",
    raw_value=0.243,        # the real quantity, in its own units
    normalized=78.4,        # 0-100, comparable across players
    confidence=0.85,        # 0-1; drops when inputs are missing or stale
    method="position_percentile",
    source="nflverse_ff_opportunity",
    source_updated_at=...,
)
```

**`confidence` is not decoration.** When it is 0, the component's weight is redistributed
across the components we *do* know, rather than scoring a neutral 50. Scoring unknowns as
average would compress the board toward the mean and hide our ignorance.

## Normalization

Min/max scaling is avoided: one outlier would rescale everyone. Instead:

| Method | Used for | Why |
|---|---|---|
| **Position percentile** | opportunity, schedule | Comparing a TE's target share to a WR's is meaningless; rank within position. |
| **Position z-score → logistic** | projection, VBD | Preserves the *size* of gaps, which is the whole point of VBD. Squashed to 0–100 without clipping the tails flat. |
| **Signed pick delta** | market value | ADP minus current pick is already in a natural, interpretable unit. |
| **Bounded ratio** | tier cliff | Points to the next tier, relative to the position's typical inter-tier gap. |

Every implemented component records which method it used in `ComponentScore.method`.

## Player Score — "how good is he, ignoring the draft?"

Weights: `config/scoring_weights.yaml` → `player_score`.

| Component | Weight | Raw input | Notes |
|---|---|---|---|
| Projection | 0.35 | Consensus projected points *in our scoring rules* | Not borrowed PPR numbers — recomputed from our `ScoringRules`. |
| VBD | 0.25 | Projection − replacement at position | See below. |
| Opportunity | 0.15 | Snap/route/target/carry share, red-zone usage, expected fantasy points | Catches strong roles that last year's box score hides. |
| Offense environment | 0.10 | Team pace, scoring rate, efficiency | A good role on a bad offense is worth less. |
| Schedule | 0.075 | Position-specific defense-vs-position over the fantasy season | Deliberately small: preseason schedule strength is weak evidence. |
| Risk | 0.075 | Injury, role competition, sample size, source disagreement | Enters *inverted* — low risk raises the score. |

## Value Based Drafting

```
VBD(player) = projected_points(player) − replacement_points(position)
```

Replacement level is **estimated, not hard-coded**:

```
starter_demand(pos) = teams × (dedicated_slots + share_of_flex_slots)
replacement_rank(pos) = starter_demand(pos) × bench_multiplier(pos)
```

In a 12-team league with 2 RB + 1 FLEX, RB starter demand is 12 × (2 + 1/3) = 28. But
managers roster far more RBs than they start, so `bench_multiplier["RB"] = 1.55` puts
replacement near RB43 — materially below RB24, which is the point the spec makes.

`method: blended` mixes this with a `fixed_rank` anchor at `blend_weight`, so a strange
roster configuration cannot produce an absurd replacement level. Replacement points are
averaged over `smoothing_window` ranks either side so one noisy projection doesn't shift
every VBD number in the league.

**Limitation:** bench multipliers are currently priors, not measurements. Once historical
draft data is ingested they should be estimated from how many players at each position
are actually drafted before the position stops being startable.

## Tiers

`method: gap` walks each position's players in projection order and starts a new tier
where the gap to the next player exceeds `gap_sigma` standard deviations of the
position's adjacent-gap distribution. This produces interpretable tiers whose boundaries
correspond to real cliffs, and it is deterministic — the same board always tiers the same
way.

Derived: `tier`, `tier_rank`, `points_to_next_player`, `points_to_next_tier`, and
`tier_cliff_score` (how costly it is to wait).

## Value Score — "what does taking him *here* capture?"

Weights → `value_score`. Market value (ADP vs. current pick), tier cliff, scarcity, and
projection-vs-market disagreement.

A falling player is not automatically good: falls often carry news. Market value is a
component, not a verdict, and a large unexplained fall *raises* the risk component.

## Draft Now Score — "what do I take at this pick?"

Weights → `draft_now`. Player Score, Value Score, next-pick urgency (driven by survival
probability), tier scarcity, roster fit, draft-room behaviour, strategy fit.

Roster fit is a **modifier, not a rule**. A team with three RBs may still take a fourth
if the value is extreme; only structurally irrational builds are heavily penalized.

## Survival and two-pick expected value

`probability_gone_before_next_pick` combines the ADP distribution (FantasyPros ECR ships
`sd`, `best`, `worst`) with the actual rosters and needs of the managers picking between
now and our next turn. It is the single input that most changes a recommendation away
from "highest ranked available".

```
two_pick_expected_value(candidate) =
    value(candidate) + E[ best available at our next pick | we took candidate ]
```

Simulation output, not false precision: reported alongside its iteration count and a
confidence interval.

## Recommendation confidence

Lowered by: projection sources disagreeing, ADP sources disagreeing, uncertain player
identity, stale injury data, unclear depth chart, small rookie sample, high simulation
variance. Reported explicitly on every recommendation, alongside a staleness table.
