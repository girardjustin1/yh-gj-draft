"""Adaptive roster-construction strategy.

Four familiar shapes — Balanced, Hero RB, Robust RB, Zero RB — held as *soft
probabilities* that move with what actually falls, not as instructions.

Nothing here ever says "draft a running back first". The strategy state is
**descriptive**: it reads what our roster has become and what the board is offering, and
reports which construction we are drifting toward and why. If we took elite receivers in
rounds 1 and 2 because the room overdrafted running backs, the right response is to
recognize we are heading for Zero RB and evaluate accordingly — not to panic and reach.

Its influence on the recommendation is deliberately the smallest weight in the Draft Now
Score (5%). It is a tiebreaker and an explanation, not a driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import LeagueConfig, StrategyPriors
from ..models import RosterSnapshot, Strategy

#: Round by which the early-round shape is considered established.
SHAPE_ROUNDS = 6


@dataclass(slots=True)
class StrategyState:
    """Where our roster construction currently sits."""

    probabilities: dict[Strategy, float] = field(default_factory=dict)
    primary: Strategy = Strategy.BALANCED
    alternative: Strategy | None = None
    confidence: float = 0.0
    reason: str = ""

    @property
    def label(self) -> str:
        names = {
            Strategy.BALANCED: "Balanced",
            Strategy.HERO_RB: "Hero RB",
            Strategy.ROBUST_RB: "Robust RB",
            Strategy.ZERO_RB: "Zero RB",
        }
        primary = names[self.primary]
        if self.alternative and self.probabilities.get(self.alternative, 0) > 0.25:
            return f"{primary} / {names[self.alternative]}"
        return primary

    def fit(self, position: str, roster: RosterSnapshot | None, league: LeagueConfig) -> float:
        """How well taking ``position`` now suits the current strategy mix, 0-100.

        Weighted by each strategy's own probability, so a genuinely uncertain state
        produces a near-neutral 50 rather than a confident nudge.
        """
        counts = dict(roster.position_counts) if roster else {}
        rb, wr = counts.get("RB", 0), counts.get("WR", 0)
        scores: dict[Strategy, float] = {}

        for strategy in Strategy:
            if strategy is Strategy.ZERO_RB:
                scores[strategy] = {"RB": 30.0, "WR": 78.0, "TE": 62.0, "QB": 55.0}.get(
                    position, 50.0
                )
            elif strategy is Strategy.ROBUST_RB:
                scores[strategy] = {"RB": 80.0, "WR": 45.0, "TE": 45.0, "QB": 40.0}.get(
                    position, 50.0
                )
            elif strategy is Strategy.HERO_RB:
                # One anchor back, then receivers — so RB is attractive only if we have none.
                scores[strategy] = (
                    {"RB": 82.0, "WR": 55.0, "TE": 50.0, "QB": 45.0}.get(position, 50.0)
                    if rb == 0
                    else {"RB": 42.0, "WR": 76.0, "TE": 60.0, "QB": 50.0}.get(position, 50.0)
                )
            else:
                # Balanced simply prefers whatever we are shortest of.
                shortfall = {
                    "RB": max(0, league.roster.rb - rb),
                    "WR": max(0, league.roster.wr - wr),
                    "TE": max(0, league.roster.te - counts.get("TE", 0)),
                    "QB": max(0, league.roster.qb - counts.get("QB", 0)),
                }
                scores[strategy] = 50.0 + 12.0 * min(shortfall.get(position, 0), 2)

        total = sum(self.probabilities.values()) or 1.0
        return sum(
            scores[strategy] * probability / total
            for strategy, probability in self.probabilities.items()
        )


def classify(
    league: LeagueConfig,
    roster: RosterSnapshot | None,
    priors: StrategyPriors,
    current_round: int,
    rb_value_available: float = 50.0,
    rb_run_intensity: float = 0.0,
) -> StrategyState:
    """Infer the current strategy mix from our roster and the board.

    ``rb_value_available`` is the draft-room "value created at RB" score; a high value
    means elite backs are still falling, which legitimises Robust RB. ``rb_run_intensity``
    is how hard the room is taking them.
    """
    weights = {
        Strategy.BALANCED: priors.balanced,
        Strategy.HERO_RB: priors.hero_rb,
        Strategy.ROBUST_RB: priors.robust_rb,
        Strategy.ZERO_RB: priors.zero_rb,
    }
    counts = dict(roster.position_counts) if roster else {}
    rb = counts.get("RB", 0)
    wr = counts.get("WR", 0)
    picks_made = sum(counts.values())
    reasons: list[str] = []

    # What we have already done dominates once a few picks are in — the roster is a fact,
    # the priors are a guess.
    if picks_made >= 2:
        early = min(current_round, SHAPE_ROUNDS)
        evidence = min(1.0, picks_made / 4.0)

        if rb == 0 and wr >= 2:
            weights[Strategy.ZERO_RB] *= 1 + 6.0 * evidence
            weights[Strategy.ROBUST_RB] *= 0.2
            reasons.append(f"{wr} WR and no RB through {early} rounds")
        elif rb == 1 and wr >= 2:
            weights[Strategy.HERO_RB] *= 1 + 4.0 * evidence
            weights[Strategy.ROBUST_RB] *= 0.35
            reasons.append("one anchor back plus receivers")
        elif rb >= 3:
            weights[Strategy.ROBUST_RB] *= 1 + 5.0 * evidence
            weights[Strategy.ZERO_RB] *= 0.05
            weights[Strategy.HERO_RB] *= 0.3
            reasons.append(f"{rb} RBs already rostered")
        elif rb >= 2 and wr >= 2:
            weights[Strategy.BALANCED] *= 1 + 3.0 * evidence
            reasons.append("even RB/WR split")

    # League format nudges. Full PPR and a third starting receiver make Zero RB more
    # viable; the reverse makes it harder.
    if league.scoring.reception >= 1.0:
        weights[Strategy.ZERO_RB] *= 1.25
        reasons.append("full PPR supports pass-catching depth")
    elif league.scoring.reception <= 0.0:
        weights[Strategy.ZERO_RB] *= 0.7
        weights[Strategy.ROBUST_RB] *= 1.2
        reasons.append("standard scoring favours volume backs")
    if league.roster.wr >= 3:
        weights[Strategy.ZERO_RB] *= 1.15
    if league.roster.flex_slots_accepting("RB") >= 2:
        weights[Strategy.ROBUST_RB] *= 1.1

    # What the board is offering right now.
    if rb_value_available >= 65:
        weights[Strategy.ROBUST_RB] *= 1.3
        weights[Strategy.ZERO_RB] *= 0.85
        reasons.append("elite RB value is falling to us")
    if rb_run_intensity >= 50:
        weights[Strategy.ZERO_RB] *= 1.25
        reasons.append("the room is overdrafting RB")

    total = sum(weights.values()) or 1.0
    probabilities = {strategy: weight / total for strategy, weight in weights.items()}
    ordered = sorted(probabilities.items(), key=lambda kv: -kv[1])
    primary, top = ordered[0]
    alternative, second = ordered[1]

    return StrategyState(
        probabilities=probabilities,
        primary=primary,
        alternative=alternative if second > 0.15 else None,
        # Confidence is the separation between the top two, not the top probability:
        # a 40/38 split is genuinely uncertain even though 40% sounds decisive.
        confidence=min(1.0, (top - second) * 2.5 + 0.25 * min(picks_made / 4.0, 1.0)),
        reason="; ".join(reasons) if reasons else "no strong signal yet; using priors",
    )
