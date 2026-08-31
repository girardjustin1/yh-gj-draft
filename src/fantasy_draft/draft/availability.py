"""Will he still be there at my next pick?

This is the model that separates a decision engine from a rankings viewer. A player who
is certainly gone and a player who will certainly last twelve more picks should be
treated completely differently even if their season projections are identical.

**The model.** For each of the N managers picking between our turn and our next, we
estimate the probability *that manager takes this specific player*:

    P(pick k takes j)  =  need(team_k, position_j) × desirability(j)
                          ────────────────────────────────────────────
                          Σ over available i of need(team_k, pos_i) × desirability(i)

``desirability`` decays exponentially with how far a player sits behind the best player
still on the board, with the decay length set by his own ADP uncertainty — a tightly
ranked player is taken close to his ADP, a volatile one has his probability smeared over
a wider band. Measuring against the best *available* player rather than against the
current pick matters: a player with an ADP of 1 who has somehow lasted to pick 43 is far
more likely to be taken than one with an ADP of 41, and anchoring on the pick number
would treat every player already past his ADP as equally attractive. ``need`` comes from
:mod:`fantasy_draft.draft.opponent_needs` and depends on roster shape, never on identity.

Survival is then the probability no one takes him:

    P(available at our next pick)  =  Π over k of (1 − P(pick k takes j))

**Why not just the ADP normal curve.** Reading survival straight off ``1 − Φ((pick −
ADP)/sd)`` ignores the specific room. The last elite tight end has a very different fate
depending on whether the eight managers in front of us already have one. We compute the
ADP-only figure too, and report it, because where the two disagree sharply that is
itself worth saying out loud.

Monte Carlo (Phase 5) checks this analytic model rather than replacing it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from ..config import LeagueConfig
from ..models import PositionNeed, SurvivalEstimate
from .state import DraftState

#: Multiplies the ADP standard deviation to get the desirability decay length, in picks.
DECAY_SCALE = 1.25

#: Fallback ADP spread when a player has none, in picks.
DEFAULT_ADP_SD = 12.0

#: Only players plausibly in range are considered competitors for a pick.
CANDIDATE_WINDOW = 60


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def adp_survival(adp: float | None, adp_sd: float | None, next_pick: int) -> float:
    """Probability of surviving to ``next_pick`` using the ADP distribution alone.

    The naive baseline. Reported alongside the roster-aware figure so a large divergence
    can be pointed at rather than buried.
    """
    if adp is None:
        return 0.5
    spread = max(float(adp_sd or DEFAULT_ADP_SD), 1.0)
    # P(he is still there) = P(his true draft slot is beyond our next pick).
    return 1.0 - _normal_cdf((next_pick - float(adp)) / spread)


@dataclass(slots=True)
class SurvivalResult:
    """Survival probabilities for every candidate, plus the diagnostics behind them."""

    estimates: dict[str, SurvivalEstimate]
    expected_position_losses: dict[str, float]
    picks_modelled: int
    method: str

    def probability_gone(self, player_key: str) -> float:
        estimate = self.estimates.get(player_key)
        return estimate.probability_gone if estimate else 0.5


def survival_probabilities(
    state: DraftState,
    league: LeagueConfig,
    board: pl.DataFrame,
    needs: list[PositionNeed],
    candidate_limit: int = CANDIDATE_WINDOW,
) -> SurvivalResult:
    """Estimate survival to our next pick for the top available players.

    ``board`` must already exclude drafted players and carry ``adp`` and ``position``.
    """
    next_pick = state.my_next_pick
    if next_pick is None or not needs or board.is_empty():
        # No next pick means nothing to survive to — everything is "available".
        return SurvivalResult(
            estimates={
                key: SurvivalEstimate(
                    player_key=key, probability_available=1.0, probability_gone=0.0,
                    method="adp_normal", confidence=0.0,
                )
                for key in board["player_key"].to_list()[:candidate_limit]
            }
            if not board.is_empty() else {},
            expected_position_losses={},
            picks_modelled=0,
            method="none",
        )

    pool = board.head(max(candidate_limit, 120)).select(
        "player_key", "position", "adp", "adp_sd", "player_score"
    ).to_dicts()

    # Desirability: exponential decay in consensus rank, widened by the player's own ADP
    # uncertainty so a volatile player competes across a broader band of picks.
    for entry in pool:
        entry["adp_value"] = float(entry["adp"]) if entry["adp"] is not None else 250.0
        entry["spread"] = max(float(entry["adp_sd"] or DEFAULT_ADP_SD), 2.0)

    alive = {entry["player_key"]: 1.0 for entry in pool}
    position_of = {entry["player_key"]: entry["position"] for entry in pool}
    losses: dict[str, float] = {}

    for need in needs:
        pick = need.pick_overall
        # Anchor desirability on the best player still plausibly on the board, not on the
        # pick number: everyone already past their ADP would otherwise look identically
        # attractive, and ADP 1 lasting to pick 43 is a very different proposition from
        # ADP 41 doing so.
        reference = min(
            (
                entry["adp_value"]
                for entry in pool
                if alive[entry["player_key"]] > 0.05
            ),
            default=float(pick),
        )
        weights: dict[str, float] = {}
        for entry in pool:
            key = entry["player_key"]
            survival_so_far = alive[key]
            if survival_so_far <= 1e-6:
                continue
            decay = DECAY_SCALE * entry["spread"]
            desirability = math.exp(-max(0.0, entry["adp_value"] - reference) / decay)
            need_weight = need.probabilities.get(entry["position"], 0.0)
            weights[key] = survival_so_far * desirability * need_weight

        total = sum(weights.values())
        if total <= 0:
            continue
        for key, weight in weights.items():
            taken_here = weight / total
            entry_position = position_of[key]
            losses[entry_position] = losses.get(entry_position, 0.0) + taken_here
            alive[key] *= 1.0 - taken_here

    estimates: dict[str, SurvivalEstimate] = {}
    for entry in pool[:candidate_limit]:
        key = entry["player_key"]
        available = min(max(alive[key], 0.0), 1.0)
        confidence = 0.75 if entry["adp"] is not None else 0.3
        estimates[key] = SurvivalEstimate(
            player_key=key,
            probability_available=available,
            probability_gone=1.0 - available,
            method="blended",
            confidence=confidence,
        )

    return SurvivalResult(
        estimates=estimates,
        expected_position_losses=losses,
        picks_modelled=len(needs),
        method="roster_aware_analytic",
    )


def attach_survival(
    board: pl.DataFrame, result: SurvivalResult, next_pick: int | None
) -> pl.DataFrame:
    """Add survival columns to the board, including the ADP-only baseline."""
    if board.is_empty():
        return board

    keys = board["player_key"].to_list()
    adps = board["adp"].to_list()
    sds = (
        board["adp_sd"].to_list() if "adp_sd" in board.columns else [None] * board.height
    )

    adp_only = [
        adp_survival(adp, sd, next_pick) if next_pick is not None else 1.0
        for adp, sd in zip(adps, sds, strict=True)
    ]

    # Every player gets a survival probability. Outside the roster-aware model's window
    # we fall back to the ADP curve at lower confidence -- because a player 90 picks past
    # his ADP is not "unknown", he is "almost certainly still there", and the difference
    # matters enormously. Leaving him null would let the composition step redistribute
    # his urgency weight, which *rewards* being unmodellable: a deep player would carry
    # no urgency penalty at all and float to the top of the board on market value alone.
    available: list[float] = []
    confidences: list[float] = []
    for index, key in enumerate(keys):
        estimate = result.estimates.get(key)
        if estimate is not None:
            available.append(estimate.probability_available)
            confidences.append(estimate.confidence)
        else:
            available.append(adp_only[index])
            confidences.append(0.4 if adps[index] is not None else 0.15)

    frame = board.with_columns(
        pl.Series("probability_available", available, dtype=pl.Float64),
        pl.Series("probability_available_adp_only", adp_only, dtype=pl.Float64),
        pl.Series("survival_confidence", confidences, dtype=pl.Float64),
    ).with_columns((1.0 - pl.col("probability_available")).alias("probability_gone"))

    # Urgency is simply how likely he is to disappear. A player who will still be sitting
    # there in twelve picks is not urgent, and scores accordingly.
    return frame.with_columns(
        (100.0 * pl.col("probability_gone").fill_null(0.5))
        .clip(0, 100)
        .alias("next_pick_urgency_score")
    )
