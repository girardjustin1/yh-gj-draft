"""Monte Carlo draft simulation and two-pick expected value.

The analytic survival model in :mod:`fantasy_draft.draft.availability` answers "will he
be there?" one player at a time. It cannot answer the question that actually decides a
pick:

    If I take this player now, how good is my *next* pick likely to be?

That requires simulating the intervening selections as a whole, because the players who
survive are correlated — the managers ahead of us cannot all take a running back.

**Two-pick expected value.**

    two_pick_EV(candidate) = value(candidate) + E[ best available at our next pick ]

A candidate with a slightly lower score can win outright if taking him leaves a much
better next pick — which is exactly the situation a pure ranking gets wrong.

**One deliberate approximation.** The intervening picks are simulated once over the full
pool, and each candidate's expected next pick is computed by excluding him from the
survivors of those same simulations. Strictly, removing a player from the pool nudges the
other managers' choices; that effect is second-order — one player out of several hundred
— and simulating separately per candidate would multiply the cost by the number of
candidates for no meaningful gain on a 90-second clock. The approximation is stated in
the output rather than hidden.

Results are seeded and reproducible, and cached on the draft state so repeated calls at
the same pick are free.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..config import AppConfig
from ..logging import get_logger
from ..models import PositionNeed
from .availability import DECAY_SCALE, DEFAULT_ADP_SD
from .state import DraftState

log = get_logger(__name__)

#: Players considered as simulation subjects. Beyond this nobody is plausibly taken.
SIM_POOL_SIZE = 220


@dataclass(slots=True)
class SimulationResult:
    """Outcome of simulating from our pick to our next one."""

    iterations: int
    picks_simulated: int
    survival: dict[str, float] = field(default_factory=dict)
    expected_position_losses: dict[str, float] = field(default_factory=dict)
    two_pick: dict[str, TwoPickValue] = field(default_factory=dict)
    seed: int = 0
    approximation_note: str = ""

    def probability_available(self, player_key: str) -> float | None:
        return self.survival.get(player_key)


@dataclass(slots=True)
class TwoPickValue:
    """Expected combined value of taking a candidate now and picking again later."""

    player_key: str
    value_now: float
    expected_next_value: float
    combined: float
    next_value_low: float
    next_value_high: float
    likely_next_position: str | None = None
    position_mix: dict[str, float] = field(default_factory=dict)


def state_fingerprint(state: DraftState, extra: str = "") -> str:
    """Stable key for caching: the draft is identical iff this is identical."""
    payload = f"{state.draft_id}|{state.picks_made}|{state.my_slot}|{state.current_pick}|{extra}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def simulate_to_next_pick(
    cfg: AppConfig,
    state: DraftState,
    board: pl.DataFrame,
    needs: list[PositionNeed],
    value_column: str = "player_score",
    iterations: int | None = None,
    candidates: list[str] | None = None,
) -> SimulationResult:
    """Simulate the picks between our turn and our next one.

    ``board`` must be the available players, carrying ``adp``, ``adp_sd``, ``position``
    and ``value_column``.
    """
    settings = cfg.weights.simulation
    n_sims = int(iterations or settings.iterations)

    if board.is_empty() or not needs:
        return SimulationResult(
            iterations=0, picks_simulated=0, seed=settings.seed,
            approximation_note="no intervening picks to simulate",
        )

    pool = board.head(SIM_POOL_SIZE)
    keys = pool["player_key"].to_list()
    positions = pool["position"].to_list()
    n_players = len(keys)

    adp = np.array(
        [float(a) if a is not None else 250.0 for a in pool["adp"].to_list()], dtype=float
    )
    spread = np.array(
        [
            max(float(s), 2.0) if s is not None else DEFAULT_ADP_SD
            for s in pool["adp_sd"].to_list()
        ],
        dtype=float,
    )
    decay = DECAY_SCALE * spread * max(settings.adp_noise_scale, 1e-6)
    values = np.array(
        [float(v) if v is not None else 0.0 for v in pool[value_column].to_list()],
        dtype=float,
    )

    unique_positions = sorted(set(positions))
    position_index = np.array(
        [unique_positions.index(p) for p in positions], dtype=int
    )

    rng = np.random.default_rng(settings.seed)
    alive = np.ones((n_sims, n_players), dtype=bool)

    losses = dict.fromkeys(unique_positions, 0.0)
    best_available_rate = settings.best_available_rate

    for need in needs:
        need_vector = np.array(
            [max(need.probabilities.get(p, 0.0), 1e-4) for p in unique_positions],
            dtype=float,
        )
        flat = np.full_like(need_vector, 1.0 / len(need_vector))

        # Some managers ignore roster need entirely and take best available. Drawn per
        # simulation so the room is a mix of behaviours rather than one archetype.
        ignores_need = rng.random(n_sims) < best_available_rate
        weights_by_position = np.where(
            ignores_need[:, None], flat[None, :], need_vector[None, :]
        )                                                        # (n_sims, n_positions)
        need_weights = weights_by_position[:, position_index]     # (n_sims, n_players)

        # Desirability is measured against the best player still on the board in each
        # simulation, not against the pick number: otherwise everyone already past their
        # ADP looks identically attractive.
        masked_adp = np.where(alive, adp[None, :], np.inf)
        reference = masked_adp.min(axis=1, keepdims=True)
        reference[~np.isfinite(reference)] = float(need.pick_overall)
        desirability = np.exp(-np.maximum(0.0, adp[None, :] - reference) / decay[None, :])

        weights = alive * desirability * need_weights
        totals = weights.sum(axis=1, keepdims=True)
        degenerate = (totals <= 0).ravel()
        if degenerate.any():
            weights[degenerate] = alive[degenerate].astype(float)
            totals = weights.sum(axis=1, keepdims=True)
            totals[totals <= 0] = 1.0

        cumulative = np.cumsum(weights / totals, axis=1)
        draws = rng.random((n_sims, 1))
        chosen = np.argmax(cumulative >= draws, axis=1)

        alive[np.arange(n_sims), chosen] = False
        taken_positions = position_index[chosen]
        for slot, position in enumerate(unique_positions):
            losses[position] += float(np.count_nonzero(taken_positions == slot)) / n_sims

    survival = {
        key: float(alive[:, index].mean()) for index, key in enumerate(keys)
    }

    two_pick: dict[str, TwoPickValue] = {}
    if candidates:
        two_pick = _two_pick_values(
            keys, positions, values, alive, candidates,
            limit=cfg.weights.simulation.two_pick_candidates,
        )

    return SimulationResult(
        iterations=n_sims,
        picks_simulated=len(needs),
        survival=survival,
        expected_position_losses=losses,
        two_pick=two_pick,
        seed=settings.seed,
        approximation_note=(
            "Intervening picks were simulated once over the full pool; each candidate's "
            "next-pick value excludes him from those same survivors. Removing one player "
            "from a pool of hundreds changes the other managers' choices only slightly."
        ),
    )


def _two_pick_values(
    keys: list[str],
    positions: list[str],
    values: np.ndarray,
    alive: np.ndarray,
    candidates: list[str],
    limit: int,
) -> dict[str, TwoPickValue]:
    """Expected best-available value at our next pick, per candidate."""
    index_of = {key: i for i, key in enumerate(keys)}
    order = np.argsort(-values)              # players by descending value
    ordered_alive = alive[:, order]          # (n_sims, n_players) in value order
    ordered_values = values[order]
    ordered_positions = [positions[i] for i in order]

    # For each simulation, the first two surviving players in value order. The best is
    # our next pick unless it is the candidate we just took, in which case it is the
    # runner-up.
    first = np.argmax(ordered_alive, axis=1)
    has_any = ordered_alive.any(axis=1)
    masked = ordered_alive.copy()
    masked[np.arange(len(first)), first] = False
    second = np.argmax(masked, axis=1)
    has_second = masked.any(axis=1)

    results: dict[str, TwoPickValue] = {}
    for key in candidates[:limit]:
        player_index = index_of.get(key)
        if player_index is None:
            continue
        position_in_order = int(np.where(order == player_index)[0][0])

        take_second = first == position_in_order
        chosen = np.where(take_second, second, first)
        valid = np.where(take_second, has_second, has_any)

        next_values = np.where(valid, ordered_values[chosen], 0.0)
        mean = float(next_values.mean())
        low, high = (float(x) for x in np.percentile(next_values, [10, 90]))

        picked_positions = [ordered_positions[i] for i in chosen[valid]]
        mix: dict[str, float] = {}
        for position in picked_positions:
            mix[position] = mix.get(position, 0.0) + 1.0 / max(len(picked_positions), 1)
        likely = max(mix, key=mix.get) if mix else None

        value_now = float(values[player_index])
        results[key] = TwoPickValue(
            player_key=key,
            value_now=value_now,
            expected_next_value=mean,
            combined=value_now + mean,
            next_value_low=low,
            next_value_high=high,
            likely_next_position=likely,
            position_mix={p: round(v, 3) for p, v in sorted(mix.items(), key=lambda kv: -kv[1])},
        )
    return results


def blend_survival(
    analytic: dict[str, float], simulated: dict[str, float], weight: float = 0.5
) -> dict[str, float]:
    """Combine the analytic and simulated survival estimates.

    They are built from the same inputs but make different approximations — the analytic
    model treats picks as independent, the simulation captures the correlation between
    them. Where they agree, confidence is warranted; where they diverge, averaging is
    more honest than picking a favourite.
    """
    keys = set(analytic) | set(simulated)
    out: dict[str, float] = {}
    for key in keys:
        a, s = analytic.get(key), simulated.get(key)
        if a is None:
            out[key] = float(s)
        elif s is None:
            out[key] = float(a)
        else:
            out[key] = (1 - weight) * float(a) + weight * float(s)
    return out
