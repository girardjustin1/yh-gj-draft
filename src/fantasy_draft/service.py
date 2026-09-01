"""``analyze_current_pick()`` — the one function that answers "who do I draft?".

Everything that asks the question goes through here: the CLI, the local web API, and
Claude. That is deliberate. If the GUI and the assistant computed their answers by
different routes they would eventually disagree mid-draft, and the one thing worse than
no recommendation is two conflicting ones.

The pipeline is the spec's workflow, in order:

    refresh live draft -> rebuild DraftState -> available pool -> our roster ->
    opponent rosters -> current pick -> next pick -> picks until next -> tiers ->
    scarcity -> draft-room behaviour -> survival probabilities -> simulation ->
    two-pick expected value -> rank -> recommendation + alternatives + confidence

Every stage degrades rather than raising: a failed sync falls back to the last stored
board and says so; a failed simulation falls back to the analytic survival model and says
so. On a 90-second clock, an answer with honest caveats beats an exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import polars as pl

from .analytics.board import Board
from .analytics.draft_room import DraftRoomRead
from .analytics.lineup_value import lineup_upgrades, positional_needs
from .analytics.outcomes import outcome_label
from .config import AppConfig
from .constants import FLEX_ELIGIBILITY, POSITION_ORDER
from .database import Database
from .draft.simulator import SimulationResult
from .draft.state import DraftState
from .draft.store import load_state, save_state
from .draft.strategies import StrategyState
from .logging import get_logger
from .models import Recommendation, RosterSnapshot
from .recommendation.ranker import recommend

log = get_logger(__name__)

#: How stale a draft sync may be before we start shouting about it, in seconds.
STALE_SYNC_SECONDS = 180


@dataclass(slots=True)
class LineupSlot:
    """One starting-lineup slot, filled or empty."""

    slot: str
    player_key: str | None = None
    name: str | None = None
    position: str | None = None
    team: str | None = None
    is_bench: bool = False

    @property
    def filled(self) -> bool:
        return self.player_key is not None


@dataclass(slots=True)
class PickAnalysis:
    """Everything the five-area interface needs, from one call."""

    state: DraftState
    recommendation: Recommendation
    board: Board
    room: DraftRoomRead
    strategy: StrategyState
    frame: pl.DataFrame
    simulation: SimulationResult | None = None
    provider: str = "none"
    synced: bool = False
    sync_error: str | None = None
    generated_at: datetime = field(default_factory=datetime.now)
    #: VBD for every player, drafted included. The board only carries the available
    #: ones, so team strength would score every rostered player at zero without this.
    all_vbd: dict[str, float] = field(default_factory=dict)

    # --- the five areas -------------------------------------------------------------

    @property
    def is_stale(self) -> bool:
        return (
            datetime.now() - self.state.synced_at
        ).total_seconds() > STALE_SYNC_SECONDS

    def on_the_clock(self) -> dict[str, Any]:
        """Area 1: the decision, and only what bears on it."""
        rec = self.recommendation
        primary = rec.primary
        return {
            "pick_label": rec.pick_label,
            "overall_pick": rec.overall_pick,
            "round": self.state.board.round_for(rec.overall_pick),
            "next_pick_label": (
                self.state.board.label(rec.next_pick_overall)
                if rec.next_pick_overall else None
            ),
            "next_pick_overall": rec.next_pick_overall,
            "picks_until_next": rec.picks_until_next,
            "is_my_pick": self.state.is_my_pick,
            "picks_until_my_turn": self.state.picks_until_my_turn,
            "my_slot": self.state.my_slot,
            "teams": self.state.board.teams,
            "recommendation": _candidate_payload(primary) if primary else None,
            "alternatives": [_candidate_payload(c) for c in rec.alternatives],
            "confidence": round(rec.confidence, 3),
            "reason": rec.explanation,
            "strategy": {
                "current": self.strategy.label,
                "confidence": round(self.strategy.confidence, 3),
                "why": self.strategy.reason,
                "probabilities": {
                    k.value: round(v, 3) for k, v in self.strategy.probabilities.items()
                },
            },
            "warnings": rec.warnings,
            "stale": self.is_stale,
            "synced_at": self.state.synced_at.isoformat(),
            "provider": self.provider,
        }

    def best_available(self, limit: int = 60, position: str | None = None) -> list[dict[str, Any]]:
        """Area 2: the available pool with every column the interface offers."""
        frame = self.frame
        if position and position.upper() != "ALL":
            frame = frame.filter(pl.col("position") == position.upper())
        rows = frame.head(limit).to_dicts()
        return [_board_row(row, rank) for rank, row in enumerate(rows, start=1)]

    def my_roster(self) -> dict[str, Any]:
        """Area 3: our actual starting lineup, slot by slot, with the holes visible."""
        roster = self.state.my_roster()
        slots = fill_lineup(self.config_league_roster(), roster, self._player_lookup())
        starters = [s for s in slots if not s.is_bench]
        return {
            "starters": [_slot_payload(s) for s in starters],
            "bench": [_slot_payload(s) for s in slots if s.is_bench],
            "position_counts": roster.position_counts if roster else {},
            "size": roster.size if roster else 0,
            "unfilled_starters": [s.slot for s in starters if not s.filled],
        }

    def what_i_need(self) -> dict[str, Any]:
        """What am I short of, and who is the best available player who fixes it?

        The direct answer to "tell me based on what I need". Reported alongside the
        recommendation rather than folded into it: early in a draft nearly every pick is
        a bench player by lineup-upgrade, and ranking by need would produce exactly the
        positional reaches the engine exists to avoid.
        """
        roster = self.state.my_roster()
        lookup = self._player_lookup()
        slots = fill_lineup(self._league.roster, roster, lookup)
        filled = {}
        for slot in slots:
            if slot.is_bench:
                continue
            filled[slot.slot] = filled.get(slot.slot, False) or slot.filled

        needs = positional_needs(self._league, self._roster_players(), filled, self.frame)
        return {
            "unfilled": [n.slot for n in needs if not n.filled],
            "slots": [
                {
                    "slot": n.slot,
                    "eligible": list(n.eligible),
                    "filled": n.filled,
                    "best": (
                        {
                            "player_key": n.best_player_key,
                            "name": n.best_name,
                            "position": n.best_position,
                            "team": n.best_team,
                            "draft_now": n.draft_now,
                            "lineup_upgrade": n.lineup_upgrade,
                            "probability_gone": n.probability_gone,
                            "tier": n.tier,
                        }
                        if n.best_player_key else None
                    ),
                }
                for n in needs
            ],
        }

    def _roster_players(self) -> list[tuple[str, float]]:
        """``(position, projected_points)`` for everyone already on our roster."""
        roster = self.state.my_roster()
        if roster is None or not roster.player_keys:
            return []
        projections = dict(
            zip(self.board.frame["player_key"], self.board.frame["vbd"], strict=True)
        ) if not self.board.frame.is_empty() else {}
        out: list[tuple[str, float]] = []
        for pick in self.state.picks:
            if pick.player_key in roster.player_keys and pick.position:
                out.append((pick.position, float(projections.get(pick.player_key) or 0.0)))
        return out

    def team_strength(self) -> dict[str, Any]:
        """How well covered is each position on *my* roster, and what should I take next?

        This was originally a comparison against the other teams in the draft. That reads
        well when every pick has been recorded, and collapses when they have not: mark
        only your own picks and there are no other teams, so every position reports "1 of
        1" and the bars mean nothing. Worse, it made a readout about *your* roster depend
        on bookkeeping about everyone else's.

        So coverage is measured against the thing that is always knowable — the lineup you
        have to fill:

            required   starting slots this position must fill (dedicated + flex share)
            have       value of the starters you already hold there
            target     that, plus the best players still available to fill the rest
            coverage   have / target

        A position with two elite backs and one slot left reads high; an empty tight end
        slot reads zero. Priority then combines being uncovered with the position drying
        up before your next pick, so it answers "what next" rather than "how am I doing".

        The league comparison survives as a secondary field, reported only when opponent
        picks actually exist.
        """
        league = self._league
        frame = self.board.frame
        available_vbd = (
            dict(zip(frame["player_key"], frame["vbd"], strict=True))
            if not frame.is_empty() and "vbd" in frame.columns else {}
        )
        vbd = {**self.all_vbd, **available_vbd}
        position_of = {
            pick.player_key: pick.position for pick in self.state.picks if pick.player_key
        }

        roster = self.state.my_roster()
        my_keys = list(roster.player_keys) if roster else []
        losses = self.simulation.expected_position_losses if self.simulation else {}

        # Best available at each position, for the "what could still fill this" side.
        by_position: dict[str, list[float]] = {}
        if not frame.is_empty():
            for row in frame.select("position", "vbd").iter_rows(named=True):
                if row["vbd"] is None:
                    continue
                by_position.setdefault(row["position"], []).append(float(row["vbd"]))
        for values in by_position.values():
            values.sort(reverse=True)

        # Opponent rosters, used only for the optional league comparison.
        rosters = self.state.rosters()
        me = self.state.my_team_id
        opponent_totals: dict[str, dict[str, float]] = {}
        for team_id, snapshot in rosters.items():
            if team_id == me or not snapshot.player_keys:
                continue
            per: dict[str, float] = {}
            for key in snapshot.player_keys:
                position = position_of.get(key)
                if position:
                    per[position] = per.get(position, 0.0) + float(vbd.get(key) or 0.0)
            opponent_totals[team_id] = per

        rows = []
        for position in ("QB", "RB", "WR", "TE"):
            required = max(
                1,
                int(round(league.starter_demand(position) / max(league.teams, 1))),
            )
            mine = sorted(
                (float(vbd.get(k) or 0.0) for k in my_keys if position_of.get(k) == position),
                reverse=True,
            )
            starters = mine[:required]
            have = sum(max(v, 0.0) for v in starters)

            # Fill the remaining slots with the best still on the board.
            gap = max(0, required - len(starters))
            fillers = by_position.get(position, [])[:gap]
            target = have + sum(max(v, 0.0) for v in fillers)
            coverage = (have / target * 100.0) if target > 0 else (100.0 if gap == 0 else 0.0)

            drain = min(1.0, float(losses.get(position, 0.0)) / 4.0)
            priority = 0.70 * (100.0 - coverage) + 0.30 * (drain * 100.0)

            # League comparison, only where there is something to compare against.
            others = sorted(t.get(position, 0.0) for t in opponent_totals.values())
            league_rank = None
            if others:
                below = sum(1 for value in others if value < have)
                league_rank = {
                    "rank": len(others) + 1 - below,
                    "teams": len(others) + 1,
                    "percentile": round(below / len(others) * 100.0),
                    "median": round(others[len(others) // 2], 1),
                }

            best = next(
                (r for r in self.best_available(limit=250) if r["position"] == position),
                None,
            )
            rows.append(
                {
                    "position": position,
                    "required": required,
                    "filled": len(starters),
                    "have_value": round(have, 1),
                    "target_value": round(target, 1),
                    "coverage": round(coverage),
                    "priority": round(min(100.0, max(0.0, priority))),
                    "expected_gone_before_next_pick": round(
                        float(losses.get(position, 0.0)), 1
                    ),
                    "league": league_rank,
                    "best_available": (
                        {
                            "name": best["name"], "player_key": best["player_key"],
                            "draft_now": best["draft_now"],
                            "probability_gone": best["probability_gone"],
                        } if best else None
                    ),
                }
            )

        rows.sort(key=lambda r: -r["priority"])

        bench = self._bench_plan(rows, my_keys, position_of)
        return {
            "positions": rows,
            "bench": bench,
            "top_priority": rows[0]["position"] if rows else None,
            "has_league_comparison": bool(opponent_totals),
            "opponent_teams": len(opponent_totals),
        }

    def _bench_plan(
        self,
        rows: list[dict[str, Any]],
        my_keys: list[str],
        position_of: dict[str, str],
    ) -> dict[str, Any]:
        """What the bench should look like, and who to stash next.

        Starters are a lineup problem; a bench is an insurance problem. The slots skew
        toward the positions you both start most and lose most — running backs miss time
        and lose jobs far more often than quarterbacks — so the target split is the
        starting requirement weighted by :data:`BENCH_ATTRITION`.

        Bench candidates are also chosen differently. For a starter you want the best
        expected season; for a bench spot you are buying the tail, so candidates are
        ranked by ceiling rather than by median. A safe 140-point backup is worth less
        than a volatile one who might return 220 if the job opens up.

        The IR slot is not part of this: you do not draft for it.
        """
        from .constants import BENCH_ATTRITION, BENCH_WINDOW

        league = self._league
        total = league.roster.bench
        required = {r["position"]: r["required"] for r in rows}

        # Everyone rostered beyond a starting slot is already bench depth.
        starters_held = sum(min(r["filled"], r["required"]) for r in rows)
        filled = max(0, len(my_keys) - starters_held)
        remaining = max(0, total - filled)

        weights = {
            position: required.get(position, 0) * BENCH_ATTRITION.get(position, 0.5)
            for position in ("QB", "RB", "WR", "TE")
        }
        weight_total = sum(weights.values()) or 1.0

        # Largest-remainder allocation, so the targets sum to the bench size exactly
        # rather than drifting by a player or two after rounding.
        exact = {p: total * w / weight_total for p, w in weights.items()}
        target = {p: int(v) for p, v in exact.items()}
        for position, _ in sorted(
            ((p, exact[p] - target[p]) for p in exact), key=lambda kv: -kv[1]
        )[: total - sum(target.values())]:
            target[position] += 1

        held_bench: dict[str, int] = {}
        for row in rows:
            position = row["position"]
            extra = max(0, row["filled"] - row["required"])
            if extra:
                held_bench[position] = extra
        # Players at a position with no starting slot filled yet still count as depth
        # once the slot is covered; anything past `required` is bench by definition.
        for key in my_keys:
            position = position_of.get(key)
            if position and position not in required:
                held_bench[position] = held_bench.get(position, 0) + 1

        pool = self.best_available(limit=250)
        slots = []
        for position in ("RB", "WR", "TE", "QB"):
            want = target.get(position, 0)
            have = held_bench.get(position, 0)
            # Skip the players you would spend on the starting slots still open at this
            # position — they are starters, not stashes. Suggesting the best player
            # available as a "bench pick" is just the board again under a different name.
            starter_gap = max(0, required.get(position, 0) - sum(
                r["filled"] for r in rows if r["position"] == position
            ))
            at_position = [r for r in pool if r["position"] == position]
            # A window of players actually in range, not the whole tail. Ranking the tail
            # purely by ceiling surfaces names with a median of 23 and a ceiling of 186 —
            # that width is our ignorance about a camp body, not upside, and drafting it
            # is worse than taking the obvious player.
            window = at_position[starter_gap : starter_gap + BENCH_WINDOW]
            candidates = [r for r in window if r["ceiling"] is not None]
            candidates.sort(key=lambda r: -(r["ceiling"] or 0))
            best = candidates[0] if candidates else None
            slots.append(
                {
                    "position": position,
                    "target": want,
                    "have": have,
                    "short": max(0, want - have),
                    "best_upside": (
                        {
                            "name": best["name"],
                            "player_key": best["player_key"],
                            "ceiling": best["ceiling"],
                            "median": best["median"],
                            "outcome": best["outcome"],
                            "probability_gone": best["probability_gone"],
                        } if best else None
                    ),
                }
            )
        slots.sort(key=lambda s: (-s["short"], -s["target"]))

        return {
            "total": total,
            "filled": filled,
            "remaining": remaining,
            "ir_slots": league.roster.ir,
            "slots": slots,
            "note": (
                "Bench slots weight the positions you start most and lose most. "
                "Candidates are ranked by ceiling, not median — a bench pick is a bet on "
                "the tail."
            ),
        }

    def who_makes_it_back(self, limit: int = 10) -> list[dict[str, Any]]:
        """Area 4: survival to our next pick, which is the whole point of the app."""
        if self.recommendation.next_pick_overall is None:
            return []
        rows = self.frame.head(limit).to_dicts()
        out = []
        for row in rows:
            available = row.get("probability_available")
            if available is None:
                continue
            out.append(
                {
                    "player_key": row["player_key"],
                    "name": row.get("player_name"),
                    "position": row.get("position"),
                    "team": row.get("team"),
                    "probability_available": round(float(available), 3),
                    "probability_gone": round(1.0 - float(available), 3),
                    "adp_only_estimate": (
                        round(float(row["probability_available_adp_only"]), 3)
                        if row.get("probability_available_adp_only") is not None else None
                    ),
                    "confidence": round(float(row.get("survival_confidence") or 0.0), 3),
                    "draft_now": _round(row.get("draft_now_score")),
                }
            )
        return sorted(out, key=lambda r: r["probability_available"])

    def what_if(self, limit: int = 4) -> list[dict[str, Any]]:
        """Area 5: taking each candidate now, priced across both picks."""
        if self.simulation is None or not self.simulation.two_pick:
            return []
        lookup = dict(zip(self.frame["player_key"], self.frame["player_name"], strict=True))
        values = sorted(self.simulation.two_pick.values(), key=lambda v: -v.combined)
        best = values[0].combined if values else 0.0
        return [
            {
                "player_key": v.player_key,
                "name": lookup.get(v.player_key, v.player_key),
                "value_now": round(v.value_now, 1),
                "expected_next_value": round(v.expected_next_value, 1),
                "combined": round(v.combined, 1),
                "delta_vs_best": round(v.combined - best, 1),
                "next_value_range": [round(v.next_value_low, 1), round(v.next_value_high, 1)],
                "likely_next_position": v.likely_next_position,
                "position_mix": v.position_mix,
            }
            for v in values[:limit]
        ]

    def draft_environment(self) -> dict[str, Any]:
        """Secondary: what the room is doing, per position."""
        losses = (
            self.simulation.expected_position_losses
            if self.simulation
            else {}
        )
        return {
            position: {
                "demand": self.room.demand.get(position, 50.0),
                "run_intensity": self.room.run_intensity.get(position, 0.0),
                "value_created": self.room.value_created.get(position, 50.0),
                "expected_gone_before_next_pick": round(losses.get(position, 0.0), 2),
                "scarcity": round(entry.score, 1),
                "startable_available": entry.startable_available,
                "remaining_starter_slots": round(entry.remaining_demand, 1),
            }
            for position, entry in self.board.scarcity.items()
        }

    # --- helpers ---------------------------------------------------------------------

    def config_league_roster(self) -> Any:
        return self._league.roster

    def _player_lookup(self) -> dict[str, dict[str, Any]]:
        return {
            pick.player_key: {
                "name": pick.player_name,
                "position": pick.position,
                "team": pick.nfl_team,
            }
            for pick in self.state.picks
            if pick.player_key
        }

    _league: Any = None

    def to_dict(self, board_limit: int = 60) -> dict[str, Any]:
        """The full JSON contract shared by the web API and Claude."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "provider": self.provider,
            "synced": self.synced,
            "sync_error": self.sync_error,
            "stale": self.is_stale,
            "draft": self.state.to_dict(),
            "on_the_clock": self.on_the_clock(),
            "best_available": self.best_available(limit=board_limit),
            "my_roster": self.my_roster(),
            "who_makes_it_back": self.who_makes_it_back(),
            "what_i_need": self.what_i_need(),
            "team_strength": self.team_strength(),
            "what_if": self.what_if(),
            "draft_environment": self.draft_environment(),
            "simulation": (
                {
                    "iterations": self.simulation.iterations,
                    "picks_simulated": self.simulation.picks_simulated,
                    "seed": self.simulation.seed,
                    "approximation": self.simulation.approximation_note,
                }
                if self.simulation else None
            ),
            "staleness": [
                {
                    "source": row.source,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    "age_hours": round(row.age_hours, 2) if row.age_hours is not None else None,
                    "stale": row.is_stale,
                }
                for row in self.recommendation.staleness
            ],
            "warnings": self.recommendation.warnings,
        }


# --- lineup ----------------------------------------------------------------------------


def lineup_slots(roster_config: Any) -> list[str]:
    """Ordered starting-lineup slot names for a league, then bench slots."""
    slots: list[str] = []
    for position in ("QB", "RB", "WR", "TE"):
        slots += [position] * getattr(roster_config, position.lower())
    for name, count in roster_config.flex_counts.items():
        slots += [name] * count
    for position in ("K", "DST"):
        slots += [position] * getattr(roster_config, position.lower())
    return slots


def fill_lineup(
    roster_config: Any,
    roster: RosterSnapshot | None,
    lookup: dict[str, dict[str, Any]],
) -> list[LineupSlot]:
    """Place our drafted players into starting slots, best players into dedicated slots.

    Dedicated slots are filled before flex ones, and within a position the players are
    assigned in draft order, so the lineup reflects what we actually have rather than an
    optimised projection. Anything left over goes to the bench.
    """
    slot_names = lineup_slots(roster_config)
    slots = [LineupSlot(slot=name) for name in slot_names]

    available: list[tuple[str, dict[str, Any]]] = []
    if roster:
        for key in roster.player_keys:
            info = lookup.get(key, {})
            available.append((key, info))

    used: set[str] = set()

    def take(eligible: tuple[str, ...]) -> tuple[str, dict[str, Any]] | None:
        for key, info in available:
            if key in used:
                continue
            if info.get("position") in eligible:
                used.add(key)
                return key, info
        return None

    # Dedicated slots first; a flex slot should not swallow our only tight end.
    for slot in slots:
        if slot.slot in FLEX_ELIGIBILITY:
            continue
        picked = take((slot.slot,))
        if picked:
            slot.player_key, info = picked
            slot.name, slot.position, slot.team = (
                info.get("name"), info.get("position"), info.get("team")
            )

    for slot in slots:
        if slot.slot not in FLEX_ELIGIBILITY or slot.filled:
            continue
        picked = take(FLEX_ELIGIBILITY[slot.slot])
        if picked:
            slot.player_key, info = picked
            slot.name, slot.position, slot.team = (
                info.get("name"), info.get("position"), info.get("team")
            )

    bench = []
    for key, info in available:
        if key in used:
            continue
        bench.append(
            LineupSlot(
                slot="BN", player_key=key, name=info.get("name"),
                position=info.get("position"), team=info.get("team"), is_bench=True,
            )
        )
    bench.sort(key=lambda s: POSITION_ORDER.get(s.position or "", 9))
    return slots + bench


# --- payload helpers -------------------------------------------------------------------


def _round(value: Any, digits: int = 1) -> float | None:
    return None if value is None else round(float(value), digits)


def _slot_payload(slot: LineupSlot) -> dict[str, Any]:
    return {
        "slot": slot.slot,
        "player_key": slot.player_key,
        "name": slot.name,
        "position": slot.position,
        "team": slot.team,
        "filled": slot.filled,
    }


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    survival = candidate.survival
    return {
        "player_key": candidate.player_key,
        "name": candidate.name,
        "position": candidate.position,
        "team": candidate.team,
        "bye_week": candidate.bye_week,
        "draft_now": _round(candidate.draft_now),
        "player_score": _round(candidate.player_score.value) if candidate.player_score else None,
        "value_score": _round(candidate.value_score.value) if candidate.value_score else None,
        "confidence": (
            _round(candidate.draft_now_score.confidence, 3)
            if candidate.draft_now_score else None
        ),
        "projected_points": _round(candidate.projected_points),
        "vbd": _round(candidate.vbd),
        "adp": _round(candidate.adp),
        "tier": candidate.tier,
        "probability_gone": _round(survival.probability_gone, 3) if survival else None,
        "probability_available": _round(survival.probability_available, 3) if survival else None,
        "two_pick_expected_value": candidate.two_pick_expected_value,
        "expected_next_pick_value": candidate.expected_next_pick_value,
    }


def _board_row(row: dict[str, Any], rank: int) -> dict[str, Any]:
    """One row of the BEST AVAILABLE table, with every offered column."""
    available = row.get("probability_available")
    adp = row.get("adp")
    return {
        "rank": rank,
        "player_key": row["player_key"],
        "name": row.get("player_name"),
        "position": row.get("position"),
        "team": row.get("team"),
        "bye_week": row.get("bye_week"),
        "draft_now": _round(row.get("draft_now_score")),
        "player_score": _round(row.get("player_score")),
        "value_score": _round(row.get("value_score")),
        "projection": _round(row.get("projected_points")),
        "vbd": _round(row.get("vbd")),
        "tier": row.get("tier"),
        "tier_rank": row.get("tier_rank"),
        "adp": _round(adp),
        "adp_value": _round(row.get("adp_delta")),
        "probability_gone": _round(1.0 - float(available), 3) if available is not None else None,
        "probability_available": _round(available, 3) if available is not None else None,
        "floor": _round(row.get("floor_points"), 0),
        "median": _round(row.get("median_points"), 0),
        "ceiling": _round(row.get("ceiling_points"), 0),
        "outcome": outcome_label(row),
        "scarcity": _round(row.get("scarcity_score")),
        "lineup_upgrade": _round(row.get("lineup_upgrade")),
        "confidence": _round(row.get("draft_now_confidence"), 3),
        "two_pick_expected_value": _round(row.get("two_pick_expected_value")),
    }


# --- the entry point --------------------------------------------------------------------


def analyze_current_pick(
    cfg: AppConfig,
    db: Database,
    draft_id: str | None = None,
    refresh: bool = True,
    limit: int = 12,
    simulate: bool = True,
    iterations: int | None = None,
) -> PickAnalysis:
    """Refresh the draft and produce the full analysis for the pick on the clock.

    Raises :class:`NoDraftError` when there is no draft to analyse and
    :class:`UnknownSlotError` when the draft slot is not set, because neither can be
    guessed and both have a specific fix the caller should surface.
    """
    from .draft.providers import ProviderError, provider_for

    state: DraftState | None = None
    synced = False
    sync_error: str | None = None
    provider_name = cfg.league.platform

    if refresh:
        try:
            provider = provider_for(cfg, db)
            target = draft_id or cfg.league.draft_id or db.get_meta("sleeper_draft_id")
            if target:
                state = provider.fetch_state(str(target))
                save_state(db, state)
                synced = True
                provider_name = provider.platform
            else:
                sync_error = "no draft id configured"
        except (ProviderError, Exception) as exc:  # noqa: BLE001 — never lose the pick
            sync_error = f"{type(exc).__name__}: {exc}"
            log.warning("live sync failed", extra={"error": sync_error})

    if state is None:
        state = load_state(db, draft_id)
    if state is None:
        raise NoDraftError(
            "No draft is available. Run `ff draft sync` for a live draft, or "
            "`ff draft mock` to practise."
        )
    if state.my_slot is None:
        raise UnknownSlotError(
            "Your draft slot is unknown, so the snake maths cannot run. Set "
            "`draft.slot` in config/league.yaml, or run `ff draft sync` once the draft "
            "order has been set."
        )

    result = recommend(
        db, cfg, state, limit=max(limit, 6), simulate=simulate, iterations=iterations
    )

    # Marginal value to *our* starting lineup. Needs the league and the roster, so it is
    # applied here rather than inside the generic board build.
    roster_snapshot = state.my_roster()
    roster_points: list[tuple[str, float]] = []
    if roster_snapshot and not result.board.frame.is_empty():
        # VBD, to match how lineup_upgrades values candidates.
        projections = dict(
            zip(result.board.frame["player_key"], result.board.frame["vbd"], strict=True)
        )
        for pick in state.picks:
            if pick.player_key in roster_snapshot.player_keys and pick.position:
                roster_points.append(
                    (pick.position, float(projections.get(pick.player_key) or 0.0))
                )
    result.frame = lineup_upgrades(cfg.league, roster_points, result.frame)

    # VBD for the whole universe, including players already off the board. Replacement
    # level comes from the board (derived, correctly, from the full pool) so a drafted
    # player is valued on exactly the same scale as an available one.
    all_vbd: dict[str, float] = {}
    try:
        from .analytics.projections import consensus_projections

        every = consensus_projections(db, cfg)
        if not every.is_empty():
            levels = {p: level.points for p, level in result.board.replacement.items()}
            for row in every.select("player_key", "position", "projected_points").iter_rows(
                named=True
            ):
                base = levels.get(row["position"])
                if base is not None and row["projected_points"] is not None:
                    all_vbd[row["player_key"]] = float(row["projected_points"]) - base
    except Exception as exc:  # noqa: BLE001 — team strength is a readout, never the pick
        log.warning("could not value drafted players", extra={"error": str(exc)})

    analysis = PickAnalysis(
        state=state,
        recommendation=result.recommendation,
        board=result.board,
        room=result.room,
        strategy=result.strategy,
        frame=result.frame,
        simulation=result.simulation,
        provider=provider_name,
        synced=synced,
        sync_error=sync_error,
        all_vbd=all_vbd,
    )
    analysis._league = cfg.league
    return analysis


def compare_picks(
    analysis: PickAnalysis, player_keys: list[str]
) -> dict[str, Any]:
    """Head-to-head: what happens to our next pick under each choice?

    Uses the same simulation the recommendation rests on, so the comparison and the
    recommendation can never disagree.
    """
    frame = analysis.frame
    lookup = dict(zip(frame["player_key"], frame["player_name"], strict=True))
    paths: list[dict[str, Any]] = []

    for key in player_keys:
        row = frame.filter(pl.col("player_key") == key)
        if row.is_empty():
            continue
        record = row.to_dicts()[0]
        value = analysis.simulation.two_pick.get(key) if analysis.simulation else None
        paths.append(
            {
                "player_key": key,
                "name": lookup.get(key, key),
                "position": record.get("position"),
                "team": record.get("team"),
                "draft_now": _round(record.get("draft_now_score")),
                "player_score": _round(record.get("player_score")),
                "probability_gone": (
                    _round(1.0 - float(record["probability_available"]), 3)
                    if record.get("probability_available") is not None else None
                ),
                "value_now": round(value.value_now, 1) if value else None,
                "expected_next_value": round(value.expected_next_value, 1) if value else None,
                "combined": round(value.combined, 1) if value else None,
                "likely_next_position": value.likely_next_position if value else None,
                "next_value_range": (
                    [round(value.next_value_low, 1), round(value.next_value_high, 1)]
                    if value else None
                ),
            }
        )

    priced = [p for p in paths if p["combined"] is not None]
    winner = max(priced, key=lambda p: p["combined"]) if priced else None
    margin = None
    if winner and len(priced) > 1:
        runner_up = sorted((p["combined"] for p in priced), reverse=True)[1]
        margin = round(winner["combined"] - runner_up, 1)

    return {
        "paths": paths,
        "winner": winner["player_key"] if winner else None,
        "winner_name": winner["name"] if winner else None,
        "margin": margin,
        "note": (
            analysis.simulation.approximation_note
            if analysis.simulation else "simulation unavailable; combined values omitted"
        ),
    }


class ServiceError(RuntimeError):
    """Base for errors a caller should show the user verbatim."""


class NoDraftError(ServiceError):
    """No draft state is available to analyse."""


class UnknownSlotError(ServiceError):
    """The draft slot is unknown, so snake maths cannot run."""
