"""Marginal lineup value: how much does this player actually improve my starting eleven?

The honest answer to "what do I still need?".

VBD asks how good a player is against a freely available replacement. That is the right
question for the draft as a whole and the wrong one for *my* roster in isolation. If I
already start two elite running backs and a flex, a fourth back adds almost nothing to
the points I will actually score — he sits on my bench. My first tight end adds his whole
projection, because the slot is currently worth zero.

So we compute, for each candidate, the value he would add to my *best legal starting
lineup*:

    lineup_upgrade(player) = lineup(with him) − lineup(without him)

Assignment respects dedicated and flex eligibility, so a receiver can take a FLEX slot but
not a QB one, and a superflex slot accepts a quarterback.

**Measured in VBD, not raw points.** This matters enough to be worth stating: an empty QB
slot is currently worth zero, so filling it with a 340-point quarterback looks like a
+340 upgrade, against +170 for the best available receiver. Read in raw points the metric
says "always take the quarterback" — which is precisely the illusion value-based drafting
exists to dispel, since the QB you could take three rounds later is nearly as good. Summed
over the lineup, VBD says how much better my starters are than an all-replacement lineup,
and the marginal version of that *is* comparable across positions.

**This is reported, not enforced.** It deliberately does not override the Draft Now score,
because early in a draft almost every pick is a bench player by this measure and chasing
lineup upgrade would produce exactly the positional reaches the engine exists to avoid.
Its job is to answer "what am I short of, and what is the best player who fixes it?" — a
question the human is better placed to weigh than a weight in a YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..config import LeagueConfig
from ..constants import FLEX_ELIGIBILITY, OFFENSE_POSITIONS


@dataclass(frozen=True, slots=True)
class NeedEntry:
    """The best available answer to one unfilled starting slot."""

    slot: str
    eligible: tuple[str, ...]
    filled: bool
    best_player_key: str | None = None
    best_name: str | None = None
    best_position: str | None = None
    best_team: str | None = None
    draft_now: float | None = None
    lineup_upgrade: float | None = None
    probability_gone: float | None = None
    tier: int | None = None


def _slot_names(league: LeagueConfig) -> list[str]:
    slots: list[str] = []
    for position in ("QB", "RB", "WR", "TE"):
        slots += [position] * getattr(league.roster, position.lower())
    for name, count in league.roster.flex_counts.items():
        slots += [name] * count
    return slots


def _eligible_for(slot: str) -> tuple[str, ...]:
    return FLEX_ELIGIBILITY.get(slot, (slot,))


def best_lineup_points(
    league: LeagueConfig, players: list[tuple[str, float]]
) -> float:
    """Total value of the best legal starting lineup from ``players``.

    ``players`` is ``(position, value)``, where value is VBD. Dedicated slots are filled
    before flex ones, best players first: a flex slot must never swallow the only player
    eligible for a dedicated slot, and filling dedicated first guarantees it cannot.
    """
    slots = _slot_names(league)
    dedicated = [s for s in slots if s not in FLEX_ELIGIBILITY]
    flex = [s for s in slots if s in FLEX_ELIGIBILITY]

    pool = sorted(players, key=lambda p: -p[1])
    used: set[int] = set()
    total = 0.0

    for slot in dedicated + flex:
        eligible = _eligible_for(slot)
        for index, (position, points) in enumerate(pool):
            if index in used or position not in eligible:
                continue
            used.add(index)
            total += points
            break
    return total


def lineup_upgrades(
    league: LeagueConfig,
    roster_players: list[tuple[str, float]],
    board: pl.DataFrame,
    limit: int = 200,
) -> pl.DataFrame:
    """Add ``lineup_upgrade`` to the board: points added to our best starting lineup."""
    if board.width == 0:
        # A frame with no columns at all. `with_columns(lit(...))` would broadcast the
        # literal and turn "no board" into a phantom one-row board.
        return board
    if board.is_empty() or "position" not in board.columns:
        return board.with_columns(pl.lit(None, dtype=pl.Float64).alias("lineup_upgrade"))

    baseline = best_lineup_points(league, roster_players)
    head = board.head(limit)
    value_column = "vbd" if "vbd" in board.columns else "projected_points"

    upgrades: dict[str, float] = {}
    for row in head.select("player_key", "position", value_column).iter_rows(named=True):
        value = row[value_column]
        if value is None or row["position"] not in OFFENSE_POSITIONS:
            continue
        with_him = best_lineup_points(
            league, [*roster_players, (row["position"], float(value))]
        )
        upgrades[row["player_key"]] = max(0.0, with_him - baseline)

    return board.with_columns(
        pl.col("player_key")
        .replace_strict(upgrades, default=None)
        .cast(pl.Float64)
        .alias("lineup_upgrade")
    )


def positional_needs(
    league: LeagueConfig,
    roster_players: list[tuple[str, float]],
    filled_slots: dict[str, bool],
    board: pl.DataFrame,
) -> list[NeedEntry]:
    """Best available player for each starting slot, unfilled slots first.

    Answers, concretely: *what am I short of, and who is the best player who fixes it?*
    """
    entries: list[NeedEntry] = []
    seen: set[str] = set()
    # An empty or partial board (nothing ingested yet, or every signal failed) must still
    # produce the slot list — knowing you have no tight end is useful even when we cannot
    # tell you who the best one is.
    usable = not board.is_empty() and "position" in board.columns

    for slot in _slot_names(league):
        # One entry per distinct slot type; two RB slots do not need two rows.
        if slot in seen:
            continue
        seen.add(slot)
        eligible = _eligible_for(slot)

        if not usable:
            entries.append(
                NeedEntry(slot=slot, eligible=eligible, filled=filled_slots.get(slot, False))
            )
            continue

        pool = board.filter(pl.col("position").is_in(list(eligible)))
        if "lineup_upgrade" in pool.columns:
            pool = pool.sort(
                ["draft_now_score"], descending=True, nulls_last=True
            )
        best = pool.head(1)
        if best.is_empty():
            entries.append(
                NeedEntry(slot=slot, eligible=eligible, filled=filled_slots.get(slot, False))
            )
            continue

        row = best.to_dicts()[0]
        available = row.get("probability_available")
        entries.append(
            NeedEntry(
                slot=slot,
                eligible=eligible,
                filled=filled_slots.get(slot, False),
                best_player_key=row["player_key"],
                best_name=row.get("player_name"),
                best_position=row.get("position"),
                best_team=row.get("team"),
                draft_now=(
                    round(float(row["draft_now_score"]), 1)
                    if row.get("draft_now_score") is not None else None
                ),
                lineup_upgrade=(
                    round(float(row["lineup_upgrade"]), 1)
                    if row.get("lineup_upgrade") is not None else None
                ),
                probability_gone=(
                    round(1.0 - float(available), 3) if available is not None else None
                ),
                tier=row.get("tier"),
            )
        )

    # Unfilled slots first — that is what the question is about.
    return sorted(entries, key=lambda e: (e.filled, -(e.draft_now or 0)))
