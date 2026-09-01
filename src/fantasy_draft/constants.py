"""Project-wide constants.

Deliberately free of tunable weights — anything a user might want to change lives in
``config/*.yaml``. This module holds only things that are structurally true about
football or about the shape of our data.
"""

from __future__ import annotations

from typing import Final

# --- Positions ---------------------------------------------------------------------

#: Offensive skill positions the draft engine scores. IDP is out of scope for MVP.
OFFENSE_POSITIONS: Final[tuple[str, ...]] = ("QB", "RB", "WR", "TE")

#: Positions that can occupy a roster slot, including special teams.
DRAFTABLE_POSITIONS: Final[tuple[str, ...]] = ("QB", "RB", "WR", "TE", "K", "DST")

#: Roster slot names that accept more than one position.
FLEX_ELIGIBILITY: Final[dict[str, tuple[str, ...]]] = {
    "FLEX": ("RB", "WR", "TE"),
    "WRRB_FLEX": ("RB", "WR"),
    "REC_FLEX": ("WR", "TE"),
    "SUPERFLEX": ("QB", "RB", "WR", "TE"),
    "OP": ("QB", "RB", "WR", "TE"),
}

#: Slot names that map one-to-one onto a position.
DEDICATED_SLOTS: Final[tuple[str, ...]] = ("QB", "RB", "WR", "TE", "K", "DST")

#: Canonical order for display.
POSITION_ORDER: Final[dict[str, int]] = {
    p: i for i, p in enumerate(("QB", "RB", "WR", "TE", "K", "DST"))
}

# --- Team abbreviations ------------------------------------------------------------

#: Historical/alternate team codes -> current nflverse abbreviation.
TEAM_ALIASES: Final[dict[str, str]] = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "JAC": "JAX",
    "LA": "LAR",
    "LVR": "LV",
    "OAK": "LV",
    "SD": "LAC",
    "SL": "LAR",
    "STL": "LAR",
    "WSH": "WAS",
    "WFT": "WAS",
    "SF0": "SF",
}

#: Sentinel for a player with no NFL team (free agent / retired).
FREE_AGENT_TEAM: Final[str] = "FA"

# --- Name normalization ------------------------------------------------------------

NAME_SUFFIXES: Final[frozenset[str]] = frozenset(
    {"jr", "sr", "ii", "iii", "iv", "v", "vi"}
)

#: Genuine nickname corrections, applied *after* normalization, mapping a normalized
#: alias to the normalized nflverse form.
#:
#: Empty on purpose. Measured against the live 2026 FantasyPros board, the general rule
#: in `normalization.players` (accent folding, punctuation stripping, suffix removal,
#: initial-run joining) matches 100% of the top 300 non-DST names with no lookups at
#: all. Every entry here is a place where two feeds are asserted to mean the same
#: person, so entries get added only when a real unmatched name proves one is needed --
#: never speculatively. Check `ff data unresolved-players` before adding to this table.
NAME_NICKNAMES: Final[dict[str, str]] = {}

# --- Data sources ------------------------------------------------------------------

SOURCE_NFLVERSE: Final[str] = "nflverse"
SOURCE_FANTASYPROS: Final[str] = "fantasypros"
SOURCE_SLEEPER: Final[str] = "sleeper"
SOURCE_MANUAL: Final[str] = "manual"

#: Season the engine is drafting for.
DEFAULT_SEASON: Final[int] = 2026

#: How many prior seasons of stats we ingest by default.
HISTORY_SEASONS: Final[int] = 4

#: Regular-season week count from 2021 onward.
REGULAR_SEASON_WEEKS: Final[int] = 18


# --- Bench construction ------------------------------------------------------------

#: Relative pull each position exerts on bench slots, per starting slot it occupies.
#:
#: A bench spot is insurance plus upside, and the insurance half is driven by how often
#: you actually need a replacement: running backs miss time and lose jobs far more than
#: quarterbacks, so a bench skews toward the positions you start most *and* lose most.
#: One backup quarterback is enough in a single-QB league and none is defensible; two is
#: waste.
#:
#: These are priors, not measurements. Once historical injury and role-change data is
#: joined per position they should be estimated rather than assumed.
BENCH_ATTRITION: Final[dict[str, float]] = {
    "RB": 1.60,
    "WR": 1.15,
    "TE": 0.70,
    "QB": 0.45,
}

#: How many players past your remaining starters count as realistic bench candidates.
#:
#: Ranking the whole tail by ceiling surfaces camp bodies with a median of 23 and a
#: ceiling of 186. That width is uncertainty about an unknown player, not upside, and the
#: outcome model says so through its confidence — but a raw ceiling sort ignores that. A
#: bounded window keeps the suggestion to players you would plausibly spend a pick on.
BENCH_WINDOW: Final[int] = 24
