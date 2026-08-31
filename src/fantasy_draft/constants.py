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

#: Nicknames seen in fantasy feeds -> the name used by nflverse. Only unambiguous
#: entries belong here; anything uncertain must fail to match and be logged instead.
NAME_NICKNAMES: Final[dict[str, str]] = {
    "kenneth walker": "kenneth walker iii",
    "mike thomas": "michael thomas",
    "will fuller": "william fuller",
    "josh palmer": "joshua palmer",
    "gabe davis": "gabriel davis",
    "chig okonkwo": "chigoziem okonkwo",
    "dj moore": "d j moore",
    "aj brown": "a j brown",
    "jk dobbins": "j k dobbins",
    "cj stroud": "c j stroud",
    "tj hockenson": "t j hockenson",
    "dk metcalf": "d k metcalf",
    "marquise brown": "hollywood brown",
}

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
