"""Player name normalization.

Names are a *fallback* identity mechanism — stable IDs always win. But feeds disagree
("D.J. Moore" / "DJ Moore", "Kenneth Walker III" / "Kenneth Walker"), so we need a
normal form that collapses cosmetic differences without collapsing distinct people.

The normal form is deliberately conservative:

1. Unicode is folded to ASCII and lowercased.
2. Punctuation (apostrophes, periods, hyphens, commas) becomes whitespace.
3. Generational suffixes (jr, sr, ii..vi) are dropped.
4. Runs of single-letter tokens are joined, so ``d j moore`` and ``dj moore`` agree.
   This handles every A.J./D.K./T.J./C.J. case without a lookup table.
5. A small explicit nickname table handles the rest.

What it deliberately does NOT do: fuzzy/edit-distance matching. Two different players
with similar names must fail to match and be reported, never merged. See
:mod:`fantasy_draft.normalization.ids`.
"""

from __future__ import annotations

import re
import unicodedata

from ..constants import NAME_NICKNAMES, NAME_SUFFIXES

_PUNCT = re.compile(r"[.'`’,\-/\\]+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")

#: Team-defence naming variants seen across feeds.
_DST_TOKENS = frozenset({"dst", "def", "defense", "d/st", "dfs"})


def strip_accents(text: str) -> str:
    """Fold accented characters to ASCII: ``Kraft`` handling for e.g. ``Aristéo``."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _join_initial_runs(tokens: list[str]) -> list[str]:
    """Merge consecutive single-character tokens: ``[d, j, moore] -> [dj, moore]``."""
    out: list[str] = []
    run: list[str] = []
    for token in tokens:
        if len(token) == 1 and token.isalpha():
            run.append(token)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(token)
    if run:
        out.append("".join(run))
    return out


def normalize_name(name: str | None) -> str:
    """Return the canonical comparison form of a player name.

    >>> normalize_name("D.J. Moore")
    'dj moore'
    >>> normalize_name("Kenneth Walker III")
    'kenneth walker'
    >>> normalize_name("Ja'Marr Chase")
    'jamarr chase'
    >>> normalize_name("Amon-Ra St. Brown")
    'amon ra st brown'
    """
    if not name:
        return ""
    text = strip_accents(str(name)).lower()
    # Apostrophes close up (Ja'Marr -> jamarr); other punctuation becomes a space.
    text = text.replace("'", "").replace("’", "")
    text = _PUNCT.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    tokens = _WS.sub(" ", text).strip().split()

    # Drop suffixes, but never the entire name (a player literally named "Vi").
    trimmed = [t for t in tokens if t not in NAME_SUFFIXES]
    tokens = trimmed or tokens

    tokens = _join_initial_runs(tokens)
    normalized = " ".join(tokens)
    return NAME_NICKNAMES.get(normalized, normalized)


def normalize_position(position: str | None) -> str | None:
    """Canonicalize a position label to one of QB/RB/WR/TE/K/DST, or None.

    >>> normalize_position("PK")
    'K'
    >>> normalize_position("D/ST")
    'DST'
    >>> normalize_position("HB")
    'RB'
    >>> normalize_position("LB")
    'LB'

    Non-fantasy positions are preserved rather than nulled: they carry real
    disambiguating information (there is a Josh Allen at QB and another at LB). Deciding
    which positions are *draftable* is the ingest layer's job, via
    ``constants.DRAFTABLE_POSITIONS``.
    """
    if not position:
        return None
    raw = str(position).strip().lower()
    if raw in _DST_TOKENS or raw.replace(" ", "") in _DST_TOKENS:
        return "DST"
    cleaned = _NON_ALNUM.sub("", raw)
    mapping = {
        "qb": "QB",
        "rb": "RB", "hb": "RB", "fb": "RB",
        "wr": "WR",
        "te": "TE",
        "k": "K", "pk": "K",
        "dst": "DST", "def": "DST",
    }
    if cleaned in mapping:
        return mapping[cleaned]
    return cleaned.upper() if cleaned else None


def is_team_defense(name: str | None, position: str | None) -> bool:
    """True when a row describes a team defence rather than an individual."""
    return normalize_position(position) == "DST"


def split_name(name: str) -> tuple[str, str]:
    """Best-effort ``(first, last)`` split of a normalized name, for blocking."""
    tokens = normalize_name(name).split()
    if not tokens:
        return "", ""
    if len(tokens) == 1:
        return "", tokens[0]
    return tokens[0], tokens[-1]


def name_key(name: str | None, position: str | None) -> str:
    """Blocking key for fallback matching: normalized name plus canonical position."""
    return f"{normalize_name(name)}|{normalize_position(position) or ''}"
