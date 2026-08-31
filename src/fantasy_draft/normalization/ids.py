"""Canonical player identity.

Every table in this project keys on ``player_key``. Getting this wrong is the most
expensive bug available to us: a bad merge silently attributes one player's usage to
another, and a bad miss drops a draftable player off the board entirely.

Resolution order, most to least trustworthy:

1. **Team defences** -> ``DST-<TEAM>``. They have no gsis_id and must not be name-matched
   against people.
2. **gsis_id** -> used directly. nflverse's canonical ID, stable across seasons.
3. **A platform ID** (sleeper/espn/yahoo/fantasypros/pfr/mfl) resolved through the
   ffverse map to a gsis_id.
4. **Normalized name + position**, and only when that pair is unique on both sides.
5. Otherwise a **synthetic key** (``fp-12345``, ``slp-4034``, ``nm-<hash>``) so the player
   still reaches the board, plus a row in ``unresolved_players`` explaining why.

Step 4 never uses fuzzy matching. Two similarly named players must fail to match and be
reported rather than merged.
"""

from __future__ import annotations

import hashlib

import polars as pl

from ..logging import get_logger
from ..models import UnresolvedPlayer
from .players import normalize_name, normalize_position
from .teams import canonical_team

log = get_logger(__name__)

#: Vendor ID columns carried through the identity map, in preference order.
PLATFORM_ID_COLUMNS: tuple[str, ...] = (
    "gsis_id", "sleeper_id", "espn_id", "yahoo_id", "fantasypros_id", "pfr_id",
    "mfl_id", "sportradar_id", "pff_id", "cbs_id", "rotowire_id", "ktc_id",
    "fantasy_data_id",
)

#: Prefix used when we mint a key from a platform ID because no gsis_id exists.
SYNTHETIC_PREFIXES: dict[str, str] = {
    "sleeper_id": "slp",
    "fantasypros_id": "fp",
    "espn_id": "esp",
    "yahoo_id": "yh",
    "mfl_id": "mfl",
    "pfr_id": "pfr",
}


def dst_key(team: str | None) -> str:
    """Canonical key for a team defence."""
    return f"DST-{canonical_team(team)}"


def synthetic_key(source_column: str, source_id: str) -> str:
    """Stable key minted from a vendor ID when no gsis_id is available."""
    prefix = SYNTHETIC_PREFIXES.get(source_column, source_column.replace("_id", ""))
    return f"{prefix}-{source_id}"


def name_hash_key(name: str, position: str | None) -> str:
    """Last-resort key derived from the name itself. Deterministic across runs."""
    payload = f"{normalize_name(name)}|{normalize_position(position) or ''}"
    digest = hashlib.sha1(payload.encode()).hexdigest()[:10]
    return f"nm-{digest}"


class IdentityMap:
    """Resolves vendor rows to ``player_key``.

    Built once per refresh from the nflverse player list plus the ffverse ID map, then
    queried by every other adapter.
    """

    def __init__(self, frame: pl.DataFrame) -> None:
        """``frame`` must carry ``player_key``, ``normalized_name``, ``position``, and
        whichever of :data:`PLATFORM_ID_COLUMNS` are available."""
        self.frame = frame
        self.unresolved: list[UnresolvedPlayer] = []
        self._by_id: dict[str, dict[str, str]] = {}
        self._by_name: dict[str, list[str]] = {}
        self._by_name_only: dict[str, list[str]] = {}
        self._build_indexes()

    # --- construction --------------------------------------------------------------

    def _build_indexes(self) -> None:
        columns = set(self.frame.columns)
        for column in PLATFORM_ID_COLUMNS:
            if column not in columns:
                continue
            subset = self.frame.select("player_key", column).filter(
                pl.col(column).is_not_null() & (pl.col(column).cast(pl.Utf8) != "")
            )
            index: dict[str, str] = {}
            collisions = 0
            for key, value in zip(
                subset["player_key"].to_list(),
                subset[column].cast(pl.Utf8).to_list(),
                strict=True,
            ):
                if value in index and index[value] != key:
                    # The source map genuinely contains a few duplicated IDs. First
                    # write wins deterministically (the frame is sorted upstream) and we
                    # count the conflict rather than silently overwriting.
                    collisions += 1
                    continue
                index[value] = key
            self._by_id[column] = index
            if collisions:
                log.warning(
                    "duplicate vendor ids ignored",
                    extra={"column": column, "count": collisions},
                )

        name_subset = self.frame.select("player_key", "normalized_name", "position")
        for key, name, position in zip(
            name_subset["player_key"].to_list(),
            name_subset["normalized_name"].to_list(),
            name_subset["position"].to_list(),
            strict=True,
        ):
            if not name:
                continue
            # Normalize on the way in as well as on the way out, so an index built from
            # a raw vendor frame agrees with a lookup that canonicalizes.
            canonical = normalize_position(position)
            self._by_name.setdefault(f"{name}|{canonical or ''}", []).append(key)
            self._by_name_only.setdefault(name, []).append(key)

    # --- lookup ---------------------------------------------------------------------

    def by_id(self, column: str, value: str | int | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan", "null"}:
            return None
        return self._by_id.get(column, {}).get(text)

    def by_name(self, name: str, position: str | None) -> tuple[str | None, list[str]]:
        """Return ``(player_key, candidates)``.

        ``player_key`` is None when there is no match or more than one — the caller
        decides what to do, and ``candidates`` explains the ambiguity.
        """
        normalized = normalize_name(name)
        canonical_position = normalize_position(position)
        everyone = self._by_name_only.get(normalized, [])

        if canonical_position is not None:
            exact = self._by_name.get(f"{normalized}|{canonical_position}", [])
            if len(exact) == 1:
                return exact[0], exact
            if exact:
                # Two players with the same name AND position: genuinely ambiguous.
                return None, exact
            # Feeds disagree about RB/FB and WR/TE more than they disagree about names,
            # so retry without the position -- but only accept a unique answer.
            if len(everyone) == 1:
                return everyone[0], everyone
            return None, everyone

        # No position supplied: only a globally unique name can be trusted.
        if len(everyone) == 1:
            return everyone[0], everyone
        return None, everyone

    def resolve(
        self,
        source: str,
        name: str,
        position: str | None = None,
        team: str | None = None,
        ids: dict[str, str | int | None] | None = None,
        allow_synthetic: bool = True,
    ) -> str | None:
        """Resolve one vendor row to a ``player_key``, recording failures.

        Returns None only when ``allow_synthetic`` is False and nothing matched.
        """
        canonical_position = normalize_position(position)

        if canonical_position == "DST":
            return dst_key(team)

        ids = ids or {}
        for column in PLATFORM_ID_COLUMNS:
            key = self.by_id(column, ids.get(column))
            if key is not None:
                return key

        key, candidates = self.by_name(name, position)
        if key is not None:
            return key

        reason = "ambiguous" if candidates else "no_match"
        source_id = next(
            (str(v) for c, v in ids.items() if v not in (None, "") for _ in [c]), None
        )
        self.unresolved.append(
            UnresolvedPlayer(
                source=source,
                source_id=source_id,
                raw_name=name,
                normalized_name=normalize_name(name),
                position=canonical_position,
                team=canonical_team(team) if team else None,
                reason=reason,
                candidates=candidates[:5],
            )
        )
        if not allow_synthetic:
            return None

        for column in ("fantasypros_id", "sleeper_id", "espn_id", "yahoo_id", "mfl_id"):
            value = ids.get(column)
            if value not in (None, ""):
                return synthetic_key(column, str(value))
        return name_hash_key(name, position)

    # --- reporting -------------------------------------------------------------------

    def unresolved_frame(self) -> pl.DataFrame:
        """Unresolved rows shaped for the ``unresolved_players`` table."""
        if not self.unresolved:
            return pl.DataFrame(
                schema={
                    "source": pl.Utf8, "source_id": pl.Utf8, "raw_name": pl.Utf8,
                    "normalized_name": pl.Utf8, "position": pl.Utf8, "team": pl.Utf8,
                    "reason": pl.Utf8, "candidates": pl.Utf8, "seen_at": pl.Datetime,
                }
            )
        frame = pl.DataFrame(
            [
                {
                    "source": u.source,
                    "source_id": u.source_id,
                    "raw_name": u.raw_name,
                    "normalized_name": u.normalized_name,
                    "position": u.position,
                    "team": u.team,
                    "reason": u.reason,
                    "candidates": ", ".join(u.candidates) or None,
                    "seen_at": u.seen_at,
                }
                for u in self.unresolved
            ]
        )
        # One player missing from five ranking pages is one identity problem.
        return frame.unique(subset=["source", "raw_name", "position"], keep="first").sort(
            ["source", "raw_name"]
        )

    def __len__(self) -> int:
        return self.frame.height


def resolve_column(
    frame: pl.DataFrame,
    identity: IdentityMap,
    source: str,
    name_column: str,
    position_column: str | None = None,
    team_column: str | None = None,
    id_columns: dict[str, str] | None = None,
    allow_synthetic: bool = True,
) -> pl.DataFrame:
    """Add a ``player_key`` column to ``frame`` by resolving each row.

    ``id_columns`` maps our canonical ID name (e.g. ``"sleeper_id"``) to the column in
    ``frame`` that holds it.
    """
    if frame.is_empty():
        return frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias("player_key"))

    id_columns = id_columns or {}
    names = frame[name_column].to_list()
    positions = (
        frame[position_column].to_list() if position_column else [None] * frame.height
    )
    teams = frame[team_column].to_list() if team_column else [None] * frame.height
    id_values = {
        canonical: frame[column].cast(pl.Utf8).to_list()
        for canonical, column in id_columns.items()
        if column in frame.columns
    }

    keys: list[str | None] = []
    for i in range(frame.height):
        keys.append(
            identity.resolve(
                source=source,
                name=names[i] or "",
                position=positions[i],
                team=teams[i],
                ids={canonical: values[i] for canonical, values in id_values.items()},
                allow_synthetic=allow_synthetic,
            )
        )
    return frame.with_columns(pl.Series("player_key", keys, dtype=pl.Utf8))
