"""Draft providers: the seam between a platform and the engine.

Everything downstream consumes :class:`~fantasy_draft.draft.state.DraftState`. A new
platform means one new class here and nothing else — no scoring code knows what a
Sleeper ID looks like.

Sleeper is the reference implementation and the only one that is wired up. Yahoo is
present as an explicit stub that raises with instructions, rather than as silence: the
config accepts ``platform: yahoo``, and a user who sets it deserves to be told exactly
what is missing instead of getting a confusing failure somewhere downstream.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import polars as pl

from ..config import AppConfig
from ..database import Database
from ..logging import get_logger
from ..models import DraftPick
from ..normalization.ids import dst_key
from ..normalization.players import normalize_name, normalize_position
from ..normalization.teams import canonical_team
from .snake import SnakeBoard
from .state import DraftState

log = get_logger(__name__)


@runtime_checkable
class DraftProvider(Protocol):
    """What the engine needs from any draft platform."""

    platform: str

    def fetch_state(self, draft_id: str) -> DraftState:
        """Return the current draft state, freshly fetched."""
        ...

    def resolve_pick(self, raw: dict[str, Any]) -> DraftPick:
        """Convert one platform pick record into our canonical form."""
        ...


class ProviderError(RuntimeError):
    """Raised with a message the caller should show the user verbatim."""


class ProviderNotConfigured(ProviderError):
    """The chosen platform exists but has not been set up on this machine."""


class ProviderNotImplemented(ProviderError):
    """The chosen platform has no working implementation yet."""


class PlayerKeyResolver:
    """Maps a platform's player IDs onto our ``player_key``.

    Resolution order for Sleeper, most to least reliable:

    1. The ffverse ID map (``player_ids.sleeper_id``) — an explicit, curated mapping.
    2. Sleeper's own ``gsis_id``, which it publishes for most active players.
    3. Team defences, whose Sleeper ID is the team abbreviation.
    4. Normalized name plus position, and only when unique.

    A pick we cannot resolve is **still recorded** — that player is off the board
    whether or not we know who he is — but it is counted in
    ``DraftState.unresolved_pick_count`` so the recommendation can say so.
    """

    def __init__(self, db: Database, platform_id_column: str = "sleeper_id") -> None:
        self.column = platform_id_column
        mapping = db.query(
            f"SELECT {platform_id_column} AS pid, player_key FROM player_ids "
            f"WHERE {platform_id_column} IS NOT NULL"
        )
        self.by_platform_id: dict[str, str] = dict(
            zip(mapping["pid"].cast(pl.Utf8), mapping["player_key"], strict=True)
        )
        gsis = db.query(
            "SELECT gsis_id, player_key FROM players WHERE gsis_id IS NOT NULL"
        )
        self.by_gsis: dict[str, str] = dict(
            zip(gsis["gsis_id"], gsis["player_key"], strict=True)
        )
        names = db.query("SELECT normalized_name, position, player_key FROM players")
        self.by_name: dict[str, list[str]] = {}
        for name, position, key in names.iter_rows():
            if name:
                self.by_name.setdefault(f"{name}|{position or ''}", []).append(key)

        self.unresolved: list[dict[str, Any]] = []

    def resolve(
        self,
        platform_id: str | None,
        name: str | None = None,
        position: str | None = None,
        team: str | None = None,
        gsis_id: str | None = None,
    ) -> str | None:
        canonical_position = normalize_position(position)
        if platform_id:
            key = self.by_platform_id.get(str(platform_id))
            if key:
                return key
        if gsis_id:
            key = self.by_gsis.get(str(gsis_id))
            if key:
                return key
        if canonical_position == "DST":
            # Sleeper uses the team abbreviation as the defence's player id.
            return dst_key(team or platform_id)
        if name:
            matches = self.by_name.get(f"{normalize_name(name)}|{canonical_position or ''}", [])
            if len(matches) == 1:
                return matches[0]
        self.unresolved.append(
            {"platform_id": platform_id, "name": name, "position": canonical_position}
        )
        return None


class SleeperDraftProvider:
    """Reads a live Sleeper draft into a :class:`DraftState`."""

    platform = "sleeper"

    def __init__(self, cfg: AppConfig, db: Database, client: Any | None = None) -> None:
        from ..data.sleeper import SleeperClient

        self.cfg = cfg
        self.db = db
        self.client = client or SleeperClient(cfg)
        self.resolver = PlayerKeyResolver(db, "sleeper_id")
        self._sleeper_players: dict[str, Any] | None = None

    # --- helpers ---------------------------------------------------------------------

    def _player_lookup(self) -> dict[str, Any]:
        if self._sleeper_players is None:
            self._sleeper_players = self.client.get_players()
        return self._sleeper_players

    def resolve_pick(self, raw: dict[str, Any]) -> DraftPick:
        """Convert one Sleeper pick record into our canonical :class:`DraftPick`."""
        metadata = raw.get("metadata") or {}
        platform_id = str(raw.get("player_id")) if raw.get("player_id") else None
        sleeper_player = self._player_lookup().get(platform_id or "", {}) or {}

        name = (
            metadata.get("first_name") and metadata.get("last_name")
            and f"{metadata['first_name']} {metadata['last_name']}"
        ) or sleeper_player.get("full_name") or metadata.get("player_name")
        position = metadata.get("position") or sleeper_player.get("position")
        team = metadata.get("team") or sleeper_player.get("team")

        player_key = self.resolver.resolve(
            platform_id=platform_id,
            name=name,
            position=position,
            team=team,
            gsis_id=sleeper_player.get("gsis_id"),
        )

        picked_at = None
        if raw.get("pick_no") and metadata.get("timestamp"):
            try:
                picked_at = datetime.fromtimestamp(int(metadata["timestamp"]) / 1000)
            except (TypeError, ValueError, OSError):
                picked_at = None

        return DraftPick(
            overall=int(raw["pick_no"]),
            round=int(raw.get("round") or 1),
            slot=int(raw.get("draft_slot") or 1),
            team_id=str(raw.get("picked_by") or raw.get("roster_id") or raw.get("draft_slot")),
            player_key=player_key,
            player_name=name,
            position=normalize_position(position),
            nfl_team=canonical_team(team),
            is_keeper=bool(raw.get("is_keeper")),
            picked_at=picked_at,
        )

    # --- the interface -----------------------------------------------------------------

    def fetch_state(self, draft_id: str) -> DraftState:
        from ..data.sleeper import infer_draft_settings

        draft = self.client.get_draft(draft_id)
        settings = infer_draft_settings(draft)
        raw_picks = self.client.get_draft_picks(draft_id)

        teams = settings["teams"] or self.cfg.league.teams
        board = SnakeBoard(
            teams=teams,
            rounds=settings["rounds"],
            draft_type=settings["type"],
            third_round_reversal=settings["third_round_reversal"],
        )

        picks = [self.resolve_pick(raw) for raw in raw_picks if raw.get("pick_no")]
        picks.sort(key=lambda p: p.overall)

        # draft_order maps a Sleeper user_id to their draft slot.
        draft_order: dict[str, int] = draft.get("draft_order") or {}
        slot_to_team = {int(slot): str(user) for user, slot in draft_order.items()}
        if not slot_to_team:
            # Some drafts (mock or pre-order) expose slots only through the picks.
            slot_to_team = {
                pick.slot: pick.team_id for pick in picks if pick.team_id
            }

        my_user_id = self._my_user_id()
        my_slot = draft_order.get(my_user_id) if my_user_id else None
        if my_slot is None and self.cfg.league.draft.slot:
            my_slot = self.cfg.league.draft.slot

        if self.resolver.unresolved:
            log.warning(
                "unresolved draft picks",
                extra={"count": len(self.resolver.unresolved), "draft_id": draft_id},
            )

        return DraftState(
            draft_id=str(draft_id),
            platform=self.platform,
            season=int(draft.get("season") or self.cfg.league.season),
            board=board,
            picks=picks,
            slot_to_team=slot_to_team,
            my_slot=int(my_slot) if my_slot else None,
            my_team_id=my_user_id,
            status=settings["status"] or "unknown",
            synced_at=datetime.now(),
        )

    def _my_user_id(self) -> str | None:
        stored = self.db.get_meta("sleeper_user_id")
        return stored or None


class YahooDraftProvider:
    """Yahoo Fantasy — interface only. Not implemented.

    This exists so that ``platform: yahoo`` in league.yaml produces a precise, actionable
    error instead of a confusing failure three layers down, and so the shape of the work
    is recorded where the next person will look for it.

    **What is actually required.** Yahoo's Fantasy Sports API is OAuth 2.0 only: there is
    no public read path equivalent to Sleeper's. Wiring it up needs, in order:

    1. A registered application at ``developer.yahoo.com``, which yields a client ID and
       secret. These belong in ``.env`` (already gitignored) and must never be committed.
    2. A three-legged OAuth consent flow, storing the refresh token locally.
    3. ``GET /fantasy/v2/users;use_login=1/games;game_keys=nfl/leagues`` to list leagues,
       then ``.../league/<key>/settings`` for scoring and roster slots, and
       ``.../league/<key>/draftresults`` for the picks.
    4. A ``resolve_pick`` that maps Yahoo's player keys onto our ``player_key`` via the
       ``yahoo_id`` column already present in ``player_ids`` from the ffverse map.

    Steps 3 and 4 are straightforward; step 1 needs credentials only the league owner can
    create, which is why this is a stub rather than an untested implementation. See
    HUMAN_TODO.md.
    """

    platform = "yahoo"

    #: What a caller should tell the user.
    MESSAGE = (
        "Yahoo is not implemented. Yahoo's Fantasy API requires OAuth 2.0 with a "
        "registered application — there is no public read path like Sleeper's — so it "
        "needs credentials only you can create. See HUMAN_TODO.md for the steps, or set "
        "`platform: sleeper` in config/league.yaml, which is fully working."
    )

    def __init__(self, cfg: AppConfig, db: Database) -> None:
        self.cfg = cfg
        self.db = db

    def fetch_state(self, draft_id: str) -> DraftState:
        raise ProviderNotImplemented(self.MESSAGE)

    def resolve_pick(self, raw: dict[str, Any]) -> DraftPick:
        raise ProviderNotImplemented(self.MESSAGE)


class ManualDraftProvider:
    """No live platform. The stored draft state is whatever was last loaded or mocked."""

    platform = "manual"

    def __init__(self, cfg: AppConfig, db: Database) -> None:
        self.cfg = cfg
        self.db = db

    def fetch_state(self, draft_id: str) -> DraftState:
        from .store import load_state

        state = load_state(self.db, draft_id)
        if state is None:
            raise ProviderNotConfigured(
                "No stored draft. Run `ff draft mock` to practise, or set "
                "`platform: sleeper` and run `ff draft sync` for a live draft."
            )
        return state

    def resolve_pick(self, raw: dict[str, Any]) -> DraftPick:
        return DraftPick(**raw)


class FixtureDraftProvider:
    """Replays a saved draft. Lets every test and simulation run without a network."""

    platform = "fixture"

    def __init__(self, state: DraftState) -> None:
        self.state = state

    def fetch_state(self, draft_id: str) -> DraftState:
        return self.state

    def resolve_pick(self, raw: dict[str, Any]) -> DraftPick:
        return DraftPick(**raw)


#: Platform name -> provider class. The recommendation engine never consults this; it
#: only ever sees a DraftState.
PROVIDERS: dict[str, type] = {
    "sleeper": SleeperDraftProvider,
    "yahoo": YahooDraftProvider,
    "manual": ManualDraftProvider,
}


def provider_for(cfg: AppConfig, db: Database) -> Any:
    """Build the provider for the configured platform."""
    platform = cfg.league.platform
    provider_class = PROVIDERS.get(platform)
    if provider_class is None:
        raise ProviderNotImplemented(
            f"Platform {platform!r} has no provider. Available: "
            f"{', '.join(sorted(PROVIDERS))}."
        )
    return provider_class(cfg, db)


def provider_status(cfg: AppConfig, db: Database) -> dict[str, Any]:
    """What the interface should show about the active data source."""
    platform = cfg.league.platform
    connected = False
    detail = ""

    if platform == "sleeper":
        user = db.get_meta("sleeper_username")
        league = db.get_meta("sleeper_league_id")
        connected = bool(user and league)
        detail = (
            f"connected as {user}, league {league}" if connected
            else "run `ff sleeper connect <username>` then `ff sleeper use-league <id>`"
        )
    elif platform == "yahoo":
        detail = YahooDraftProvider.MESSAGE
    elif platform == "manual":
        connected = True
        detail = "using the locally stored draft (mock or last sync)"

    return {
        "platform": platform,
        "connected": connected,
        "implemented": platform in {"sleeper", "manual"},
        "detail": detail,
        "available": sorted(PROVIDERS),
    }
