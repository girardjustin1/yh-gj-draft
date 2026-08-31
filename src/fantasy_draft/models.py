"""Core domain models shared across ingestion, analytics, draft state, and scoring.

These are transport/contract objects. Bulk numeric work happens in Polars DataFrames;
these types exist where a single record crosses a module boundary or gets serialized
for the CLI, the API, or Claude.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

# --- Enums --------------------------------------------------------------------------


class Position(StrEnum):
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DST = "DST"


class Strategy(StrEnum):
    BALANCED = "balanced"
    HERO_RB = "hero_rb"
    ROBUST_RB = "robust_rb"
    ZERO_RB = "zero_rb"


class Confidence(StrEnum):
    """Coarse confidence bucket, for when a float would imply false precision."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


# --- Identity -----------------------------------------------------------------------


class PlayerIdentity(BaseModel):
    """Cross-platform identity for one player, keyed by our internal ``player_key``."""

    model_config = ConfigDict(extra="forbid")

    player_key: str = Field(description="Canonical internal key; stable across refreshes.")
    full_name: str
    normalized_name: str
    position: str
    team: str | None = None

    gsis_id: str | None = None
    sleeper_id: str | None = None
    espn_id: str | None = None
    yahoo_id: str | None = None
    fantasypros_id: str | None = None
    pfr_id: str | None = None
    mfl_id: str | None = None
    sportradar_id: str | None = None

    birth_date: str | None = None
    rookie_season: int | None = None
    draft_year: int | None = None

    def id_for(self, platform: str) -> str | None:
        return getattr(self, f"{platform}_id", None)


class UnresolvedPlayer(BaseModel):
    """A record we could not confidently map. Never silently merged — always logged."""

    model_config = ConfigDict(extra="forbid")

    source: str
    source_id: str | None
    raw_name: str
    normalized_name: str
    position: str | None
    team: str | None
    reason: Literal["no_match", "ambiguous", "position_conflict", "team_conflict"]
    candidates: list[str] = Field(default_factory=list)
    seen_at: datetime = Field(default_factory=datetime.now)


# --- Component scores ---------------------------------------------------------------


class ComponentScore(BaseModel):
    """One scoring component.

    Every component carries its raw value alongside the normalized 0-100 score, plus a
    confidence so downstream consumers can tell "average" from "we don't know".
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    raw_value: float | None = None
    normalized: float = Field(50.0, ge=0, le=100)
    confidence: float = Field(1.0, ge=0, le=1)
    method: str = Field("", description="How raw_value became normalized.")
    source: str | None = None
    source_updated_at: datetime | None = None
    notes: str = ""

    @property
    def is_known(self) -> bool:
        return self.raw_value is not None and self.confidence > 0


class ScoreBundle(BaseModel):
    """A named composite score plus the components that produced it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float = Field(ge=0, le=100)
    confidence: float = Field(1.0, ge=0, le=1)
    components: dict[str, ComponentScore] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)

    def contribution(self, component: str) -> float:
        """Points this component contributed to ``value``."""
        comp = self.components.get(component)
        if comp is None:
            return 0.0
        return comp.normalized * self.weights.get(component, 0.0)


# --- Draft --------------------------------------------------------------------------


class DraftPick(BaseModel):
    """One selection, as reconstructed from a platform or a fixture."""

    model_config = ConfigDict(extra="forbid")

    overall: int = Field(ge=1)
    round: int = Field(ge=1)
    slot: int = Field(ge=1, description="1-indexed draft slot that made this pick.")
    team_id: str
    player_key: str | None = None
    player_name: str | None = None
    position: str | None = None
    nfl_team: str | None = None
    is_keeper: bool = False
    picked_at: datetime | None = None


class RosterSnapshot(BaseModel):
    """One fantasy team's roster at a point in the draft."""

    model_config = ConfigDict(extra="forbid")

    team_id: str
    slot: int = Field(ge=1)
    is_me: bool = False
    player_keys: list[str] = Field(default_factory=list)
    positions: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def position_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pos in self.positions:
            counts[pos] = counts.get(pos, 0) + 1
        return counts

    @property
    def size(self) -> int:
        return len(self.player_keys)


class PositionNeed(BaseModel):
    """Estimated probability an opponent takes each position with their next pick."""

    model_config = ConfigDict(extra="forbid")

    team_id: str
    slot: int
    pick_overall: int
    probabilities: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""


# --- Recommendation -----------------------------------------------------------------


class SurvivalEstimate(BaseModel):
    """Probability a player is still on the board at our next pick."""

    model_config = ConfigDict(extra="forbid")

    player_key: str
    probability_available: float = Field(ge=0, le=1)
    probability_gone: float = Field(ge=0, le=1)
    method: Literal["adp_normal", "monte_carlo", "blended", "unknown"] = "unknown"
    iterations: int | None = None
    confidence: float = Field(1.0, ge=0, le=1)

    @classmethod
    def unknown(cls, player_key: str) -> SurvivalEstimate:
        return cls(
            player_key=player_key,
            probability_available=0.5,
            probability_gone=0.5,
            method="unknown",
            confidence=0.0,
        )


class Candidate(BaseModel):
    """A ranked draft option with everything needed to explain the recommendation."""

    model_config = ConfigDict(extra="forbid")

    player_key: str
    name: str
    position: str
    team: str | None = None
    bye_week: int | None = None

    projected_points: float | None = None
    vbd: float | None = None
    adp: float | None = None
    adp_sd: float | None = None
    positional_rank: int | None = None
    tier: int | None = None
    tier_rank: int | None = None
    points_to_next_player: float | None = None
    points_to_next_tier: float | None = None

    player_score: ScoreBundle | None = None
    value_score: ScoreBundle | None = None
    draft_now_score: ScoreBundle | None = None

    survival: SurvivalEstimate | None = None
    two_pick_expected_value: float | None = None
    expected_next_pick_value: float | None = None

    @property
    def draft_now(self) -> float:
        return self.draft_now_score.value if self.draft_now_score else 0.0

    @property
    def adp_delta(self) -> float | None:
        """Positive means he has fallen past his ADP (value); negative means a reach."""
        return None if self.adp is None else self.adp

    def summary(self) -> dict[str, Any]:
        """Compact dict for JSON/tool output — no nested component detail."""
        return {
            "player_key": self.player_key,
            "name": self.name,
            "position": self.position,
            "team": self.team,
            "draft_now": round(self.draft_now, 1),
            "player_score": round(self.player_score.value, 1) if self.player_score else None,
            "value_score": round(self.value_score.value, 1) if self.value_score else None,
            "projected_points": self.projected_points,
            "vbd": self.vbd,
            "adp": self.adp,
            "tier": self.tier,
            "probability_gone": (
                round(self.survival.probability_gone, 3) if self.survival else None
            ),
            "two_pick_ev": self.two_pick_expected_value,
        }


class DataFreshness(BaseModel):
    """One row of ``ff data status``; also attached to every recommendation."""

    model_config = ConfigDict(extra="forbid")

    source: str
    updated_at: datetime | None = None
    rows: int | None = None
    max_age_hours: float = 24.0
    ok: bool = False
    error: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def age_hours(self) -> float | None:
        if self.updated_at is None:
            return None
        return (datetime.now() - self.updated_at).total_seconds() / 3600.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_stale(self) -> bool:
        age = self.age_hours
        return True if age is None else age > self.max_age_hours


class Recommendation(BaseModel):
    """The answer to "who should I draft?", with its reasoning and its caveats."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime = Field(default_factory=datetime.now)
    pick_label: str = Field(description='e.g. "4.06"')
    overall_pick: int
    next_pick_overall: int | None = None
    picks_until_next: int | None = None

    primary: Candidate | None = None
    alternatives: list[Candidate] = Field(default_factory=list)
    board: list[Candidate] = Field(default_factory=list)

    strategy: Strategy = Strategy.BALANCED
    strategy_confidence: float = Field(0.5, ge=0, le=1)
    alternative_strategy: Strategy | None = None
    strategy_reason: str = ""

    position_demand: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(0.5, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    staleness: list[DataFreshness] = Field(default_factory=list)
    explanation: str = ""
