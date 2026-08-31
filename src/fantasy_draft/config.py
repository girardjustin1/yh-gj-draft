"""Configuration: paths, league settings, scoring weights, data sources.

Every tunable number in this project lives in ``config/*.yaml`` and is parsed here into
validated Pydantic models. Nothing downstream should hard-code a weight, a replacement
level, or a roster assumption — it should read it off these objects.
"""

from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from .constants import DEFAULT_SEASON, FLEX_ELIGIBILITY, OFFENSE_POSITIONS

# --- Paths --------------------------------------------------------------------------


def _find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` looking for a pyproject.toml; fall back to cwd."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current


class Paths(BaseModel):
    """Filesystem layout. Overridable via ``FF_DATA_DIR`` / ``FF_CONFIG_DIR`` / ``FF_DB_PATH``."""

    model_config = ConfigDict(frozen=True)

    root: Path
    config_dir: Path
    data_dir: Path
    db_path: Path

    @classmethod
    def resolve(cls, root: Path | None = None) -> Paths:
        base = _find_project_root(root)
        config_dir = Path(os.environ.get("FF_CONFIG_DIR", base / "config"))
        data_dir = Path(os.environ.get("FF_DATA_DIR", base / "data"))
        db_path = Path(os.environ.get("FF_DB_PATH", data_dir / "fantasy.duckdb"))
        return cls(
            root=base,
            config_dir=config_dir.expanduser(),
            data_dir=data_dir.expanduser(),
            db_path=db_path.expanduser(),
        )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def league_file(self) -> Path:
        return self.config_dir / "league.yaml"

    @property
    def league_example_file(self) -> Path:
        return self.config_dir / "league.example.yaml"

    @property
    def weights_file(self) -> Path:
        return self.config_dir / "scoring_weights.yaml"

    @property
    def data_sources_file(self) -> Path:
        return self.config_dir / "data_sources.yaml"

    def ensure_dirs(self) -> None:
        for directory in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.cache_dir,
            self.log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# --- League -------------------------------------------------------------------------

NonNegFloat = Annotated[float, Field(ge=0)]


class StrictModel(BaseModel):
    """Base for config models: rejects unknown keys but stays round-trippable.

    ``extra="forbid"`` turns a typo in league.yaml into a loud error instead of a
    silently ignored setting. On its own that also breaks
    ``Model.model_validate(instance.model_dump())``, because ``model_dump`` emits
    computed fields that are not accepted as input. We drop exactly those keys on the
    way in, so round-tripping works while real typos still fail.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_fields(cls, data: Any) -> Any:
        if isinstance(data, dict) and cls.model_computed_fields:
            computed = set(cls.model_computed_fields)
            if computed & data.keys():
                return {k: v for k, v in data.items() if k not in computed}
        return data


class ScoringRules(StrictModel):
    """Points per event. Defaults describe a common half-PPR league."""

    passing_yards_per_point: float = Field(25.0, gt=0)
    passing_td: float = 4.0
    passing_interception: float = -2.0
    passing_2pt: float = 2.0

    rushing_yards_per_point: float = Field(10.0, gt=0)
    rushing_td: float = 6.0
    rushing_2pt: float = 2.0

    receiving_yards_per_point: float = Field(10.0, gt=0)
    receiving_td: float = 6.0
    receiving_2pt: float = 2.0
    reception: float = 0.5

    fumble_lost: float = -2.0

    # Optional milestone bonuses; 0.0 disables.
    bonus_pass_300_yards: float = 0.0
    bonus_rush_100_yards: float = 0.0
    bonus_rec_100_yards: float = 0.0
    bonus_pass_400_yards: float = 0.0
    bonus_rush_200_yards: float = 0.0
    bonus_rec_200_yards: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ppr_label(self) -> str:
        """Human label for the reception setting."""
        if self.reception >= 1.0:
            return "PPR"
        if self.reception <= 0.0:
            return "Standard"
        if abs(self.reception - 0.5) < 1e-9:
            return "Half-PPR"
        return f"{self.reception:g}-PPR"


class RosterSlots(StrictModel):
    """Starting lineup requirements plus bench depth."""

    qb: int = Field(1, ge=0)
    rb: int = Field(2, ge=0)
    wr: int = Field(2, ge=0)
    te: int = Field(1, ge=0)
    flex: int = Field(1, ge=0, description="RB/WR/TE")
    wrrb_flex: int = Field(0, ge=0, description="RB/WR only")
    rec_flex: int = Field(0, ge=0, description="WR/TE only")
    superflex: int = Field(0, ge=0, description="QB/RB/WR/TE")
    k: int = Field(1, ge=0)
    dst: int = Field(1, ge=0)
    bench: int = Field(6, ge=0)
    ir: int = Field(1, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def starters(self) -> int:
        return (
            self.qb
            + self.rb
            + self.wr
            + self.te
            + self.flex
            + self.wrrb_flex
            + self.rec_flex
            + self.superflex
            + self.k
            + self.dst
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> int:
        """Roster size excluding IR, which does not consume a draft pick."""
        return self.starters + self.bench

    @property
    def dedicated(self) -> dict[str, int]:
        """Position -> guaranteed starting slots (no flex)."""
        return {"QB": self.qb, "RB": self.rb, "WR": self.wr, "TE": self.te,
                "K": self.k, "DST": self.dst}

    @property
    def flex_counts(self) -> dict[str, int]:
        """Flex slot name -> count, omitting zeros."""
        raw = {
            "FLEX": self.flex,
            "WRRB_FLEX": self.wrrb_flex,
            "REC_FLEX": self.rec_flex,
            "SUPERFLEX": self.superflex,
        }
        return {name: n for name, n in raw.items() if n > 0}

    @property
    def is_superflex(self) -> bool:
        return self.superflex > 0

    def flex_slots_accepting(self, position: str) -> int:
        """How many flex slots this position is eligible for."""
        return sum(
            count
            for name, count in self.flex_counts.items()
            if position in FLEX_ELIGIBILITY[name]
        )


class DraftSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["snake", "linear"] = "snake"
    rounds: int = Field(15, ge=1, le=40)
    slot: int | None = Field(
        None, ge=1, description="Our 1-indexed draft position; None until known."
    )
    third_round_reversal: bool = Field(
        False, description="Some Sleeper leagues reverse again in round 3."
    )
    seconds_on_clock: int = Field(90, ge=0)


class LeagueConfig(StrictModel):
    """Everything about *our* league. Drives projections, VBD, scarcity, and strategy."""

    name: str = "My League"
    season: int = Field(DEFAULT_SEASON, ge=2000, le=2100)
    platform: Literal["sleeper", "espn", "yahoo", "manual"] = "sleeper"
    league_id: str | None = None
    draft_id: str | None = None
    teams: int = Field(12, ge=2, le=32)
    playoff_weeks: list[int] = Field(default_factory=lambda: [15, 16, 17])
    regular_season_weeks: list[int] = Field(default_factory=lambda: list(range(1, 15)))

    draft: DraftSettings = Field(default_factory=DraftSettings)
    roster: RosterSlots = Field(default_factory=RosterSlots)
    scoring: ScoringRules = Field(default_factory=ScoringRules)

    @field_validator("playoff_weeks", "regular_season_weeks")
    @classmethod
    def _weeks_in_range(cls, weeks: list[int]) -> list[int]:
        bad = [w for w in weeks if not 1 <= w <= 22]
        if bad:
            raise ValueError(f"weeks must be between 1 and 22; got {bad}")
        return sorted(set(weeks))

    @model_validator(mode="after")
    def _check_slot(self) -> LeagueConfig:
        if self.draft.slot is not None and self.draft.slot > self.teams:
            raise ValueError(
                f"draft.slot {self.draft.slot} exceeds teams ({self.teams}). "
                "Slot must be between 1 and the number of teams."
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def scoring_type(self) -> str:
        return self.scoring.ppr_label

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_drafted_players(self) -> int:
        return self.teams * self.draft.rounds

    @property
    def label(self) -> str:
        sf = " Superflex" if self.roster.is_superflex else ""
        return f"{self.teams}-team {self.scoring_type}{sf}"

    def starter_demand(self, position: str) -> float:
        """League-wide starting slots consumed by ``position``.

        Dedicated slots count fully. Flex slots are shared, so we attribute them by an
        even split across eligible positions here; :mod:`analytics.replacement` refines
        that with actual usage rates. Kept simple and explicit on purpose.
        """
        per_team = float(self.roster.dedicated.get(position, 0))
        for name, count in self.roster.flex_counts.items():
            eligible = FLEX_ELIGIBILITY[name]
            if position in eligible:
                per_team += count / len(eligible)
        return per_team * self.teams

    @cached_property
    def scoring_dict(self) -> dict[str, float]:
        return self.scoring.model_dump()


# --- Scoring weights ----------------------------------------------------------------


class WeightBlock(BaseModel):
    """A set of component weights that must sum to 1.0."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _sums_to_one(self) -> WeightBlock:
        total = sum(
            v for k, v in self.model_dump().items() if isinstance(v, int | float)
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"{type(self).__name__} weights must sum to 1.0, got {total:.6f}. "
                f"Fix them in config/scoring_weights.yaml."
            )
        return self

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.model_dump().items()}


class PlayerScoreWeights(WeightBlock):
    """"How good is this player, ignoring where we are in the draft?\""""

    projection: NonNegFloat = 0.35
    vbd: NonNegFloat = 0.25
    opportunity: NonNegFloat = 0.15
    offense_environment: NonNegFloat = 0.10
    schedule: NonNegFloat = 0.075
    risk: NonNegFloat = 0.075


class ValueScoreWeights(WeightBlock):
    """"How much value does taking him *here* capture?\""""

    market_value: NonNegFloat = 0.40
    tier_cliff: NonNegFloat = 0.25
    scarcity: NonNegFloat = 0.20
    projection_vs_market: NonNegFloat = 0.15


class DraftNowWeights(WeightBlock):
    """"What should I select at this exact moment?\""""

    player_score: NonNegFloat = 0.35
    value_score: NonNegFloat = 0.20
    next_pick_urgency: NonNegFloat = 0.15
    tier_scarcity: NonNegFloat = 0.10
    roster_fit: NonNegFloat = 0.075
    draft_room: NonNegFloat = 0.075
    strategy_fit: NonNegFloat = 0.05


class ReplacementConfig(BaseModel):
    """How replacement level is estimated for VBD. See docs/scoring.md."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["starter_demand", "fixed_rank", "blended"] = "blended"
    #: Multiplier on league-wide starter demand; >1 accounts for bench hoarding.
    bench_multiplier: dict[str, float] = Field(
        default_factory=lambda: {"QB": 1.35, "RB": 1.55, "WR": 1.50, "TE": 1.25,
                                 "K": 1.0, "DST": 1.0}
    )
    #: Hard fallback ranks used when starter demand cannot be computed.
    fixed_rank: dict[str, int] = Field(
        default_factory=lambda: {"QB": 14, "RB": 36, "WR": 42, "TE": 14,
                                 "K": 13, "DST": 13}
    )
    #: In "blended" mode, weight on the starter-demand estimate (rest on fixed_rank).
    blend_weight: float = Field(0.7, ge=0, le=1)
    #: Smooth replacement points over this many ranks around the replacement index.
    smoothing_window: int = Field(3, ge=1)


class TierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["gap", "kmeans_1d"] = "gap"
    #: A gap larger than this many standard deviations of adjacent gaps starts a tier.
    gap_sigma: float = Field(1.0, gt=0)
    min_tier_size: int = Field(1, ge=1)
    max_tiers: dict[str, int] = Field(
        default_factory=lambda: {"QB": 8, "RB": 10, "WR": 10, "TE": 8,
                                 "K": 4, "DST": 4}
    )
    #: Only tier the top N players per position; the tail is one residual tier.
    depth: dict[str, int] = Field(
        default_factory=lambda: {"QB": 32, "RB": 60, "WR": 72, "TE": 32,
                                 "K": 24, "DST": 24}
    )


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iterations: int = Field(2000, ge=10, le=100_000)
    seed: int = 20260831
    #: Std-dev of the noise added to a pick's rank, in picks, as a fraction of ADP sd.
    adp_noise_scale: float = Field(1.0, ge=0)
    #: Probability an opponent ignores need and simply takes best available.
    best_available_rate: float = Field(0.35, ge=0, le=1)
    #: Cap on candidates evaluated for two-pick expected value (cost control).
    two_pick_candidates: int = Field(8, ge=1, le=50)


class StrategyPriors(BaseModel):
    model_config = ConfigDict(extra="forbid")

    balanced: NonNegFloat = 0.35
    hero_rb: NonNegFloat = 0.40
    robust_rb: NonNegFloat = 0.15
    zero_rb: NonNegFloat = 0.10

    @model_validator(mode="after")
    def _sums_to_one(self) -> StrategyPriors:
        total = self.balanced + self.hero_rb + self.robust_rb + self.zero_rb
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"strategy priors must sum to 1.0, got {total:.6f}")
        return self

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.model_dump().items()}


class WeightsConfig(BaseModel):
    """Root of ``config/scoring_weights.yaml``."""

    model_config = ConfigDict(extra="forbid")

    player_score: PlayerScoreWeights = Field(default_factory=PlayerScoreWeights)
    value_score: ValueScoreWeights = Field(default_factory=ValueScoreWeights)
    draft_now: DraftNowWeights = Field(default_factory=DraftNowWeights)
    replacement: ReplacementConfig = Field(default_factory=ReplacementConfig)
    tiers: TierConfig = Field(default_factory=TierConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    strategy_priors: StrategyPriors = Field(default_factory=StrategyPriors)
    #: Exponential decay applied to older seasons when blending historical signals.
    season_recency_halflife: float = Field(1.5, gt=0)


# --- Data sources -------------------------------------------------------------------


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    #: Data older than this many hours is reported stale by ``ff data status``.
    max_age_hours: float = Field(24.0, gt=0)
    #: Relative weight when combining multiple projection/ADP sources.
    weight: float = Field(1.0, ge=0)
    notes: str = ""


class DataSourcesConfig(BaseModel):
    """Root of ``config/data_sources.yaml``."""

    model_config = ConfigDict(extra="forbid")

    history_seasons: int = Field(4, ge=1, le=25)
    sources: dict[str, SourceSpec] = Field(default_factory=dict)

    def spec(self, name: str) -> SourceSpec:
        return self.sources.get(name, SourceSpec())

    def enabled_sources(self) -> list[str]:
        return [name for name, spec in self.sources.items() if spec.enabled]


# --- Aggregate ----------------------------------------------------------------------


class ConfigError(RuntimeError):
    """Raised with a human-readable message when configuration cannot be loaded."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML:\n  {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level.")
    return loaded


def _build(model: type[BaseModel], data: dict[str, Any], path: Path) -> Any:
    from pydantic import ValidationError

    try:
        return model.model_validate(data)
    except ValidationError as exc:
        lines = [f"{path.name} is invalid:"]
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "(root)"
            lines.append(f"  - {loc}: {err['msg']}")
        raise ConfigError("\n".join(lines)) from exc


class AppConfig(BaseModel):
    """Everything the engine needs to run, loaded once and passed down."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    paths: Paths
    league: LeagueConfig
    weights: WeightsConfig
    data_sources: DataSourcesConfig
    league_file_exists: bool

    @classmethod
    def load(cls, root: Path | None = None) -> AppConfig:
        paths = Paths.resolve(root)
        league_path = paths.league_file
        league_exists = league_path.is_file()
        # Fall back to the committed example so the whole engine still runs before the
        # user has written their own league.yaml.
        source = league_path if league_exists else paths.league_example_file
        return cls(
            paths=paths,
            league=_build(LeagueConfig, _read_yaml(source), source),
            weights=_build(WeightsConfig, _read_yaml(paths.weights_file), paths.weights_file),
            data_sources=_build(
                DataSourcesConfig, _read_yaml(paths.data_sources_file), paths.data_sources_file
            ),
            league_file_exists=league_exists,
        )

    @property
    def positions(self) -> tuple[str, ...]:
        """Positions worth scoring for this league (K/DST only if they start)."""
        extra = tuple(
            p for p in ("K", "DST") if self.league.roster.dedicated.get(p, 0) > 0
        )
        return OFFENSE_POSITIONS + extra


def load_config(root: Path | None = None) -> AppConfig:
    """Convenience wrapper around :meth:`AppConfig.load`."""
    return AppConfig.load(root)
