"""Turning the numbers into a reason.

Every sentence written here is generated from computed values — none of it is a template
of football opinions. If the engine cannot support a claim with a number, the sentence
does not appear.

The explanation is written for someone on a 90-second clock, so it leads with the
decision and the single fact that drove it, then supports it.
"""

from __future__ import annotations

from ..analytics.draft_room import DraftRoomRead
from ..draft.state import DraftState
from ..draft.strategies import StrategyState
from ..models import Candidate, Recommendation


def _pct(value: float | None) -> str:
    return "unknown" if value is None else f"{value * 100:.0f}%"


def _survival_phrase(candidate: Candidate) -> str:
    if candidate.survival is None or candidate.survival.confidence <= 0:
        return "his survival to our next pick could not be modelled"
    gone = candidate.survival.probability_gone
    if gone >= 0.85:
        return f"he is {_pct(gone)} likely to be gone before our next pick"
    if gone >= 0.55:
        return f"he is more likely than not gone before our next pick ({_pct(gone)})"
    if gone <= 0.25:
        return f"there is a good chance he lasts to our next pick ({_pct(1 - gone)} available)"
    return f"he is a coin flip to last to our next pick ({_pct(1 - gone)} available)"


def _tier_phrase(candidate: Candidate) -> str | None:
    if candidate.points_to_next_tier is None or candidate.tier is None:
        return None
    drop = candidate.points_to_next_tier
    if drop < 8:
        return None
    return (
        f"He is in {candidate.position} tier {candidate.tier}, and the drop to the next "
        f"tier is {drop:.0f} projected points"
    )


def compare_two(primary: Candidate, other: Candidate) -> str:
    """The core argument: why this one rather than that one."""
    lines: list[str] = []
    player_gap = (primary.player_score.value if primary.player_score else 0) - (
        other.player_score.value if other.player_score else 0
    )

    if player_gap < -0.5:
        lines.append(
            f"{other.name} rates higher on season-long value "
            f"({other.player_score.value:.1f} vs {primary.player_score.value:.1f} Player "
            f"Score), so this is a decision about timing, not talent."
        )
    elif abs(player_gap) <= 0.5:
        lines.append(
            f"{primary.name} and {other.name} are effectively tied on season-long value, "
            f"so the decision comes down to who survives."
        )
    else:
        lines.append(
            f"{primary.name} rates higher on season-long value "
            f"(+{player_gap:.1f} Player Score)."
        )

    if primary.survival and other.survival and primary.survival.confidence > 0:
        primary_gone = primary.survival.probability_gone
        other_available = other.survival.probability_available
        if primary_gone > other.survival.probability_gone + 0.15:
            lines.append(
                f"But {primary.name} is {_pct(primary_gone)} likely to be gone before "
                f"our next pick, while {other.name} has a {_pct(other_available)} chance "
                f"of surviving it. Taking {primary.name} now is the only way to have "
                f"both; taking {other.name} risks having neither."
            )
        elif other.survival.probability_gone > primary_gone + 0.15:
            lines.append(
                f"{other.name} is the more urgent of the two "
                f"({_pct(other.survival.probability_gone)} gone versus "
                f"{_pct(primary_gone)}), which is why he is the leading alternative."
            )
    return " ".join(lines)


def write_explanation(
    recommendation: Recommendation,
    state: DraftState,
    room: DraftRoomRead,
    strategy: StrategyState,
    simulation: object | None = None,
    two_pick_override: str | None = None,
) -> str:
    """Compose the full narrative for the recommendation."""
    primary = recommendation.primary
    if primary is None:
        return "No candidate could be scored. Check `ff data status` and `ff draft sync`."

    parts: list[str] = []

    # 1. The decision and the fact behind it.
    opening = f"Take {primary.name} ({primary.position}"
    if primary.team:
        opening += f", {primary.team}"
    opening += ")."
    parts.append(opening)

    reasons: list[str] = []
    if primary.vbd is not None:
        reasons.append(
            f"He projects {primary.projected_points:.0f} points, "
            f"{primary.vbd:.0f} above {primary.position} replacement"
        )
    survival_text = _survival_phrase(primary)
    reasons.append(survival_text[0].upper() + survival_text[1:])
    parts.append(". ".join(reasons) + ".")

    # 2. The tier cliff, if there is one worth naming.
    tier = _tier_phrase(primary)
    if tier:
        parts.append(tier + ".")

    # 3. Market position.
    if primary.adp is not None:
        delta = primary.adp - recommendation.overall_pick
        if delta >= 4:
            parts.append(
                f"He is going {delta:.0f} picks later than his consensus ADP of "
                f"{primary.adp:.0f}, so this is value rather than a reach."
            )
        elif delta <= -6:
            parts.append(
                f"This is {abs(delta):.0f} picks ahead of his consensus ADP of "
                f"{primary.adp:.0f} — a deliberate reach, justified by the urgency above "
                f"rather than by his ranking."
            )

    # 4. The alternative, and why not.
    if recommendation.alternatives:
        parts.append(compare_two(primary, recommendation.alternatives[0]))

    # 5. The pair of picks, which is the decision that actually matters.
    if two_pick_override:
        parts.append(two_pick_override)
    elif primary.two_pick_expected_value is not None:
        paths = [
            f"{c.name} {c.two_pick_expected_value:.1f}"
            for c in [primary, *recommendation.alternatives]
            if c.two_pick_expected_value is not None
        ]
        if len(paths) > 1:
            parts.append(
                "Across both picks, simulated expected value: " + "; ".join(paths) + "."
            )

    # 6. What the room is doing.
    if room.notes:
        parts.extend(room.notes)

    # 7. Roster construction.
    roster = state.my_roster()
    if roster and roster.size:
        shape = ", ".join(f"{n} {p}" for p, n in sorted(roster.position_counts.items()))
        parts.append(
            f"Our roster is {shape}. Construction is drifting toward "
            f"{strategy.label} ({strategy.reason})."
        )

    # 8. Honest caveats.
    stale = [row.source for row in recommendation.staleness if row.is_stale]
    if stale:
        parts.append(
            f"Caution: {', '.join(stale)} {'is' if len(stale) == 1 else 'are'} stale, "
            f"which lowers confidence in this recommendation."
        )
    parts.append(f"Recommendation confidence: {recommendation.confidence * 100:.0f}%.")

    return " ".join(parts)
