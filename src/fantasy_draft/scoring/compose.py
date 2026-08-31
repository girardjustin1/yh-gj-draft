"""Weighted composition of component scores.

The rule that makes confidence meaningful lives here.

When a component's confidence is low, its weight is **redistributed across the
components we do know**, rather than the component contributing a neutral 50. Scoring
unknowns as average would quietly drag every uncertain player toward the middle of the
board and, worse, would hide our ignorance behind a confident-looking number.

Concretely, with weights ``{projection: 0.35, vbd: 0.25, schedule: 0.075, ...}`` and no
schedule data at all, the schedule weight is spread proportionally over the rest. The
resulting composite is still 0-100 and still comparable, and the bundle's own confidence
falls to record that we were working with less.
"""

from __future__ import annotations

from ..models import ComponentScore, ScoreBundle

#: Below this, a component is treated as unknown and its weight is redistributed.
MIN_USABLE_CONFIDENCE = 0.05


def compose(
    name: str,
    components: dict[str, ComponentScore],
    weights: dict[str, float],
) -> ScoreBundle:
    """Blend ``components`` using ``weights``, redistributing unknown components' weight.

    Each component's weight is scaled by its confidence, then the weights are
    renormalized. A component with confidence 1.0 gets its full share; one with 0.5 gets
    half, with the freed half spread across the others.
    """
    effective: dict[str, float] = {}
    for key, weight in weights.items():
        component = components.get(key)
        if component is None or component.confidence < MIN_USABLE_CONFIDENCE:
            continue
        effective[key] = weight * component.confidence

    total = sum(effective.values())
    if total <= 0:
        # Nothing is known. Say so rather than inventing a midpoint with confidence.
        return ScoreBundle(
            name=name, value=50.0, confidence=0.0, components=components, weights={}
        )

    normalized = {key: weight / total for key, weight in effective.items()}
    value = sum(components[key].normalized * weight for key, weight in normalized.items())

    # Bundle confidence is the share of the *intended* weight that was actually backed by
    # known data, so dropping the 7.5% schedule component costs far less than dropping
    # the 35% projection.
    intended = sum(weights.values()) or 1.0
    covered = sum(
        weights[key] * components[key].confidence for key in effective
    ) / intended

    return ScoreBundle(
        name=name,
        value=max(0.0, min(100.0, value)),
        confidence=max(0.0, min(1.0, covered)),
        components=components,
        weights=normalized,
    )


def component(
    name: str,
    raw_value: float | None,
    normalized: float | None,
    confidence: float = 1.0,
    method: str = "",
    source: str | None = None,
    notes: str = "",
) -> ComponentScore:
    """Build a :class:`ComponentScore`, treating a missing normalized value as unknown."""
    if normalized is None:
        return ComponentScore(
            name=name, raw_value=raw_value, normalized=50.0, confidence=0.0,
            method=method, source=source, notes=notes or "no data",
        )
    return ComponentScore(
        name=name,
        raw_value=raw_value,
        normalized=max(0.0, min(100.0, float(normalized))),
        confidence=max(0.0, min(1.0, float(confidence))),
        method=method,
        source=source,
        notes=notes,
    )
