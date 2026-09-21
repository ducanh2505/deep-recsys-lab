from __future__ import annotations

from collections.abc import Iterable

from .models import PositiveInteraction, RatingEvent


def derive_positive_interactions(
    events: Iterable[RatingEvent], threshold: float = 4.0
) -> tuple[PositiveInteraction, ...]:
    """Keep only Rating Events at or above the declared implicit-signal threshold."""

    return tuple(
        PositiveInteraction.from_event(event) for event in events if event.rating >= threshold
    )
