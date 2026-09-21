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


def history_movie_ids(interactions: Iterable[PositiveInteraction]) -> tuple[int, ...]:
    """Return one deterministic observed Movie history from Positive Interactions."""

    seen: set[int] = set()
    history: list[int] = []
    for interaction in sorted(interactions, key=lambda item: (item.event_time, item.event_id)):
        if interaction.movie_id not in seen:
            seen.add(interaction.movie_id)
            history.append(interaction.movie_id)
    return tuple(history)
