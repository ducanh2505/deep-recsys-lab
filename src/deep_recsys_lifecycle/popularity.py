from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from .event_store import DataSnapshot
from .models import PositiveInteraction
from .positive import history_movie_ids
from .serving import PopularityRetriever


def fit_popularity(
    snapshot: DataSnapshot, interactions: Iterable[PositiveInteraction]
) -> PopularityRetriever:
    snapshot_event_ids = {event.event_id for event in snapshot.events}
    interaction_values = tuple(
        interaction for interaction in interactions if interaction.event_id in snapshot_event_ids
    )
    catalog = tuple(sorted({event.movie_id for event in snapshot.events}))
    counts = Counter(interaction.movie_id for interaction in interaction_values)
    histories: defaultdict[int, list[PositiveInteraction]] = defaultdict(list)
    for interaction in interaction_values:
        histories[interaction.subject_id].append(interaction)

    subject_histories = {
        subject_id: history_movie_ids(subject_interactions)
        for subject_id, subject_interactions in histories.items()
    }
    return PopularityRetriever(catalog=catalog, counts=counts, subject_histories=subject_histories)
