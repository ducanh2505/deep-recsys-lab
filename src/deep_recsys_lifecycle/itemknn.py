from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

from .event_store import DataSnapshot
from .models import Candidate, PositiveInteraction
from .positive import history_movie_ids
from .retriever import MAX_CANDIDATE_POOL, validate_candidate_pool_limit


@dataclass(frozen=True, slots=True)
class ItemKNNModel:
    """Sparse binary item-item cosine retriever fitted on a Data Snapshot."""

    catalog: tuple[int, ...]
    item_subjects: Mapping[int, frozenset[int]]
    subject_histories: Mapping[int, tuple[int, ...]]

    @property
    def name(self) -> str:
        return "itemknn"

    def similarity(self, left_movie_id: int, right_movie_id: int) -> float:
        left_subjects = self.item_subjects.get(left_movie_id, frozenset())
        right_subjects = self.item_subjects.get(right_movie_id, frozenset())
        if not left_subjects or not right_subjects:
            return 0.0
        return len(left_subjects & right_subjects) / math.sqrt(
            len(left_subjects) * len(right_subjects)
        )

    def score(self, movie_id: int, history: Collection[int]) -> float:
        return sum(
            self.similarity(movie_id, history_movie_id)
            for history_movie_id in dict.fromkeys(history)
        )

    def candidate_pool(
        self,
        history: Collection[int],
        limit: int = MAX_CANDIDATE_POOL,
    ) -> tuple[Candidate, ...]:
        validate_candidate_pool_limit(limit)
        excluded = set(history)
        scored = (
            (movie_id, self.score(movie_id, history))
            for movie_id in self.catalog
            if movie_id not in excluded
        )
        ordered = sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]
        return tuple(
            Candidate(movie_id=movie_id, score=score, rank=rank)
            for rank, (movie_id, score) in enumerate(ordered, start=1)
        )

    def recommend(self, history: Collection[int], top_n: int) -> tuple[Candidate, ...]:
        if not 1 <= top_n <= 100:
            raise ValueError("top_n must be between 1 and 100")
        return self.candidate_pool(history, limit=top_n)


def fit_itemknn(
    snapshot: DataSnapshot, interactions: Iterable[PositiveInteraction]
) -> ItemKNNModel:
    """Fit ItemKNN from binary Positive Interactions in the supplied snapshot only."""

    snapshot_event_ids = {event.event_id for event in snapshot.events}
    histories: defaultdict[int, list[PositiveInteraction]] = defaultdict(list)
    item_subjects: defaultdict[int, set[int]] = defaultdict(set)
    for interaction in interactions:
        if interaction.event_id not in snapshot_event_ids:
            continue
        histories[interaction.subject_id].append(interaction)
        item_subjects[interaction.movie_id].add(interaction.subject_id)

    subject_histories = {
        subject_id: history_movie_ids(subject_interactions)
        for subject_id, subject_interactions in histories.items()
    }
    return ItemKNNModel(
        catalog=tuple(sorted({event.movie_id for event in snapshot.events})),
        item_subjects={
            movie_id: frozenset(subject_ids) for movie_id, subject_ids in item_subjects.items()
        },
        subject_histories=subject_histories,
    )
