from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

from .event_store import DataSnapshot
from .models import Candidate, PositiveInteraction


@dataclass(frozen=True, slots=True)
class PopularityModel:
    catalog: tuple[int, ...]
    counts: Mapping[int, int]
    subject_histories: Mapping[int, tuple[int, ...]]

    def ranked_movie_ids(self, excluded_movie_ids: Collection[int] = ()) -> tuple[int, ...]:
        excluded = set(excluded_movie_ids)
        return tuple(
            movie_id
            for movie_id in sorted(self.catalog, key=lambda item: (-self.counts.get(item, 0), item))
            if movie_id not in excluded
        )

    def recommend(self, excluded_movie_ids: Collection[int], top_n: int) -> tuple[Candidate, ...]:
        if not 1 <= top_n <= 100:
            raise ValueError("top_n must be between 1 and 100")
        movie_ids = self.ranked_movie_ids(excluded_movie_ids)[:top_n]
        return tuple(
            Candidate(movie_id=movie_id, score=self.counts.get(movie_id, 0), rank=rank)
            for rank, movie_id in enumerate(movie_ids, start=1)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "catalog": list(self.catalog),
            "counts": {str(movie_id): count for movie_id, count in self.counts.items()},
            "subject_histories": {
                str(subject_id): list(movie_ids)
                for subject_id, movie_ids in self.subject_histories.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PopularityModel:
        raw_catalog = value["catalog"]
        raw_counts = value["counts"]
        raw_histories = value["subject_histories"]
        if (
            not isinstance(raw_catalog, list)
            or not isinstance(raw_counts, Mapping)
            or not isinstance(raw_histories, Mapping)
        ):
            raise ValueError("Popularity payload has invalid catalog, count, or history mappings")

        subject_histories: dict[int, tuple[int, ...]] = {}
        for subject_id, movie_ids in raw_histories.items():
            if not isinstance(movie_ids, list):
                raise ValueError("Popularity payload has an invalid Subject history")
            subject_histories[int(subject_id)] = tuple(int(movie_id) for movie_id in movie_ids)

        return cls(
            catalog=tuple(int(movie_id) for movie_id in raw_catalog),
            counts={int(movie_id): int(count) for movie_id, count in raw_counts.items()},
            subject_histories=subject_histories,
        )


def fit_popularity(
    snapshot: DataSnapshot, interactions: Iterable[PositiveInteraction]
) -> PopularityModel:
    interaction_values = tuple(interactions)
    catalog = tuple(sorted({event.movie_id for event in snapshot.events}))
    counts = Counter(interaction.movie_id for interaction in interaction_values)
    histories: defaultdict[int, list[PositiveInteraction]] = defaultdict(list)
    for interaction in interaction_values:
        histories[interaction.subject_id].append(interaction)

    subject_histories = {
        subject_id: tuple(
            interaction.movie_id
            for interaction in sorted(
                subject_interactions,
                key=lambda item: (item.event_time, item.event_id),
            )
        )
        for subject_id, subject_interactions in histories.items()
    }
    return PopularityModel(catalog=catalog, counts=counts, subject_histories=subject_histories)
