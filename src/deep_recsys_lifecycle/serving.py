from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from .models import Candidate
from .retriever import MAX_CANDIDATE_POOL, validate_candidate_pool_limit


@dataclass(frozen=True, slots=True)
class PopularityRetriever:
    """CPU-only Popularity state used by the serving runtime.

    This module intentionally has no Event Store, dataset, or fitting imports.  The
    training module can produce this state, while the API only needs this runtime
    representation to answer queries from an immutable artifact.
    """

    catalog: tuple[int, ...]
    counts: Mapping[int, int]
    subject_histories: Mapping[int, tuple[int, ...]]

    @property
    def name(self) -> str:
        return "popularity"

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
        return self.candidate_pool(excluded_movie_ids, limit=top_n)

    def candidate_pool(
        self,
        excluded_movie_ids: Collection[int],
        limit: int = MAX_CANDIDATE_POOL,
    ) -> tuple[Candidate, ...]:
        validate_candidate_pool_limit(limit)
        movie_ids = self.ranked_movie_ids(excluded_movie_ids)[:limit]
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
    def from_dict(cls, value: Mapping[str, object]) -> PopularityRetriever:
        raw_catalog = value.get("catalog")
        raw_counts = value.get("counts")
        raw_histories = value.get("subject_histories")
        if (
            not isinstance(raw_catalog, list)
            or not isinstance(raw_counts, Mapping)
            or not isinstance(raw_histories, Mapping)
        ):
            raise ValueError("Popularity payload has invalid catalog, count, or history mappings")

        catalog = tuple(_strict_int(item, "catalog movie id") for item in raw_catalog)
        counts: dict[int, int] = {}
        for movie_id, count in raw_counts.items():
            if not isinstance(movie_id, str):
                raise ValueError("Popularity payload count keys must be strings")
            counts[_parse_int(movie_id, "count movie id")] = _strict_int(count, "movie count")

        subject_histories: dict[int, tuple[int, ...]] = {}
        for subject_id, movie_ids in raw_histories.items():
            if not isinstance(subject_id, str) or not isinstance(movie_ids, list):
                raise ValueError("Popularity payload has an invalid Subject history")
            subject_histories[_parse_int(subject_id, "history Subject id")] = tuple(
                _strict_int(movie_id, "history movie id") for movie_id in movie_ids
            )

        if len(set(catalog)) != len(catalog):
            raise ValueError("Popularity payload catalog contains duplicate Movies")
        if any(count < 0 for count in counts.values()):
            raise ValueError("Popularity payload counts cannot be negative")

        return cls(
            catalog=catalog,
            counts=counts,
            subject_histories=subject_histories,
        )


def _strict_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Popularity payload {label} must be an integer")
    return value


def _parse_int(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Popularity payload {label} must be an integer") from error
