from __future__ import annotations

from collections.abc import Collection, Mapping
from math import sqrt

import numpy as np
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from .models import Candidate


class SparseItemKNNScorer:
    """Score the exact binary cosine ItemKNN formula with two sparse products."""

    def __init__(
        self,
        catalog: tuple[int, ...],
        item_subjects: Mapping[int, frozenset[int]],
        subject_histories: Mapping[int, tuple[int, ...]],
    ) -> None:
        self.catalog = catalog
        self.movie_index = {movie_id: index for index, movie_id in enumerate(catalog)}
        subject_index = {
            subject_id: index for index, subject_id in enumerate(sorted(subject_histories))
        }
        counts = np.fromiter(
            (len(item_subjects.get(movie_id, ())) for movie_id in catalog),
            dtype=np.int32,
            count=len(catalog),
        )
        edge_count = int(counts.sum())
        columns = np.repeat(np.arange(len(catalog), dtype=np.int32), counts)
        rows = np.fromiter(
            (
                subject_index[subject_id]
                for movie_id in catalog
                for subject_id in item_subjects.get(movie_id, ())
            ),
            dtype=np.int32,
            count=edge_count,
        )
        self.matrix = csr_matrix(
            (np.ones(edge_count, dtype=np.float64), (rows, columns)),
            shape=(len(subject_index), len(catalog)),
        )
        self.movie_counts = np.asarray(self.matrix.sum(axis=0)).ravel()
        self.movie_ids = np.asarray(catalog, dtype=np.int64)

    def candidate_pool(self, history: Collection[int], limit: int) -> tuple[Candidate, ...]:
        query = np.zeros(len(self.catalog), dtype=np.float64)
        excluded: list[int] = []
        for movie_id in set(history):
            index = self.movie_index.get(movie_id)
            if index is not None:
                excluded.append(index)
                count = self.movie_counts[index]
                if count:
                    query[index] = 1.0 / sqrt(float(count))
        subject_weights = self.matrix @ query
        scores = np.asarray(self.matrix.T @ subject_weights).ravel()
        denominators = np.sqrt(np.maximum(self.movie_counts, 1.0))
        scores /= denominators
        if excluded:
            scores[np.asarray(excluded, dtype=np.int32)] = -np.inf
        order = np.lexsort((self.movie_ids, -scores))
        order = order[np.isfinite(scores[order])][:limit]
        return tuple(
            Candidate(movie_id=int(self.movie_ids[index]), score=float(scores[index]), rank=rank)
            for rank, index in enumerate(order, start=1)
        )
