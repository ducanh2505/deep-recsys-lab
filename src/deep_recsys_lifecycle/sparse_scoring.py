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
        # Column slicing touches only the observed Movies instead of scanning every
        # Positive Interaction for the first matrix product on each Query.
        self.matrix_by_movie = self.matrix.tocsc()
        self.movie_counts = np.asarray(self.matrix.sum(axis=0)).ravel()
        self.denominators = np.sqrt(np.maximum(self.movie_counts, 1.0))
        self.movie_ids = np.asarray(catalog, dtype=np.int64)

    def candidate_pool(self, history: Collection[int], limit: int) -> tuple[Candidate, ...]:
        excluded: list[int] = []
        for movie_id in set(history):
            index = self.movie_index.get(movie_id)
            if index is not None:
                excluded.append(index)
        observed = sorted(index for index in excluded if self.movie_counts[index])
        if observed:
            weights = np.fromiter(
                (1.0 / sqrt(float(self.movie_counts[index])) for index in observed),
                dtype=np.float64,
                count=len(observed),
            )
            subject_weights = self.matrix_by_movie[:, observed] @ weights
            scores = np.asarray(self.matrix.T @ subject_weights).ravel()
            scores /= self.denominators
        else:
            scores = np.zeros(len(self.catalog), dtype=np.float64)
        if excluded:
            scores[np.asarray(excluded, dtype=np.int32)] = -np.inf
        finite = np.flatnonzero(np.isfinite(scores))
        if 0 < limit < len(finite):
            finite_scores = scores[finite]
            threshold = np.partition(finite_scores, len(finite) - limit)[len(finite) - limit]
            finite = finite[finite_scores >= threshold]
        order = finite[np.lexsort((self.movie_ids[finite], -scores[finite]))][:limit]
        return tuple(
            Candidate(movie_id=int(self.movie_ids[index]), score=float(scores[index]), rank=rank)
            for rank, index in enumerate(order, start=1)
        )
