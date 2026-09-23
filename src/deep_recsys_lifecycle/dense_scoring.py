from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np

from .models import Candidate


def _top_candidates(
    movie_ids: np.ndarray, scores: np.ndarray, excluded_indices: Collection[int], limit: int
) -> tuple[Candidate, ...]:
    if not np.all(np.isfinite(scores)):
        raise ValueError("retriever produced a non-finite score")
    if excluded_indices:
        scores[list(excluded_indices)] = -np.inf
    order = np.lexsort((movie_ids, -scores))
    order = order[np.isfinite(scores[order])][:limit]
    return tuple(
        Candidate(movie_id=int(movie_ids[index]), score=float(scores[index]), rank=rank)
        for rank, index in enumerate(order, start=1)
    )


class MultVAEScorer:
    def __init__(
        self,
        catalog: tuple[int, ...],
        weights: Mapping[str, tuple[float, ...] | tuple[tuple[float, ...], ...]],
    ) -> None:
        self.movie_ids = np.asarray(catalog, dtype=np.int64)
        self.movie_index = {movie_id: index for index, movie_id in enumerate(catalog)}
        self.encoder_weight = np.asarray(weights["encoder_weight"], dtype=np.float64)
        self.encoder_bias = np.asarray(weights["encoder_bias"], dtype=np.float64)
        self.mean_weight = np.asarray(weights["mean_weight"], dtype=np.float64)
        self.mean_bias = np.asarray(weights["mean_bias"], dtype=np.float64)
        self.decoder_weight = np.asarray(weights["decoder_weight"], dtype=np.float64)
        self.decoder_bias = np.asarray(weights["decoder_bias"], dtype=np.float64)
        self.output_weight = np.asarray(weights["output_weight"], dtype=np.float64)
        self.output_bias = np.asarray(weights["output_bias"], dtype=np.float64)

    def candidate_pool(self, history: Collection[int], limit: int) -> tuple[Candidate, ...]:
        observed = {
            self.movie_index[movie_id]
            for movie_id in history
            if movie_id in self.movie_index
        }
        if not observed:
            scores = np.zeros(len(self.movie_ids), dtype=np.float64)
        else:
            profile = np.zeros(len(self.movie_ids), dtype=np.float64)
            profile[list(observed)] = 1.0 / np.sqrt(len(observed))
            hidden = np.tanh(self.encoder_weight @ profile + self.encoder_bias)
            latent = self.mean_weight @ hidden + self.mean_bias
            decoded = np.tanh(self.decoder_weight @ latent + self.decoder_bias)
            scores = self.output_weight @ decoded + self.output_bias
        return _top_candidates(self.movie_ids, scores, observed, limit)


class LightGCNScorer:
    def __init__(
        self,
        subject_ids: tuple[int, ...],
        movie_ids: tuple[int, ...],
        subject_embeddings: tuple[tuple[float, ...], ...],
        movie_embeddings: tuple[tuple[float, ...], ...],
    ) -> None:
        self.subject_index = {
            subject_id: index for index, subject_id in enumerate(subject_ids)
        }
        self.movie_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}
        self.movie_ids = np.asarray(movie_ids, dtype=np.int64)
        self.subject_embeddings = np.asarray(subject_embeddings, dtype=np.float64)
        self.movie_embeddings = np.asarray(movie_embeddings, dtype=np.float64)

    def candidate_pool(
        self, subject_id: int, history: Collection[int], limit: int
    ) -> tuple[Candidate, ...]:
        index = self.subject_index.get(subject_id)
        if index is None:
            raise KeyError(f"LightGCN has no embedding for Subject {subject_id}")
        scores = self.movie_embeddings @ self.subject_embeddings[index]
        excluded = {
            self.movie_index[movie_id]
            for movie_id in history
            if movie_id in self.movie_index
        }
        return _top_candidates(self.movie_ids, scores, excluded, limit)
