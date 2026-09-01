"""All-ranking evaluation and deterministic negative sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from .model import LightGCN


@dataclass
class BPRSampler:
    """User-uniform sampler whose negatives never occur in its observed graph."""

    observed: sp.csr_matrix
    seed: int

    def __post_init__(self) -> None:
        self.observed = self.observed.tocsr(copy=True)
        self.observed.sort_indices()
        self.rng = np.random.default_rng(self.seed)
        row_counts = np.diff(self.observed.indptr)
        self.eligible_users = np.flatnonzero(
            (row_counts > 0) & (row_counts < self.observed.shape[1])
        ).astype(np.int64)
        if self.eligible_users.size == 0:
            raise ValueError("sampler needs at least one user with a positive edge")
        rows = np.repeat(
            np.arange(self.observed.shape[0], dtype=np.int64), np.diff(self.observed.indptr)
        )
        self.encoded_positives = rows * self.observed.shape[1] + self.observed.indices
        self.encoded_positives.sort()

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw a deterministic batch of user/positive/negative triples."""

        users = self.rng.choice(self.eligible_users, size=batch_size, replace=True)
        lengths = self.observed.indptr[users + 1] - self.observed.indptr[users]
        offsets = (self.rng.random(batch_size) * lengths).astype(np.int64)
        positives = self.observed.indices[self.observed.indptr[users] + offsets]
        negatives = self.rng.integers(0, self.observed.shape[1], size=batch_size, dtype=np.int64)
        keys = users * self.observed.shape[1] + negatives
        positions = np.searchsorted(self.encoded_positives, keys)
        clipped = np.minimum(positions, len(self.encoded_positives) - 1)
        collisions = self.encoded_positives[clipped] == keys
        while np.any(collisions):
            negatives[collisions] = self.rng.integers(
                0, self.observed.shape[1], size=int(collisions.sum()), dtype=np.int64
            )
            keys = users * self.observed.shape[1] + negatives
            positions = np.searchsorted(self.encoded_positives, keys)
            clipped = np.minimum(positions, len(self.encoded_positives) - 1)
            collisions = self.encoded_positives[clipped] == keys
        return users, positives.astype(np.int64, copy=False), negatives

    def state_dict(self) -> dict[str, Any]:
        """Capture the NumPy bit-generator state for exact resume."""

        return {"bit_generator": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a state returned by :meth:`state_dict`."""

        self.rng.bit_generator.state = state["bit_generator"]


def ranking_metrics(top_items: np.ndarray, truth: sp.csr_matrix) -> dict[str, float | int]:
    """Compute Recall and NDCG with paper-compatible ideal rankings."""

    truth = truth.tocsr(copy=False)
    if top_items.shape[0] != truth.shape[0]:
        raise ValueError("prediction and truth user counts differ")
    k = top_items.shape[1]
    truth_counts = np.diff(truth.indptr)
    eligible = truth_counts > 0
    if not np.any(eligible):
        raise ValueError("truth has no evaluable users")
    rows = np.repeat(np.arange(truth.shape[0], dtype=np.int64), truth_counts)
    encoded_truth = rows * truth.shape[1] + truth.indices
    encoded_truth.sort()
    prediction_keys = (
        np.arange(truth.shape[0], dtype=np.int64)[:, None] * truth.shape[1] + top_items
    )
    positions = np.searchsorted(encoded_truth, prediction_keys)
    clipped = np.minimum(positions, len(encoded_truth) - 1)
    hits = encoded_truth[clipped] == prediction_keys
    recall = hits.sum(axis=1) / np.maximum(truth_counts, 1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = (hits * discounts).sum(axis=1)
    ideal_prefix = np.concatenate(([0.0], np.cumsum(discounts)))
    idcg = ideal_prefix[np.minimum(truth_counts, k)]
    ndcg = np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)
    return {
        f"recall@{k}": float(recall[eligible].mean()),
        f"ndcg@{k}": float(ndcg[eligible].mean()),
        "evaluated_users": int(eligible.sum()),
        "truth_edges": int(truth.nnz),
    }


@torch.inference_mode()
def evaluate_all_ranking(
    model: LightGCN,
    observed: sp.csr_matrix,
    truth: sp.csr_matrix,
    *,
    k: int = 20,
    user_batch_size: int = 512,
) -> dict[str, float | int]:
    """Rank the complete unseen catalog and return Recall/NDCG at ``k``."""

    if observed.shape != truth.shape or observed.shape != (model.n_users, model.n_items):
        raise ValueError("evaluation matrices do not match the model")
    if k > model.n_items:
        raise ValueError("k cannot exceed catalog size")
    model.eval()
    user_embeddings, item_embeddings = model.propagated_embeddings()
    predictions = np.empty((model.n_users, k), dtype=np.int32)
    for start in range(0, model.n_users, user_batch_size):
        stop = min(start + user_batch_size, model.n_users)
        scores = user_embeddings[start:stop] @ item_embeddings.T
        seen = observed[start:stop].tocoo(copy=False)
        if seen.nnz:
            row = torch.from_numpy(seen.row.astype(np.int64, copy=False))
            col = torch.from_numpy(seen.col.astype(np.int64, copy=False))
            scores[row, col] = -torch.inf
        predictions[start:stop] = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
    return ranking_metrics(predictions, truth)
