"""Deterministic NumPy ranking helpers for the Torch-free serving path."""

from __future__ import annotations

import numpy as np


def topk_unseen(
    scores: np.ndarray,
    seen: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unseen indices and scores ordered by score, then catalog index.

    ``numpy.argpartition`` keeps the selection linear in the catalog size. The
    explicit cutoff handling makes ties deterministic even when they cross the
    top-k boundary.
    """

    score_array = np.asarray(scores, dtype=np.float32)
    seen_array = np.asarray(seen, dtype=np.bool_)
    if score_array.ndim != 2 or seen_array.shape != score_array.shape:
        raise ValueError("scores and seen must be [batch, n_items] with equal shapes")
    if k < 1:
        raise ValueError("k must be positive")
    k_eff = min(k, score_array.shape[1])
    available = (~seen_array).sum(axis=1)
    if np.any(available < k_eff):
        raise ValueError("fewer than k unseen items are available")

    masked = np.array(score_array, copy=True)
    masked[seen_array] = -np.inf
    indices = np.empty((masked.shape[0], k_eff), dtype=np.int64)
    values = np.empty((masked.shape[0], k_eff), dtype=np.float32)
    partition_start = masked.shape[1] - k_eff

    for row_index, row in enumerate(masked):
        candidates = np.argpartition(row, partition_start)[partition_start:]
        cutoff = float(row[candidates].min())
        above = np.flatnonzero(row > cutoff)
        tied = np.flatnonzero(row == cutoff)
        selected = np.concatenate((above, tied[: k_eff - len(above)]))
        order = np.lexsort((selected, -row[selected]))
        ranked = selected[order]
        indices[row_index] = ranked
        values[row_index] = row[ranked]
    return indices, values
