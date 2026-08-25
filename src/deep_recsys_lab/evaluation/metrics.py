"""Seen-item masking and ranking metric goldens."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _tensor(
    value: torch.Tensor | np.ndarray | Any, *, dtype: torch.dtype | None = None
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value if dtype is None else value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def seen_item_mask(scores: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
    """Set scores for observed items to negative infinity without mutation."""

    if scores.shape != seen.shape:
        raise ValueError("scores and seen must have the same shape")
    masked = scores.clone()
    masked.masked_fill_(seen.to(dtype=torch.bool), float("-inf"))
    return masked


def topk_unseen(
    scores: torch.Tensor, seen: torch.Tensor | None, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted unseen item indices and scores."""

    if k < 1:
        raise ValueError("k must be positive")
    masked = seen_item_mask(scores, seen) if seen is not None else scores
    k_eff = min(k, masked.shape[-1])
    values, indices = torch.topk(masked, k=k_eff, dim=-1, largest=True, sorted=True)
    return indices, values


def _validate_ranking_inputs(scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
    if scores.ndim != 2 or ground_truth.ndim != 2 or scores.shape != ground_truth.shape:
        raise ValueError("scores and ground_truth must be [batch, n_items] with equal shapes")


def recall_at_k(
    scores: torch.Tensor | np.ndarray,
    ground_truth: torch.Tensor | np.ndarray,
    k: int,
    seen: torch.Tensor | np.ndarray | None = None,
) -> float:
    """Mean Recall@k, with the standard ``min(k, |truth|)`` denominator."""

    score_tensor = _tensor(scores, dtype=torch.float32)
    truth_tensor = _tensor(ground_truth, dtype=torch.bool)
    _validate_ranking_inputs(score_tensor, truth_tensor)
    indices, _ = topk_unseen(
        score_tensor, _tensor(seen, dtype=torch.bool) if seen is not None else None, k
    )
    hits = truth_tensor.gather(1, indices).sum(dim=1).float()
    truth_count = truth_tensor.sum(dim=1)
    valid = truth_count > 0
    if not valid.any():
        return 0.0
    denominator = torch.minimum(
        truth_count.float(), torch.tensor(float(k), device=truth_count.device)
    )
    return float((hits[valid] / denominator[valid]).mean().item())


def ndcg_at_k(
    scores: torch.Tensor | np.ndarray,
    ground_truth: torch.Tensor | np.ndarray,
    k: int,
    seen: torch.Tensor | np.ndarray | None = None,
) -> float:
    """Mean binary NDCG@k using the same unseen ranking as Recall@k."""

    score_tensor = _tensor(scores, dtype=torch.float32)
    truth_tensor = _tensor(ground_truth, dtype=torch.bool)
    _validate_ranking_inputs(score_tensor, truth_tensor)
    indices, _ = topk_unseen(
        score_tensor, _tensor(seen, dtype=torch.bool) if seen is not None else None, k
    )
    relevance = truth_tensor.gather(1, indices).float()
    discounts = 1.0 / torch.log2(
        torch.arange(relevance.shape[1], device=relevance.device).float() + 2.0
    )
    dcg = (relevance * discounts).sum(dim=1)
    truth_count = truth_tensor.sum(dim=1)
    valid = truth_count > 0
    if not valid.any():
        return 0.0
    ideal_count = torch.minimum(truth_count, torch.tensor(k, device=truth_count.device))
    ideal = torch.zeros_like(dcg)
    for row, count in enumerate(ideal_count.tolist()):
        if count:
            ideal[row] = discounts[: int(count)].sum()
    return float((dcg[valid] / ideal[valid].clamp_min(torch.finfo(dcg.dtype).eps)).mean().item())
