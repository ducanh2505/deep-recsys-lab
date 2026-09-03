"""Seen-item masking and ranking metric goldens."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

RankingProtocol = Literal["multvae", "lightgcn"]


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


def _validate_ranking_protocol(protocol: RankingProtocol) -> None:
    if protocol not in {"multvae", "lightgcn"}:
        raise ValueError(f"unsupported ranking protocol: {protocol!r}")


def ranking_batch_stats(
    scores: torch.Tensor | np.ndarray,
    ground_truth: torch.Tensor | np.ndarray,
    k: int,
    seen: torch.Tensor | np.ndarray | None = None,
    *,
    protocol: RankingProtocol = "multvae",
) -> dict[str, float | int]:
    """Return batch ranking metrics and valid-user counts.

    ``multvae`` preserves the original evaluator's Recall denominator of
    ``min(k, |truth|)``. ``lightgcn`` follows the LightGCN reference metric:
    Recall is hits divided by the number of truth items, while NDCG keeps the
    usual ideal prefix denominator.
    """

    _validate_ranking_protocol(protocol)
    score_tensor = _tensor(scores, dtype=torch.float32)
    truth_tensor = _tensor(ground_truth, dtype=torch.bool)
    _validate_ranking_inputs(score_tensor, truth_tensor)
    indices, _ = topk_unseen(
        score_tensor, _tensor(seen, dtype=torch.bool) if seen is not None else None, k
    )
    relevance = truth_tensor.gather(1, indices).float()
    truth_count = truth_tensor.sum(dim=1)
    valid = truth_count > 0
    valid_count = int(valid.sum().item())
    truth_edges = int(truth_count.sum().item())
    if valid_count == 0:
        return {
            "recall": 0.0,
            "ndcg": 0.0,
            "evaluated_users": 0,
            "truth_edges": truth_edges,
        }

    hits = relevance.sum(dim=1)
    if protocol == "lightgcn":
        recall_denominator = truth_count.float()
    else:
        recall_denominator = torch.minimum(
            truth_count.float(), torch.tensor(float(k), device=truth_count.device)
        )
    recall = (hits[valid] / recall_denominator[valid]).mean()

    discounts = 1.0 / torch.log2(
        torch.arange(relevance.shape[1], device=relevance.device).float() + 2.0
    )
    dcg = (relevance * discounts).sum(dim=1)
    ideal_count = torch.minimum(truth_count, torch.tensor(k, device=truth_count.device))
    ideal = torch.zeros_like(dcg)
    for row, count in enumerate(ideal_count.tolist()):
        if count:
            ideal[row] = discounts[: int(count)].sum()
    ndcg = (dcg[valid] / ideal[valid].clamp_min(torch.finfo(dcg.dtype).eps)).mean()
    return {
        "recall": float(recall.item()),
        "ndcg": float(ndcg.item()),
        "evaluated_users": valid_count,
        "truth_edges": truth_edges,
    }


def recall_at_k(
    scores: torch.Tensor | np.ndarray,
    ground_truth: torch.Tensor | np.ndarray,
    k: int,
    seen: torch.Tensor | np.ndarray | None = None,
    *,
    protocol: RankingProtocol = "multvae",
) -> float:
    """Mean Recall@k under the requested ranking protocol."""

    return float(
        ranking_batch_stats(scores, ground_truth, k, seen, protocol=protocol)["recall"]
    )


def ndcg_at_k(
    scores: torch.Tensor | np.ndarray,
    ground_truth: torch.Tensor | np.ndarray,
    k: int,
    seen: torch.Tensor | np.ndarray | None = None,
    *,
    protocol: RankingProtocol = "multvae",
) -> float:
    """Mean binary NDCG@k using the same unseen ranking as Recall@k."""

    return float(
        ranking_batch_stats(scores, ground_truth, k, seen, protocol=protocol)["ndcg"]
    )
