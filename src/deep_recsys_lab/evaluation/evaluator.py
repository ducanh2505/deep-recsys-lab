"""Evaluation loop and deterministic recommendation helper."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import torch

from ..model.base import BaseRecommender
from .metrics import RankingProtocol, ndcg_at_k, ranking_batch_stats, recall_at_k, topk_unseen


def _batch_tensors(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        data = cast(torch.Tensor, batch["data"])
        truth = cast(torch.Tensor, batch.get("ground_truth", data))
        return data, truth
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return cast(torch.Tensor, batch[0]), cast(torch.Tensor, batch[1])
    raise TypeError("evaluation batches must be mappings with data/ground_truth")


def evaluate_model(
    model: BaseRecommender,
    loader: Iterable[Any],
    *,
    device: torch.device | str = "cpu",
    ks: tuple[int, ...] = (20, 50, 100),
    mask_seen: bool = True,
    max_batches: int | None = None,
    ranking_protocol: RankingProtocol = "multvae",
) -> dict[str, float]:
    """Evaluate only fold-out positives, masking fold-in items first."""

    model_device = torch.device(device)
    was_training = model.training
    model.eval()
    totals: dict[str, list[float]] = {f"recall@{k}": [0.0, 0.0] for k in ks}
    totals.update({f"ndcg@{k}": [0.0, 0.0] for k in ks})
    evaluated_users = 0
    truth_edges = 0
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            if max_batches is not None and batch_number >= max_batches:
                break
            data, truth = _batch_tensors(batch)
            data = data.to(model_device, dtype=torch.float32)
            truth = truth.to(model_device, dtype=torch.float32)
            scores = model(data, sample=False).logits
            for k in ks:
                stats = ranking_batch_stats(
                    scores,
                    truth,
                    k,
                    data if mask_seen else None,
                    protocol=ranking_protocol,
                )
                weight = (
                    int(stats["evaluated_users"])
                    if ranking_protocol == "lightgcn"
                    else int(data.shape[0])
                )
                totals[f"recall@{k}"][0] += float(stats["recall"]) * weight
                totals[f"recall@{k}"][1] += weight
                totals[f"ndcg@{k}"][0] += float(stats["ndcg"]) * weight
                totals[f"ndcg@{k}"][1] += weight
                if ranking_protocol == "lightgcn" and k == ks[0]:
                    evaluated_users += int(stats["evaluated_users"])
                    truth_edges += int(stats["truth_edges"])
    if was_training:
        model.train()
    result = {
        name: float(values[0] / values[1]) if values[1] > 0 else 0.0
        for name, values in totals.items()
    }
    if ranking_protocol == "lightgcn":
        result["evaluated_users"] = float(evaluated_users)
        result["truth_edges"] = float(truth_edges)
    return result


def recommend(
    model: BaseRecommender,
    interactions: torch.Tensor | np.ndarray,
    *,
    k: int,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted, unseen recommendations for each interaction row."""

    data = torch.as_tensor(interactions, dtype=torch.float32, device=device)
    with torch.inference_mode():
        scores = model.to(device).eval()(data, sample=False).logits
        indices, values = topk_unseen(scores, data, k)
    return indices.cpu().numpy(), values.cpu().numpy()


# Compatibility aliases used by the preserved notebook's original vocabulary.
Recall_at_k = recall_at_k
NDCG_at_k = ndcg_at_k
