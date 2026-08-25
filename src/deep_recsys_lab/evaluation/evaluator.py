"""Evaluation loop and deterministic recommendation helper."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import torch

from ..model.base import BaseRecommender
from .metrics import ndcg_at_k, recall_at_k, topk_unseen


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
) -> dict[str, float]:
    """Evaluate only fold-out positives, masking fold-in items first."""

    model_device = torch.device(device)
    was_training = model.training
    model.eval()
    totals: dict[str, list[tuple[float, int]]] = {f"recall@{k}": [] for k in ks}
    totals.update({f"ndcg@{k}": [] for k in ks})
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            if max_batches is not None and batch_number >= max_batches:
                break
            data, truth = _batch_tensors(batch)
            data = data.to(model_device, dtype=torch.float32)
            truth = truth.to(model_device, dtype=torch.float32)
            scores = model(data, sample=False).logits
            for k in ks:
                batch_size = int(data.shape[0])
                totals[f"recall@{k}"].append(
                    (recall_at_k(scores, truth, k, data if mask_seen else None), batch_size)
                )
                totals[f"ndcg@{k}"].append(
                    (ndcg_at_k(scores, truth, k, data if mask_seen else None), batch_size)
                )
    if was_training:
        model.train()
    return {
        name: float(
            sum(value * count for value, count in values) / sum(count for _, count in values)
        )
        if values
        else 0.0
        for name, values in totals.items()
    }


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
