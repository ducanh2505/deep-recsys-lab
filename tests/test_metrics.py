from __future__ import annotations

import numpy as np
import torch

from deep_recsys_lab.evaluation.metrics import ndcg_at_k, recall_at_k, seen_item_mask, topk_unseen


def test_seen_mask_and_sorted_topk() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.8, 0.7]])
    seen = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    masked = seen_item_mask(scores, seen)
    assert masked[0, 0].item() == float("-inf")
    assert np.isclose(scores[0, 0].item(), 0.1)
    indices, values = topk_unseen(scores, seen, 3)
    assert indices.tolist() == [[1, 2, 3]]
    assert np.allclose(values.numpy(), [[0.9, 0.8, 0.7]])


def test_metric_goldens_with_seen_mask() -> None:
    scores = np.array([[0.1, 0.9, 0.8, 0.7]], dtype=np.float32)
    seen = np.array([[1, 0, 0, 0]], dtype=np.float32)
    truth = np.array([[0, 0, 1, 0]], dtype=np.float32)
    assert recall_at_k(scores, truth, 1, seen) == 0.0
    assert recall_at_k(scores, truth, 2, seen) == 1.0
    assert np.isclose(ndcg_at_k(scores, truth, 2, seen), 1.0 / np.log2(3.0))


def test_empty_ground_truth_is_zero() -> None:
    scores = torch.randn(2, 5)
    truth = torch.zeros(2, 5)
    assert recall_at_k(scores, truth, 3) == 0.0
    assert ndcg_at_k(scores, truth, 3) == 0.0
