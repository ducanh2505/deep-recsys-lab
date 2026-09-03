"""Ranking metrics and deterministic model evaluation."""

from .evaluator import evaluate_model, recommend
from .metrics import (
    RankingProtocol,
    ndcg_at_k,
    ranking_batch_stats,
    recall_at_k,
    seen_item_mask,
    topk_unseen,
)

__all__ = [
    "evaluate_model",
    "RankingProtocol",
    "ndcg_at_k",
    "ranking_batch_stats",
    "recall_at_k",
    "recommend",
    "seen_item_mask",
    "topk_unseen",
]
