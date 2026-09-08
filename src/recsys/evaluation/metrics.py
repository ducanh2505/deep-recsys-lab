"""Catalog-independent top-k ranking metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable


def ranking_metrics(
    recommendations: Iterable[list[int]], truth: Iterable[set[int]], *, k: int
) -> dict[str, float | int]:
    recalls: list[float] = []
    ndcgs: list[float] = []
    for ranked, relevant in zip(recommendations, truth, strict=True):
        if not relevant:
            continue
        selected = ranked[:k]
        hits = [1.0 if item in relevant else 0.0 for item in selected]
        recalls.append(sum(hits) / len(relevant))
        dcg = sum(hit / math.log2(position + 2) for position, hit in enumerate(hits))
        ideal = sum(1.0 / math.log2(position + 2) for position in range(min(k, len(relevant))))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    return {
        f"recall@{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "evaluated_users": len(recalls),
    }
