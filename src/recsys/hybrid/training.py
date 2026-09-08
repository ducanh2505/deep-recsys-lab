"""Validation-only training for learned hybrid fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from recsys.artifacts import RuntimeSpec
from recsys.artifacts.runtimes import RuntimeContext, RuntimeQuery, load_runtime
from recsys.core.types import QueryMode
from recsys.datasets import PreparedDataset, ordered_train_item_indices
from recsys.fusion.strategies import LearnedLightGBMFusion, reciprocal_features
from recsys.retrieval.algorithms import deterministic_topk


def train_learned_fusion(
    data: PreparedDataset,
    components: list[RuntimeSpec],
    mode: QueryMode,
    scratch: Path,
    *,
    seed: int,
    parameters: dict[str, Any],
) -> Path:
    runtimes = []
    for spec in components:
        context = RuntimeContext(
            root=scratch,
            arrays=spec.arrays,
            metadata=spec.metadata,
            files=spec.files,
            n_users=data.shape[0],
            n_items=data.shape[1],
        )
        runtimes.append((spec.plugin, load_runtime(spec.runtime, context)))

    candidate_pool_size = int(parameters.get("candidate_pool_size", 100))
    rank_constant = float(parameters.get("rank_constant", 60))
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    groups: list[int] = []
    positive_count = 0
    ordered_items = ordered_train_item_indices(data)
    for user_index in range(data.shape[0]):
        truth = data.validation.indices[
            data.validation.indptr[user_index] : data.validation.indptr[user_index + 1]
        ]
        if not len(truth):
            continue
        vector = data.train.getrow(user_index).toarray().ravel().astype(np.float32)
        query = RuntimeQuery(
            mode,
            vector,
            ordered_items[user_index],
            user_index if mode is QueryMode.KNOWN_USER else None,
        )
        scores = {
            name: np.asarray(runtime.score(query), dtype=np.float64) for name, runtime in runtimes
        }
        seen = np.flatnonzero(vector)
        candidates: set[int] = set()
        for values in scores.values():
            values[seen] = -np.inf
            selected, _ = deterministic_topk(values, set(), candidate_pool_size)
            candidates.update(selected.tolist())
        if not candidates:
            continue
        selected = np.asarray(sorted(candidates), dtype=np.int64)
        features = reciprocal_features(scores, rank_constant)[selected]
        labels = np.isin(selected, truth).astype(np.int8)
        positive_count += int(labels.sum())
        feature_rows.append(features)
        label_rows.append(labels)
        groups.append(len(selected))
    if not feature_rows or positive_count == 0:
        raise ValueError("learned fusion needs validation positives in retrieved candidate pools")
    lightgbm_parameters = {
        key: value
        for key, value in parameters.items()
        if key not in {"candidate_pool_size", "rank_constant"}
    }
    trained = LearnedLightGBMFusion.fit(
        np.concatenate(feature_rows),
        np.concatenate(label_rows),
        groups,
        seed=seed,
        parameters=lightgbm_parameters,
    )
    target = scratch / "fusion_model.txt"
    trained.save(target)
    return target
