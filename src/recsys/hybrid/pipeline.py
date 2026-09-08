"""Compose arbitrary retriever plugins into one immutable hybrid artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from recsys.artifacts import RuntimeSpec
from recsys.conf.schema import FusionConfig, RerankerConfig, RetrievalConfig, TrainingConfig
from recsys.core.types import QueryMode
from recsys.datasets import PreparedDataset
from recsys.fusion import FUSION_REGISTRY
from recsys.models.base import TrainingContext
from recsys.reranking import RERANKER_REGISTRY
from recsys.retrieval import fit_retriever

from .training import train_learned_fusion


def fit_hybrid(
    data: PreparedDataset,
    retrieval: RetrievalConfig,
    fusion: FusionConfig,
    reranker: RerankerConfig,
    training: TrainingConfig,
    scratch: Path,
) -> RuntimeSpec:
    if not retrieval.components:
        raise ValueError("hybrid retrieval requires at least one component")
    if len(set(retrieval.components)) != len(retrieval.components):
        raise ValueError("hybrid retriever components must be unique")
    arrays: dict[str, np.ndarray] = {}
    files: dict[str, Path] = {}
    component_metadata: list[dict[str, Any]] = []
    fitted_components: list[RuntimeSpec] = []
    valid_modes = set(QueryMode)
    for index, name in enumerate(retrieval.components):
        component_scratch = scratch / f"component_{index}"
        context = TrainingContext(
            data=data,
            training=training,
            parameters=retrieval.parameters.get(name, {}),
            scratch=component_scratch,
        )
        spec = fit_retriever(name, context)
        fitted_components.append(spec)
        prefix = f"component_{index}_"
        arrays.update({f"{prefix}{key}": value for key, value in spec.arrays.items()})
        renamed_files: dict[str, str] = {}
        for payload_name, source in spec.files.items():
            renamed = f"component_{index}_{payload_name}"
            files[renamed] = source
            renamed_files[payload_name] = renamed
        valid_modes.intersection_update(spec.capabilities)
        component_metadata.append(
            {
                "name": name,
                "runtime": spec.runtime,
                "capabilities": [mode.value for mode in spec.capabilities],
                "array_prefix": prefix,
                "metadata": spec.metadata,
                "files": renamed_files,
            }
        )
    if not valid_modes:
        raise ValueError("configured hybrid components do not share a supported query mode")
    if fusion.name not in FUSION_REGISTRY:
        raise ValueError(f"unknown fusion strategy: {fusion.name}")
    if fusion.name == "learned_lightgbm":
        mode = sorted(valid_modes, key=lambda value: value.value)[0]
        files["fusion_model.txt"] = train_learned_fusion(
            data,
            fitted_components,
            mode,
            scratch,
            seed=training.seed,
            parameters=fusion.parameters,
        )
    if reranker.name not in RERANKER_REGISTRY:
        raise ValueError(f"unknown reranker: {reranker.name}")
    return RuntimeSpec(
        plugin="hybrid",
        runtime="hybrid",
        capabilities=tuple(sorted(valid_modes, key=lambda mode: mode.value)),
        arrays=arrays,
        files=files,
        metadata={
            "components": component_metadata,
            "fusion": {
                "name": fusion.name,
                "weights": fusion.weights,
                "parameters": fusion.parameters,
            },
            "reranker": {"name": reranker.name, "parameters": reranker.parameters},
        },
    )
