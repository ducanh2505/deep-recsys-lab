"""Framework-neutral recommendation engine."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np

from recsys.artifacts import LoadedArtifact, load_artifact
from recsys.artifacts.runtimes import RuntimeQuery, runtime_from_artifact
from recsys.artifacts.store import arrays_csr
from recsys.core.types import Candidate, Identifier, QueryMode
from recsys.reranking import Reranker, load_reranker
from recsys.retrieval.algorithms import deterministic_topk

from .errors import CatalogExhaustedError, UnknownIdentifierError, UnsupportedQueryModeError
from .schemas import Recommendation, RecommendationRequest, RecommendationResponse


class RecommendationEngine:
    def __init__(self, artifact: Path | LoadedArtifact) -> None:
        self.artifact = load_artifact(artifact) if isinstance(artifact, Path) else artifact
        self.runtime = runtime_from_artifact(self.artifact)
        self.train = arrays_csr(self.artifact.arrays, "train")
        self.users: dict[Identifier, int] = {
            value: index for index, value in enumerate(self.artifact.user_ids)
        }
        self.items: dict[Identifier, int] = {
            value: index for index, value in enumerate(self.artifact.item_ids)
        }
        self.reranker = self._load_reranker()

    def _load_reranker(self) -> Reranker | None:
        configured = self.artifact.manifest.metadata.get("reranker", {})
        if not isinstance(configured, dict):
            raise ValueError("reranker configuration must be a mapping")
        name = str(configured.get("name", "none"))
        parameters = configured.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("reranker parameters must be a mapping")
        try:
            return load_reranker(name, parameters)
        except KeyError as exc:
            raise ValueError(f"unsupported reranker: {name}") from exc

    def _query(self, request: RecommendationRequest) -> RuntimeQuery:
        if request.user_id is not None:
            if QueryMode.KNOWN_USER not in self.artifact.manifest.capabilities:
                raise UnsupportedQueryModeError("artifact does not support known-user queries")
            if request.user_id not in self.users:
                raise UnknownIdentifierError(f"unknown user_id: {request.user_id!r}")
            user_index = self.users[request.user_id]
            vector = self.train.getrow(user_index).toarray().ravel().astype(np.float32)
            ordered = tuple(np.flatnonzero(vector).tolist())
            return RuntimeQuery(QueryMode.KNOWN_USER, vector, ordered, user_index)

        if QueryMode.HISTORY not in self.artifact.manifest.capabilities:
            raise UnsupportedQueryModeError("artifact does not support interaction-history queries")
        vector = np.zeros(len(self.items), dtype=np.float32)
        ordered_records: list[tuple[int, int]] = []
        complete_timestamps = all(
            interaction.timestamp is not None for interaction in request.interactions
        )
        for position, interaction in enumerate(request.interactions):
            if interaction.item_id not in self.items:
                raise UnknownIdentifierError(f"unknown item_id: {interaction.item_id!r}")
            item_index = self.items[interaction.item_id]
            vector[item_index] += float(interaction.value)
            order_value = (
                int(interaction.timestamp.timestamp() * 1_000_000)
                if complete_timestamps and interaction.timestamp is not None
                else position
            )
            ordered_records.append((order_value, item_index))
        ordered_records.sort(key=lambda value: value[0])
        return RuntimeQuery(
            QueryMode.HISTORY,
            vector,
            tuple(item for _, item in ordered_records),
        )

    def recommend(
        self, request: RecommendationRequest, *, request_id: str | None = None
    ) -> RecommendationResponse:
        query = self._query(request)
        scores = np.asarray(self.runtime.score(query), dtype=np.float64)
        if scores.shape != (len(self.items),) or not np.isfinite(scores).all():
            raise RuntimeError("artifact runtime returned an invalid score vector")
        seen = np.flatnonzero(query.vector) if request.exclude_seen else np.empty(0, dtype=np.int64)
        item_indices, values = deterministic_topk(scores, seen, request.top_k)
        if not len(item_indices):
            raise CatalogExhaustedError("no recommendable items remain in the artifact catalog")
        candidates = [
            Candidate(
                int(item_index),
                float(score),
                self.artifact.item_metadata[int(item_index)],
            )
            for item_index, score in zip(item_indices, values, strict=True)
        ]
        if self.reranker is not None:
            history: list[dict[str, object]] = [
                {
                    "item_id": self.artifact.item_ids[item],
                    "value": float(query.vector[item]),
                    "metadata": self.artifact.item_metadata[item],
                }
                for item in query.ordered_items
            ]
            candidates = self.reranker.rerank(history, candidates, top_k=request.top_k)
        recommendations = [
            Recommendation(
                item_id=self.artifact.item_ids[candidate.item_index],
                score=candidate.score,
                metadata=candidate.metadata,
            )
            for candidate in candidates
        ]
        return RecommendationResponse(
            request_id=request_id or str(uuid.uuid4()),
            artifact_id=self.artifact.manifest.artifact_id,
            recommendations=recommendations,
        )

    def model_info(self) -> dict[str, Any]:
        manifest = self.artifact.manifest
        return {
            "schema": manifest.contract_schema,
            "artifact_id": manifest.artifact_id,
            "plugin": manifest.plugin,
            "capabilities": [mode.value for mode in manifest.capabilities],
            "runtime": manifest.runtime,
            "catalog_size": manifest.catalog_size,
        }
