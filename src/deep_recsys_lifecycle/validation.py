from __future__ import annotations

from collections.abc import Collection
from math import isfinite

from .artifact import ServingArtifact
from .query import (
    EmptyHistoryQuery,
    HistoryOnlyQuery,
    KnownUserQuery,
    Query,
    RecommendationService,
)


class SmokeValidationError(ValueError):
    """Raised when an exported artifact cannot answer its serving smoke queries."""


def validate_smoke_queries(artifact: ServingArtifact) -> None:
    """Exercise the same query service used by the API before activation."""

    service = RecommendationService(artifact)
    examples: tuple[tuple[Query, Collection[int]], ...] = (
        _known_user_example(artifact),
        _history_only_example(artifact),
        (EmptyHistoryQuery(), ()),
    )
    for query, observed in examples:
        first = service.recommend(query).to_dict()
        second = service.recommend(query).to_dict()
        if first != second:
            raise SmokeValidationError(f"smoke query is not deterministic: {query.mode}")
        expected_retriever = "popularity" if query.mode == "empty_history" else "lhf"
        _validate_response(
            artifact,
            first,
            query.mode,
            observed,
            expected_retriever=expected_retriever,
        )

    if artifact.manifest.get("configuration", {}).get("multivae_enabled"):
        if artifact.multivae is None:
            raise SmokeValidationError("artifact is missing the Mult-VAE payload")
        multivae_examples: tuple[tuple[Query, Collection[int]], ...] = (
            _known_user_example(artifact),
            _history_only_example(artifact),
        )
        for query, observed in multivae_examples:
            first = service.recommend_with_retriever(query, "multivae").to_dict()
            second = service.recommend_with_retriever(query, "multivae").to_dict()
            if first != second:
                raise SmokeValidationError(
                    f"Mult-VAE smoke query is not deterministic: {query.mode}"
                )
            _validate_response(artifact, first, query.mode, observed, expected_retriever="multivae")

    if artifact.manifest.get("configuration", {}).get("itemknn_enabled"):
        if artifact.itemknn is None:
            raise SmokeValidationError("artifact is missing the ItemKNN payload")
        itemknn_examples: tuple[tuple[Query, Collection[int]], ...] = (
            _known_user_example(artifact),
            _history_only_example(artifact),
        )
        for query, observed in itemknn_examples:
            first = service.recommend_with_retriever(query, "itemknn").to_dict()
            second = service.recommend_with_retriever(query, "itemknn").to_dict()
            if first != second:
                raise SmokeValidationError(
                    f"ItemKNN smoke query is not deterministic: {query.mode}"
                )
            _validate_response(
                artifact,
                first,
                query.mode,
                observed,
                expected_retriever="itemknn",
            )

    if artifact.manifest.get("configuration", {}).get("lightgcn_enabled"):
        if artifact.lightgcn is None:
            raise SmokeValidationError("artifact is missing the LightGCN payload")
        query, observed = _known_user_example(artifact)
        first = service.recommend_with_retriever(query, "lightgcn").to_dict()
        second = service.recommend_with_retriever(query, "lightgcn").to_dict()
        if first != second:
            raise SmokeValidationError("LightGCN smoke query is not deterministic")
        _validate_response(artifact, first, query.mode, observed, expected_retriever="lightgcn")

    health = artifact.health_metadata()
    if health.get("status") != "ok":
        raise SmokeValidationError("artifact health metadata is not healthy")
    if health.get("artifact_id") != artifact.artifact_id:
        raise SmokeValidationError("artifact health metadata has the wrong artifact identity")
    if health.get("artifact_fingerprint") != artifact.artifact_fingerprint:
        raise SmokeValidationError("artifact health metadata has the wrong fingerprint")


def _known_user_example(artifact: ServingArtifact) -> tuple[Query, Collection[int]]:
    if not artifact.model.subject_histories:
        raise SmokeValidationError("artifact has no Subject history for KnownUser smoke query")
    subject_id = min(artifact.model.subject_histories)
    history = artifact.model.subject_histories[subject_id]
    return KnownUserQuery(subject_id=subject_id), history


def _history_only_example(artifact: ServingArtifact) -> tuple[Query, Collection[int]]:
    history = (artifact.model.catalog[0],) if artifact.model.catalog else ()
    return HistoryOnlyQuery(movie_ids=history), history


def _validate_response(
    artifact: ServingArtifact,
    response: dict[str, object],
    mode: str,
    observed: Collection[int],
    *,
    expected_retriever: str = "popularity",
) -> None:
    if response.get("query_mode") != mode or response.get("retriever") != expected_retriever:
        raise SmokeValidationError(f"smoke response has an invalid schema for {mode}")
    provenance = response.get("provenance")
    if not isinstance(provenance, dict):
        raise SmokeValidationError("smoke response is missing provenance")
    if provenance.get("artifact_id") != artifact.artifact_id:
        raise SmokeValidationError("smoke response has the wrong artifact identity")
    if provenance.get("artifact_fingerprint") != artifact.artifact_fingerprint:
        raise SmokeValidationError("smoke response has the wrong artifact fingerprint")

    candidates = response.get("candidates")
    if not isinstance(candidates, list):
        raise SmokeValidationError("smoke response candidates must be a list")
    observed_ids = set(observed)
    candidate_ids: set[int] = set()
    for expected_rank, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict) or set(candidate) != {"movie_id", "score", "rank"}:
            raise SmokeValidationError("smoke response has an invalid candidate schema")
        movie_id = candidate["movie_id"]
        rank = candidate["rank"]
        score = candidate["score"]
        if (
            not isinstance(movie_id, int)
            or isinstance(movie_id, bool)
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not isfinite(float(score))
        ):
            raise SmokeValidationError("smoke response candidate types are invalid")
        if rank != expected_rank or movie_id in observed_ids or movie_id in candidate_ids:
            raise SmokeValidationError("smoke response violates ordering or history exclusion")
        candidate_ids.add(movie_id)
