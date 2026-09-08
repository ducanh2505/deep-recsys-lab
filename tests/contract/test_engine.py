from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from recsys.artifacts import LoadedArtifact
from recsys.core.types import QueryMode
from recsys.serving import RecommendationEngine, RecommendationRequest
from recsys.serving.errors import (
    CatalogExhaustedError,
    UnknownIdentifierError,
    UnsupportedQueryModeError,
)


def test_request_requires_exactly_one_mode() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        RecommendationRequest()
    with pytest.raises(ValidationError, match="exactly one"):
        RecommendationRequest(user_id="u", interactions=[{"item_id": "i"}])
    with pytest.raises(ValidationError):
        RecommendationRequest(user_id=True)


def test_engine_known_user_response_includes_identity_and_metadata(
    popularity_artifact: Callable[..., LoadedArtifact],
) -> None:
    artifact = popularity_artifact()
    engine = RecommendationEngine(artifact)
    response = engine.recommend(
        RecommendationRequest(user_id=artifact.user_ids[0], top_k=3),
        request_id="request-1",
    )
    assert response.request_id == "request-1"
    assert response.artifact_id == artifact.manifest.artifact_id
    assert len(response.recommendations) == 3
    assert "category" in response.recommendations[0].metadata


def test_engine_history_unknown_ids_and_capabilities(
    popularity_artifact: Callable[..., LoadedArtifact],
) -> None:
    artifact = popularity_artifact()
    engine = RecommendationEngine(artifact)
    response = engine.recommend(
        RecommendationRequest(interactions=[{"item_id": artifact.item_ids[0]}], top_k=2)
    )
    assert all(value.item_id != artifact.item_ids[0] for value in response.recommendations)
    with pytest.raises(UnknownIdentifierError, match="user_id"):
        engine.recommend(RecommendationRequest(user_id="missing"))
    with pytest.raises(UnknownIdentifierError, match="item_id"):
        engine.recommend(RecommendationRequest(interactions=[{"item_id": "missing"}]))

    history_only = RecommendationEngine(popularity_artifact(capabilities=(QueryMode.HISTORY,)))
    with pytest.raises(UnsupportedQueryModeError, match="known-user"):
        history_only.recommend(RecommendationRequest(user_id=artifact.user_ids[0]))


def test_engine_catalog_exhaustion(popularity_artifact: Callable[..., LoadedArtifact]) -> None:
    artifact = popularity_artifact()
    engine = RecommendationEngine(artifact)
    interactions = [{"item_id": value} for value in artifact.item_ids]
    with pytest.raises(CatalogExhaustedError):
        engine.recommend(RecommendationRequest(interactions=interactions))
    response = engine.recommend(
        RecommendationRequest(interactions=interactions, exclude_seen=False, top_k=2)
    )
    assert len(response.recommendations) == 2


def test_model_info_contract(popularity_artifact: Callable[..., LoadedArtifact]) -> None:
    engine = RecommendationEngine(popularity_artifact())
    info = engine.model_info()
    assert set(info) == {
        "schema",
        "artifact_id",
        "plugin",
        "capabilities",
        "runtime",
        "catalog_size",
    }
