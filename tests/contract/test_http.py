from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from recsys.artifacts import LoadedArtifact
from recsys.serving import RecommendationEngine
from recsys.serving.http import RecommendationASGI


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.asyncio
async def test_health_model_metrics_and_auth(
    popularity_artifact: Callable[..., LoadedArtifact],
) -> None:
    artifact = popularity_artifact()
    app = RecommendationASGI(RecommendationEngine(artifact), api_key="secret")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/livez")).json() == {"status": "live"}
        assert (await client.get("/readyz")).status_code == 200
        assert "recsys_requests_total" in (await client.get("/metrics")).text
        assert (await client.get("/model")).status_code == 401
        model = await client.get("/model", headers={"x-api-key": "secret"})
        assert model.json()["artifact_id"] == artifact.manifest.artifact_id


@pytest.mark.asyncio
async def test_recommend_request_id_errors_and_body_limit(
    popularity_artifact: Callable[..., LoadedArtifact],
) -> None:
    artifact = popularity_artifact()
    app = RecommendationASGI(RecommendationEngine(artifact), max_body_bytes=180)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/recommend",
            headers={"x-request-id": "correlation"},
            json={"user_id": artifact.user_ids[0], "top_k": 2},
        )
        assert response.status_code == 200
        assert response.headers["x-request-id"] == "correlation"
        assert response.json()["request_id"] == "correlation"

        invalid = await client.post("/recommend", json={})
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "invalid_request"

        unknown = await client.post("/recommend", json={"user_id": "not-known"})
        assert unknown.status_code == 422
        assert unknown.json()["error"]["code"] == "unknown_identifier"

        large = await client.post(
            "/recommend",
            content=b"{" + b'"padding":"' + b"x" * 300 + b'"}',
            headers={"content-type": "application/json"},
        )
        assert large.status_code == 413
        assert large.json()["error"]["code"] == "payload_too_large"
        assert (await client.get("/missing")).status_code == 404
        assert "recsys_failures_total" in (await client.get("/metrics")).text
