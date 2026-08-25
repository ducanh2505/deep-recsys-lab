"""Native BentoML recommendation Service and its minimal ASGI guard."""

from __future__ import annotations

import hmac
import json
import os
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import bentoml
import numpy as np
from bentoml.exceptions import InvalidArgument

from .model_store import (
    ARTIFACT_SCHEMA_VERSION,
    EXECUTION_PROVIDER,
    INFERENCE_BACKEND,
    MODEL_NAME,
    ONNX_OPSET_VERSION,
    LoadedModel,
    load_model_artifact,
)
from .ranking import topk_unseen
from .schemas import (
    BatchRecommendationResult,
    ModelResponse,
    Recommendation,
    RecommendationRequest,
    RecommendationResponse,
)

MAX_CONTENT_LENGTH = 64 * 1024
PROTECTED_PATHS = frozenset({"/recommend", "/model_info"})
INFERENCE_SERVICE_NAME = "deep_recsys_inference"
ADAPTIVE_MAX_BATCH_SIZE = 8
ADAPTIVE_MAX_LATENCY_MS = 10
UNKNOWN_MOVIE_ERROR = "movie_ids contains an unknown MovieLens ID"
CATALOG_EXHAUSTED_ERROR = "history contains every item in the model catalog"

ASGIScope = dict[str, Any]
ASGIMessage = dict[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[ASGIScope, ASGIReceive, ASGISend], Awaitable[None]]


class _PayloadTooLarge(Exception):
    """Internal control-flow exception for chunked request bodies."""


def _header_map(scope: ASGIScope) -> dict[str, str]:
    return {
        bytes(name).decode("latin-1").lower(): bytes(value).decode("latin-1")
        for name, value in scope.get("headers", [])
    }


class ServingMiddleware:
    """Apply API-key, body-size, and request-ID policy without another framework."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        configured_key = os.getenv("DEEP_RECSYS_API_KEY")
        self.api_key = configured_key if configured_key else None

    @staticmethod
    async def _json_response(
        send: ASGISend,
        status: int,
        payload: dict[str, str],
        request_id: str,
        *,
        authenticate: bool = False,
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"x-request-id", request_id.encode("latin-1", errors="replace")),
        ]
        if authenticate:
            headers.append((b"www-authenticate", b"ApiKey"))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: ASGIScope, receive: ASGIReceive, send: ASGISend) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        scope = dict(scope)
        path = str(scope.get("path", "/")).rstrip("/") or "/"
        body_limited = path in PROTECTED_PATHS
        headers = _header_map(scope)
        request_id = headers.get("x-request-id") or str(uuid.uuid4())
        if "x-request-id" not in headers:
            scope["headers"] = [
                *scope.get("headers", []),
                (b"x-request-id", request_id.encode("latin-1", errors="replace")),
            ]

        if path in PROTECTED_PATHS and self.api_key is not None:
            supplied_key = headers.get("x-api-key", "")
            if not hmac.compare_digest(supplied_key, self.api_key):
                await self._json_response(
                    send,
                    401,
                    {"error": "missing or invalid API key"},
                    request_id,
                    authenticate=True,
                )
                return

        if body_limited:
            content_length = headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > MAX_CONTENT_LENGTH:
                        await self._json_response(
                            send,
                            413,
                            {"error": "request payload is too large"},
                            request_id,
                        )
                        return
                except ValueError:
                    await self._json_response(
                        send, 400, {"error": "content-length is invalid"}, request_id
                    )
                    return

        received_bytes = 0

        async def limited_receive() -> ASGIMessage:
            nonlocal received_bytes
            message = await receive()
            if body_limited and message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > MAX_CONTENT_LENGTH:
                    raise _PayloadTooLarge
            return message

        async def correlated_send(message: ASGIMessage) -> None:
            if message.get("type") == "http.response.start":
                response_headers = list(message.get("headers", []))
                if not any(bytes(name).lower() == b"x-request-id" for name, _ in response_headers):
                    response_headers.append(
                        (b"x-request-id", request_id.encode("latin-1", errors="replace"))
                    )
                message = {**message, "headers": response_headers}
            await send(message)

        try:
            await self.app(scope, limited_receive, correlated_send)
        except _PayloadTooLarge:
            await self._json_response(
                send, 413, {"error": "request payload is too large"}, request_id
            )


class Predictor:
    """Deterministic recommendation logic over a validated Bento model artifact."""

    def __init__(self, loaded: LoadedModel) -> None:
        self.loaded = loaded
        self.positions = {
            int(movie_id): index for index, movie_id in enumerate(loaded.item_ids.tolist())
        }

    def _records(self, indices: np.ndarray, values: np.ndarray) -> list[Recommendation]:
        records: list[Recommendation] = []
        for item_index, score in zip(indices.tolist(), values.tolist(), strict=True):
            movie_id = int(self.loaded.item_ids[item_index])
            metadata = self.loaded.metadata[movie_id]
            records.append(
                Recommendation(
                    movie_id=movie_id,
                    title=str(metadata.get("title", "")),
                    genres=str(metadata.get("genres", "")),
                    score=float(score),
                )
            )
        return records

    def recommend_batch(
        self, requests: list[RecommendationRequest]
    ) -> list[BatchRecommendationResult]:
        """Score all valid histories once and isolate request-specific failures."""

        if not requests:
            return []

        model_version = self.loaded.model_version
        n_items = self.loaded.scorer.n_items
        results: dict[int, BatchRecommendationResult] = {}
        valid: list[tuple[int, np.ndarray, int]] = []

        for request_index, request in enumerate(requests):
            unknown = [movie_id for movie_id in request.movie_ids if movie_id not in self.positions]
            if unknown:
                results[request_index] = BatchRecommendationResult(
                    model_version=model_version,
                    error=UNKNOWN_MOVIE_ERROR,
                )
                continue

            available = n_items - len(request.movie_ids)
            if available <= 0:
                results[request_index] = BatchRecommendationResult(
                    model_version=model_version,
                    error=CATALOG_EXHAUSTED_ERROR,
                )
                continue

            positions = np.fromiter(
                (self.positions[movie_id] for movie_id in request.movie_ids),
                dtype=np.int64,
                count=len(request.movie_ids),
            )
            valid.append((request_index, positions, min(request.top_k, available)))

        if valid:
            interactions = np.zeros((len(valid), n_items), dtype=np.float32)
            lengths = np.fromiter(
                (len(positions) for _, positions, _ in valid),
                dtype=np.int64,
                count=len(valid),
            )
            row_indices = np.repeat(np.arange(len(valid), dtype=np.int64), lengths)
            column_indices = np.concatenate([positions for _, positions, _ in valid])
            interactions[row_indices, column_indices] = 1.0
            scores = self.loaded.scorer.score(interactions)

            for batch_index, (request_index, _, effective_k) in enumerate(valid):
                indices, values = topk_unseen(
                    scores[batch_index : batch_index + 1],
                    interactions[batch_index : batch_index + 1],
                    effective_k,
                )
                results[request_index] = BatchRecommendationResult(
                    model_version=model_version,
                    recommendations=self._records(indices[0], values[0]),
                )

        return [results[index] for index in range(len(requests))]

    def recommend(self, movie_ids: list[int], top_k: int) -> list[Recommendation]:
        request = RecommendationRequest.model_construct(movie_ids=movie_ids, top_k=top_k)
        result = self.recommend_batch([request])[0]
        if result.error is not None:
            raise InvalidArgument(result.error)
        return result.recommendations


def validate_explicit_model_tag(model_tag: str) -> str:
    """Require the fixed model name and an immutable, non-latest version."""

    tag = model_tag.strip()
    name, separator, version = tag.partition(":")
    if name != MODEL_NAME or separator != ":" or not version or version.lower() == "latest":
        raise ValueError(f"model_tag must be explicit and match {MODEL_NAME}:<version>")
    return tag


SERVING_IMAGE = bentoml.images.Image(python_version="3.12").python_packages(
    "--index-url https://pypi.org/simple",
    "bentoml==1.4.39",
    "numpy==2.5.2",
    "onnxruntime==1.29.0",
    "pydantic==2.13.4",
)


def create_bento_service(model_tag: str) -> Any:
    """Create the deployable Service for one explicit Bento model version."""

    resolved_tag = validate_explicit_model_tag(model_tag)
    artifact_reference = bentoml.models.BentoModel(resolved_tag)

    @bentoml.service(
        name=INFERENCE_SERVICE_NAME,
        workers=1,
        traffic={"timeout": 60, "max_concurrency": ADAPTIVE_MAX_BATCH_SIZE},
        metrics={"enabled": True},
    )
    class RecommendationInferenceService:
        model_ref = artifact_reference

        def __init__(self) -> None:
            resolved_model = self.model_ref
            loaded = load_model_artifact(resolved_model.path, model_tag=str(resolved_model.tag))
            self.predictor = Predictor(loaded)

        @bentoml.api(  # type: ignore[untyped-decorator]
            batchable=True,
            max_batch_size=ADAPTIVE_MAX_BATCH_SIZE,
            max_latency_ms=ADAPTIVE_MAX_LATENCY_MS,
        )
        def recommend_batch(
            self, requests: list[RecommendationRequest]
        ) -> list[BatchRecommendationResult]:
            return self.predictor.recommend_batch(requests)

        @bentoml.api  # type: ignore[untyped-decorator]
        def model_info(self) -> ModelResponse:
            loaded = self.predictor.loaded
            return ModelResponse(
                model_version=loaded.model_version,
                model_tag=loaded.model_tag,
                schema_version=ARTIFACT_SCHEMA_VERSION,
                item_count=len(loaded.item_ids),
                mapping_hash=str(loaded.manifest["mapping"]["sha256"]),
                score_note=str(loaded.manifest["uncalibrated_score_note"]),
                inference_backend=INFERENCE_BACKEND,
                execution_provider=EXECUTION_PROVIDER,
                onnx_opset=ONNX_OPSET_VERSION,
            )

    @bentoml.service(
        name="deep_recsys_service",
        image=SERVING_IMAGE,
        workers=1,
        traffic={"timeout": 60, "max_concurrency": 8},
        metrics={"enabled": True},
    )
    class RecommendationService:
        inference = bentoml.depends(RecommendationInferenceService)

        @bentoml.api(  # type: ignore[untyped-decorator]
            route="/recommend", input_spec=RecommendationRequest
        )
        async def recommend(
            self, context: bentoml.Context, **parameters: Any
        ) -> RecommendationResponse:
            body = RecommendationRequest.model_validate(parameters)
            request_id = context.request.headers.get("X-Request-ID") or str(uuid.uuid4())
            results = await self.inference.to_async.recommend_batch([body])
            result = results[0]
            if result.error is not None:
                raise InvalidArgument(result.error)
            return RecommendationResponse(
                request_id=request_id,
                model_version=result.model_version,
                recommendations=result.recommendations,
            )

        @bentoml.api(route="/model_info")  # type: ignore[untyped-decorator]
        async def model_info(self) -> ModelResponse:
            return await self.inference.to_async.model_info()  # type: ignore[no-any-return]

    RecommendationService.add_asgi_middleware(ServingMiddleware)  # type: ignore[attr-defined]
    return RecommendationService
