"""Minimal ASGI transport with health, metrics, auth, and body-size guards."""

from __future__ import annotations

import hmac
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from .engine import RecommendationEngine
from .errors import RecommendationError
from .schemas import ErrorDetail, ErrorResponse, RecommendationRequest

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class RecommendationASGI:
    def __init__(
        self,
        engine: RecommendationEngine,
        *,
        api_key: str | None = None,
        max_body_bytes: int = 65_536,
    ) -> None:
        self.engine = engine
        self.api_key = api_key or None
        self.max_body_bytes = max_body_bytes
        self.requests = 0
        self.failures = 0

    @staticmethod
    def _headers(scope: Scope) -> dict[str, str]:
        return {
            bytes(name).decode("latin-1").lower(): bytes(value).decode("latin-1")
            for name, value in scope.get("headers", [])
        }

    async def _respond(
        self,
        send: Send,
        status: int,
        body: bytes,
        request_id: str,
        content_type: str = "application/json",
    ) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", content_type.encode("ascii")),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"x-request-id", request_id.encode("latin-1", errors="replace")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def _json(self, send: Send, status: int, value: Any, request_id: str) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        await self._respond(send, status, body, request_id)

    async def _body(self, receive: Receive) -> bytes:
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            chunk = bytes(message.get("body", b""))
            size += len(chunk)
            if size > self.max_body_bytes:
                raise OverflowError("request payload is too large")
            chunks.append(chunk)
            if not message.get("more_body", False):
                return b"".join(chunks)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            return
        path = str(scope.get("path", "/")).rstrip("/") or "/"
        method = str(scope.get("method", "GET")).upper()
        headers = self._headers(scope)
        request_id = headers.get("x-request-id") or str(uuid.uuid4())
        protected = path in {"/recommend", "/model"}
        if (
            protected
            and self.api_key is not None
            and not hmac.compare_digest(headers.get("x-api-key", ""), self.api_key)
        ):
            error = ErrorResponse(
                request_id=request_id,
                error=ErrorDetail(code="unauthorized", message="missing or invalid API key"),
            )
            await self._json(send, 401, error.model_dump(mode="json"), request_id)
            return
        if protected and "content-length" in headers:
            try:
                too_large = int(headers["content-length"]) > self.max_body_bytes
            except ValueError:
                too_large = False
            if too_large:
                await self._error(
                    send, request_id, 413, "payload_too_large", "request payload is too large"
                )
                return

        if path == "/livez" and method == "GET":
            await self._json(send, 200, {"status": "live"}, request_id)
            return
        if path == "/readyz" and method == "GET":
            await self._json(send, 200, {"status": "ready"}, request_id)
            return
        if path == "/metrics" and method == "GET":
            body = (
                f"recsys_requests_total {self.requests}\nrecsys_failures_total {self.failures}\n"
            ).encode("ascii")
            await self._respond(send, 200, body, request_id, "text/plain; version=0.0.4")
            return
        if path == "/model" and method == "GET":
            await self._json(send, 200, self.engine.model_info(), request_id)
            return
        if path == "/recommend" and method == "POST":
            self.requests += 1
            try:
                raw = await self._body(receive)
                request = RecommendationRequest.model_validate_json(raw)
                response = self.engine.recommend(request, request_id=request_id)
            except OverflowError:
                self.failures += 1
                await self._error(
                    send, request_id, 413, "payload_too_large", "request payload is too large"
                )
                return
            except ValidationError as exc:
                self.failures += 1
                await self._error(send, request_id, 422, "invalid_request", str(exc))
                return
            except RecommendationError as exc:
                self.failures += 1
                await self._error(send, request_id, exc.status_code, exc.code, exc.message)
                return
            await self._json(send, 200, response.model_dump(mode="json"), request_id)
            return
        await self._error(send, request_id, 404, "not_found", "route not found")

    async def _error(
        self, send: Send, request_id: str, status: int, code: str, message: str
    ) -> None:
        error = ErrorResponse(
            request_id=request_id,
            error=ErrorDetail(code=code, message=message),
        )
        await self._json(send, status, error.model_dump(mode="json"), request_id)
