"""BentoML adapter for the framework-neutral recommendation engine."""

from __future__ import annotations

import os
from pathlib import Path

import bentoml

from .engine import RecommendationEngine
from .http import RecommendationASGI


def _application() -> RecommendationASGI:
    artifact = os.environ.get("RECSYS_ARTIFACT")
    if not artifact:
        raise RuntimeError("RECSYS_ARTIFACT must point to an immutable artifact directory")
    maximum = int(os.environ.get("RECSYS_MAX_BODY_BYTES", "65536"))
    return RecommendationASGI(
        RecommendationEngine(Path(artifact)),
        api_key=os.environ.get("RECSYS_API_KEY"),
        max_body_bytes=maximum,
    )


@bentoml.asgi_app(_application(), path="/")
@bentoml.service(name="recsys", workers=1, metrics={"enabled": True})
class RecommendationService:
    """Thin Bento lifecycle wrapper; artifacts stay outside the Bento Model Store."""
