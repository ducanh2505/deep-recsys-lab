"""Provider-neutral reranker factory registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from recsys.core.registry import Registry

from .base import Reranker
from .ollama import OllamaReranker, OllamaSettings

RerankerFactory = Callable[[dict[str, Any]], Reranker | None]
RERANKER_REGISTRY: Registry[RerankerFactory] = Registry("reranker")
RERANKER_REGISTRY.register("none", lambda _: None)
RERANKER_REGISTRY.register("ollama", lambda values: OllamaReranker(OllamaSettings(**values)))


def load_reranker(name: str, parameters: dict[str, Any]) -> Reranker | None:
    return RERANKER_REGISTRY.get(name)(parameters)
