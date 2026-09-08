"""Provider-neutral reranking interfaces and optional adapters."""

from .base import Reranker
from .ollama import OllamaReranker, OllamaSettings
from .registry import RERANKER_REGISTRY, load_reranker

__all__ = [
    "OllamaReranker",
    "OllamaSettings",
    "RERANKER_REGISTRY",
    "Reranker",
    "load_reranker",
]
