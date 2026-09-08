"""Registry of every built-in candidate generator."""

from __future__ import annotations

from recsys.artifacts import RuntimeSpec
from recsys.core.registry import Registry
from recsys.models.base import TrainingContext

from .base import RetrieverPlugin
from .builtins import (
    GraphCooccurrenceRetriever,
    GraphEmbeddingRetriever,
    ItemKNNRetriever,
    MarkovRetriever,
    ModelRetriever,
    PopularityRetriever,
    SemanticRetriever,
    TfidfRetriever,
)


class RetrieverRegistry(Registry[RetrieverPlugin]):
    def __init__(self) -> None:
        super().__init__("retriever")


RETRIEVER_REGISTRY = RetrieverRegistry()
for _plugin in (
    PopularityRetriever(),
    ItemKNNRetriever(),
    TfidfRetriever(),
    SemanticRetriever(),
    MarkovRetriever(),
    GraphCooccurrenceRetriever(),
    GraphEmbeddingRetriever(),
    ModelRetriever("bpr"),
    ModelRetriever("sasrec"),
    ModelRetriever("multivae"),
    ModelRetriever("lightgcn"),
    ModelRetriever("two_tower"),
):
    RETRIEVER_REGISTRY.register(_plugin.name, _plugin)


def fit_retriever(name: str, context: TrainingContext) -> RuntimeSpec:
    return RETRIEVER_REGISTRY.get(name).fit(context)
