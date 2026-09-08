"""Built-in retriever plugin implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from recsys.artifacts import RuntimeSpec
from recsys.artifacts.store import csr_arrays
from recsys.core.types import QueryMode
from recsys.datasets import ordered_train_item_indices
from recsys.models import fit_model
from recsys.models.base import TrainingContext

from .algorithms import (
    cooccurrence,
    hashed_text_embeddings,
    item_knn,
    similarity_from_embeddings,
    tfidf_matrix,
    transition_matrix,
)


def _documents(context: TrainingContext) -> list[str]:
    metadata = context.data.item_metadata
    if metadata.empty:
        return [str(value) for value in context.data.item_ids]
    lookup = {
        record["item_id"]: " ".join(
            str(value) for key, value in record.items() if key != "item_id" and value is not None
        )
        for record in metadata.to_dict(orient="records")
    }
    return [lookup.get(value, str(value)) or str(value) for value in context.data.item_ids]


def _sparse_spec(name: str, matrix: sp.csr_matrix, **metadata: Any) -> RuntimeSpec:
    return RuntimeSpec(
        plugin=name,
        runtime="sparse",
        capabilities=(QueryMode.HISTORY,),
        arrays=csr_arrays("similarity", matrix),
        metadata=metadata,
    )


@dataclass(slots=True)
class PopularityRetriever:
    name: str = "popularity"

    def fit(self, context: TrainingContext) -> RuntimeSpec:
        scores = np.asarray(context.data.train.sum(axis=0)).ravel().astype(np.float32)
        return RuntimeSpec(
            plugin=self.name,
            runtime="popularity",
            capabilities=(QueryMode.KNOWN_USER, QueryMode.HISTORY),
            arrays={"global_scores": scores},
        )


@dataclass(slots=True)
class ItemKNNRetriever:
    name: str = "itemknn"

    def fit(self, context: TrainingContext) -> RuntimeSpec:
        neighbours = int(context.parameters.get("neighbours", 100))
        return _sparse_spec(self.name, item_knn(context.data.train, neighbours=neighbours))


@dataclass(slots=True)
class TfidfRetriever:
    name: str = "tfidf"

    def fit(self, context: TrainingContext) -> RuntimeSpec:
        matrix = tfidf_matrix(_documents(context))
        similarity = matrix @ matrix.T
        similarity.setdiag(0)
        similarity.eliminate_zeros()
        return _sparse_spec(self.name, similarity.tocsr(), representation="tfidf")


@dataclass(slots=True)
class SemanticRetriever:
    name: str = "semantic"

    def fit(self, context: TrainingContext) -> RuntimeSpec:
        dimension = int(context.parameters.get("embedding_dim", 128))
        neighbours = int(context.parameters.get("neighbours", 100))
        embeddings = hashed_text_embeddings(_documents(context), dimension)
        return _sparse_spec(
            self.name,
            similarity_from_embeddings(embeddings, neighbours),
            representation="feature_hash_embeddings",
        )


@dataclass(slots=True)
class MarkovRetriever:
    name: str = "markov"

    def fit(self, context: TrainingContext) -> RuntimeSpec:
        sequences = ordered_train_item_indices(context.data)
        user_indices = np.asarray(
            [user for user, values in enumerate(sequences) for _ in values], dtype=np.int64
        )
        item_indices = np.asarray([item for values in sequences for item in values], dtype=np.int64)
        timestamps = np.asarray(
            [position for values in sequences for position, _ in enumerate(values)], dtype=np.int64
        )
        transition = transition_matrix(
            user_indices, item_indices, timestamps, context.data.shape[1]
        )
        last_items = np.full(context.data.shape[0], -1, dtype=np.int64)
        for user_index, values in enumerate(sequences):
            if values:
                last_items[user_index] = values[-1]
        fallback = np.asarray(context.data.train.sum(axis=0)).ravel().astype(np.float32)
        return RuntimeSpec(
            plugin=self.name,
            runtime="sequential",
            capabilities=(QueryMode.KNOWN_USER, QueryMode.HISTORY),
            arrays={
                **csr_arrays("similarity", transition),
                "last_items": last_items,
                "global_scores": fallback,
            },
        )


@dataclass(slots=True)
class GraphCooccurrenceRetriever:
    name: str = "graph_cooccurrence"

    def fit(self, context: TrainingContext) -> RuntimeSpec:
        minimum = int(context.parameters.get("min_count", 1))
        return _sparse_spec(self.name, cooccurrence(context.data.train, minimum))


@dataclass(slots=True)
class GraphEmbeddingRetriever:
    name: str = "graph_embeddings"

    def fit(self, context: TrainingContext) -> RuntimeSpec:
        graph = cooccurrence(context.data.train)
        dense = graph.toarray().astype(np.float64)
        _, singular, right = np.linalg.svd(dense, full_matrices=False)
        take = max(1, min(int(context.parameters.get("embedding_dim", 32)), len(singular)))
        embeddings = (right[:take].T * np.sqrt(singular[:take])).astype(np.float32)
        neighbours = int(context.parameters.get("neighbours", 100))
        return _sparse_spec(
            self.name,
            similarity_from_embeddings(embeddings, neighbours),
            representation="graph_svd",
        )


@dataclass(slots=True)
class ModelRetriever:
    name: str

    def fit(self, context: TrainingContext) -> RuntimeSpec:
        spec = fit_model(self.name, context)
        spec.plugin = self.name
        return spec
