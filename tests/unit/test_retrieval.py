from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recsys.conf.schema import TrainingConfig
from recsys.datasets.types import PreparedDataset
from recsys.models.base import TrainingContext
from recsys.retrieval import RETRIEVER_REGISTRY, fit_retriever
from recsys.retrieval.algorithms import (
    deterministic_topk,
    hashed_text_embeddings,
    item_knn,
    similarity_from_embeddings,
    tfidf_matrix,
)


def test_deterministic_topk_masks_and_breaks_ties() -> None:
    indices, values = deterministic_topk(np.array([1.0, 2.0, 2.0, np.nan]), {1}, 3)
    assert indices.tolist() == [2, 0]
    assert values.tolist() == [2.0, 1.0]
    empty, _ = deterministic_topk(np.array([1.0]), {0}, 4)
    assert not len(empty)


def test_text_representations_are_finite_and_deterministic() -> None:
    documents = ["red apple", "green apple", "blue sky"]
    tfidf = tfidf_matrix(documents)
    assert tfidf.shape[0] == 3
    embeddings = hashed_text_embeddings(documents, 16)
    assert np.array_equal(embeddings, hashed_text_embeddings(documents, 16))
    assert similarity_from_embeddings(embeddings, neighbours=2).shape == (3, 3)
    with pytest.raises(ValueError, match="dimension"):
        hashed_text_embeddings(documents, 1)


def test_item_knn_has_square_item_graph(prepared: PreparedDataset) -> None:
    similarity = item_knn(prepared.train, neighbours=3, block_size=2)
    assert similarity.shape == (prepared.shape[1], prepared.shape[1])
    assert np.allclose(similarity.diagonal(), 0)


def test_retriever_registry_and_classical_plugins(
    prepared: PreparedDataset, tmp_path: Path
) -> None:
    expected = {
        "popularity",
        "itemknn",
        "tfidf",
        "semantic",
        "markov",
        "graph_cooccurrence",
        "graph_embeddings",
        "bpr",
        "sasrec",
        "multivae",
        "lightgcn",
        "two_tower",
    }
    assert set(RETRIEVER_REGISTRY.names()) == expected
    for name in expected - {"bpr", "sasrec", "multivae", "lightgcn", "two_tower"}:
        spec = fit_retriever(
            name,
            TrainingContext(
                prepared,
                TrainingConfig(epochs=1, batch_size=4),
                {"neighbours": 3, "embedding_dim": 4},
                tmp_path / name,
            ),
        )
        assert spec.plugin == name
        if name == "markov":
            assert spec.runtime == "sequential"
            assert {mode.value for mode in spec.capabilities} == {"known_user", "history"}
