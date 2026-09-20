from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from recsys.conf.schema import ColumnsConfig, DatasetConfig, SplitConfig, TrainingConfig
from recsys.core.types import QueryMode
from recsys.datasets.prepare import prepare_frame
from recsys.datasets.types import PreparedDataset
from recsys.models import MODEL_REGISTRY, fit_model
from recsys.models.base import TrainingContext
from recsys.models.lightgcn import LightGCN, normalized_adjacency
from recsys.models.multivae import MultiVAE
from recsys.models.sampling import PairwiseSampler
from recsys.models.sasrec import _sequences


def context(
    prepared: PreparedDataset, tmp_path: Path, parameters: dict[str, object] | None = None
) -> TrainingContext:
    return TrainingContext(
        prepared,
        TrainingConfig(seed=3, epochs=1, batch_size=4, learning_rate=0.01),
        parameters or {},
        tmp_path,
    )


def test_model_registry_has_all_builtins() -> None:
    assert MODEL_REGISTRY.names() == ("bpr", "lightgcn", "multivae", "sasrec", "two_tower")


def test_multivae_forward_contract() -> None:
    model = MultiVAE(6, hidden_dim=4, latent_dim=2, dropout=0)
    model.eval()
    logits, mean, log_variance = model(torch.eye(6)[:2])
    assert logits.shape == (2, 6)
    assert mean.shape == log_variance.shape == (2, 2)
    with pytest.raises(ValueError, match="dimensions"):
        MultiVAE(0)
    with pytest.raises(ValueError, match="dropout"):
        MultiVAE(6, dropout=1)


def test_lightgcn_propagates_and_validates(prepared: PreparedDataset) -> None:
    adjacency = normalized_adjacency(prepared.train, torch.device("cpu"))
    model = LightGCN(*prepared.shape, 4, 2, adjacency)
    users, items = model.embeddings()
    assert users.shape == (prepared.shape[0], 4)
    assert items.shape == (prepared.shape[1], 4)
    with pytest.raises(ValueError, match="layers"):
        LightGCN(*prepared.shape, 4, 0, adjacency)


def test_pairwise_sampler_never_returns_seen_items(prepared: PreparedDataset) -> None:
    sampler = PairwiseSampler(prepared.train, 4)
    users, positives, negatives = sampler.sample(20)
    for user, positive, negative in zip(users, positives, negatives, strict=True):
        seen = set(
            prepared.train.indices[prepared.train.indptr[user] : prepared.train.indptr[user + 1]]
        )
        assert positive in seen
        assert negative not in seen


def test_sasrec_sequences_only_contain_training_rows(
    prepared: PreparedDataset, tmp_path: Path
) -> None:
    known, _, _ = _sequences(context(prepared, tmp_path), max_length=20)
    for user_index in range(prepared.shape[0]):
        expected = set(
            prepared.train.indices[
                prepared.train.indptr[user_index] : prepared.train.indptr[user_index + 1]
            ]
            + 1
        )
        assert set(known[user_index]) - {0} <= expected

    repeated = pd.DataFrame(
        {
            "user_id": ["u"] * 5,
            "item_id": ["a", "b", "c", "d", "a"],
            "timestamp": pd.date_range("2024-01-01", periods=5, tz="UTC"),
        }
    )
    temporal = prepare_frame(
        repeated,
        DatasetConfig(
            name="tabular",
            columns=ColumnsConfig(timestamp="timestamp"),
            split=SplitConfig(strategy="temporal", validation_ratio=0.2, test_ratio=0.2),
        ),
    )
    known, _, _ = _sequences(context(temporal, tmp_path), max_length=5)
    expected = [temporal.item_index[value] + 1 for value in ["a", "b", "c"]]
    assert known[0].tolist() == [0, 0, *expected]


@pytest.mark.parametrize(
    ("name", "runtime", "capability"),
    [
        ("bpr", "embedding", QueryMode.HISTORY),
        ("two_tower", "embedding", QueryMode.HISTORY),
        ("sasrec", "sequential_onnx", QueryMode.HISTORY),
        ("lightgcn", "embedding", QueryMode.KNOWN_USER),
    ],
)
def test_model_plugins_emit_safe_runtime_specs(
    name: str,
    runtime: str,
    capability: QueryMode,
    prepared: PreparedDataset,
    tmp_path: Path,
) -> None:
    parameters: dict[str, object] = {"embedding_dim": 4, "steps": 1, "layers": 1}
    spec = fit_model(name, context(prepared, tmp_path / name, parameters))
    assert spec.runtime == runtime
    assert capability in spec.capabilities
    if name == "lightgcn":
        assert QueryMode.HISTORY in spec.capabilities
    assert all(isinstance(value, np.ndarray) for value in spec.arrays.values())
