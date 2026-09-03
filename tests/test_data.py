from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from deep_recsys_lab.config import DatasetConfig
from deep_recsys_lab.data.dataset import InteractionDataset, load_prepared_data
from deep_recsys_lab.data.preprocess import prepare_from_rows
from deep_recsys_lab.data.types import PreparedData


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"userId": user, "movieId": movie, "rating": 5.0 if movie % 2 else 3.0}
            for user in range(1, 9)
            for movie in range(1, 9)
        ]
    )


def test_preparation_is_deterministic_and_has_no_fold_leakage(tmp_path: Path) -> None:
    config = DatasetConfig(
        name="synthetic",
        min_positive_ratings=3,
        n_validation_users=2,
        n_test_users=2,
        strict_user_counts=True,
        seed=98765,
    )
    first = prepare_from_rows(_rows(), tmp_path / "one", config=config)
    second = prepare_from_rows(_rows(), tmp_path / "two", config=config)
    assert np.array_equal(first.item_ids, second.item_ids)
    for name in (
        "train",
        "validation_fold_in",
        "validation_fold_out",
        "test_fold_in",
        "test_fold_out",
    ):
        assert np.array_equal(getattr(first, name).toarray(), getattr(second, name).toarray())
    assert first.manifest["preprocessing"]["seed"] == 98765
    assert np.all(first.validation_fold_in.multiply(first.validation_fold_out).toarray() == 0)
    assert np.all(first.test_fold_in.multiply(first.test_fold_out).toarray() == 0)
    loaded = load_prepared_data(tmp_path / "one")
    assert loaded.n_items == first.n_items
    assert loaded.manifest["files"]["train.npz"]["sha256"]


def test_preparation_rejects_insufficient_users(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eligible users"):
        prepare_from_rows(
            _rows().query("userId < 3"),
            tmp_path / "processed",
            config=DatasetConfig(
                min_positive_ratings=3,
                n_validation_users=2,
                n_test_users=2,
                strict_user_counts=True,
            ),
        )


def test_interaction_dataset_split_is_repeatable() -> None:
    matrix = np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=np.float32)
    first = InteractionDataset(matrix, eval=True, prop=0.5, seed=7)
    second = InteractionDataset(matrix, eval=True, prop=0.5, seed=7)
    assert torch_equal(first[0]["data"], second[0]["data"])
    assert torch_equal(first[0]["ground_truth"], second[0]["ground_truth"])
    assert not torch_equal(first[0]["data"], first[0]["ground_truth"])


def torch_equal(first: torch.Tensor, second: torch.Tensor) -> bool:
    return bool((first == second).all().item())


def test_prepared_checksum_is_checked(prepared_data: PreparedData) -> None:
    root = prepared_data.root
    assert root is not None
    path = root / "train.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_prepared_data(root)
