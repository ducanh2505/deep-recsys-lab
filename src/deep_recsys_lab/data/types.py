"""Typed data artifact descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class PreparedData:
    """All matrices and mappings needed by training, evaluation, and registration."""

    train: csr_matrix
    validation_fold_in: csr_matrix
    validation_fold_out: csr_matrix
    test_fold_in: csr_matrix
    test_fold_out: csr_matrix
    item_ids: np.ndarray
    movies: list[dict[str, Any]]
    manifest: dict[str, Any]
    root: Path | None = None

    @property
    def n_items(self) -> int:
        return int(self.train.shape[1])

    @property
    def n_train_users(self) -> int:
        return int(self.train.shape[0])


MATRIX_FILES = {
    "train": "train.npz",
    "validation_fold_in": "validation_fold_in.npz",
    "validation_fold_out": "validation_fold_out.npz",
    "test_fold_in": "test_fold_in.npz",
    "test_fold_out": "test_fold_out.npz",
}
