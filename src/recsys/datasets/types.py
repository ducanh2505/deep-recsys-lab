"""Prepared dataset representation shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.core.types import Identifier


@dataclass(slots=True)
class PreparedDataset:
    events: pd.DataFrame
    user_ids: tuple[Identifier, ...]
    item_ids: tuple[Identifier, ...]
    train: sp.csr_matrix
    validation: sp.csr_matrix
    test: sp.csr_matrix
    train_rows: np.ndarray
    validation_rows: np.ndarray
    test_rows: np.ndarray
    dataset_digest: str
    manifest: dict[str, Any]
    item_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    root: Path | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.train.shape[0]), int(self.train.shape[1])

    @property
    def user_index(self) -> dict[Identifier, int]:
        return {value: index for index, value in enumerate(self.user_ids)}

    @property
    def item_index(self) -> dict[Identifier, int]:
        return {value: index for index, value in enumerate(self.item_ids)}

    @property
    def train_events(self) -> pd.DataFrame:
        """Raw normalized event rows assigned to the training split."""

        return self.events.iloc[self.train_rows]
