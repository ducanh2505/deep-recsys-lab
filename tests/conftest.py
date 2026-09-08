from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from recsys.artifacts import LoadedArtifact, RuntimeSpec, write_artifact
from recsys.conf.schema import ColumnsConfig, DatasetConfig, SplitConfig
from recsys.core.paths import WorkspacePaths
from recsys.core.types import QueryMode
from recsys.datasets.prepare import prepare_frame
from recsys.datasets.types import PreparedDataset


@pytest.fixture
def interaction_frame() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for user in range(5):
        for offset in range(6):
            item = (user * 2 + offset) % 10
            records.append(
                {
                    "user_id": user if user % 2 == 0 else f"user-{user}",
                    "item_id": item if item % 2 == 0 else f"item-{item}",
                    "value": float(offset + 1),
                    "timestamp": pd.Timestamp("2024-01-01", tz="UTC")
                    + pd.Timedelta(user * 10 + offset, unit="D"),
                    "category": f"category-{item % 3}",
                }
            )
    return pd.DataFrame(records)


@pytest.fixture
def prepared(interaction_frame: pd.DataFrame) -> PreparedDataset:
    config = DatasetConfig(
        name="tabular",
        columns=ColumnsConfig(value="value", timestamp="timestamp"),
        split=SplitConfig(strategy="temporal", validation_ratio=0.2, test_ratio=0.2),
    )
    return prepare_frame(interaction_frame, config)


@pytest.fixture
def popularity_artifact(tmp_path: Path, prepared: PreparedDataset) -> Callable[..., LoadedArtifact]:
    def factory(*, capabilities: tuple[QueryMode, ...] = tuple(QueryMode)) -> LoadedArtifact:
        scores = prepared.train.sum(axis=0).A1.astype("float32")
        return write_artifact(
            tmp_path / "models",
            RuntimeSpec("popularity", "popularity", capabilities, {"global_scores": scores}),
            prepared,
            {"model": {"name": "popularity"}},
        )

    return factory


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    return WorkspacePaths.from_value(tmp_path).ensure()
