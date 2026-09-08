from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from recsys.conf.schema import ColumnsConfig, DatasetConfig, SplitConfig
from recsys.datasets.mapping import decode_identifier, deterministic_mapping, encode_identifier
from recsys.datasets.prepare import prepare_frame
from recsys.datasets.schema import Interaction
from recsys.datasets.splitting import split_rows
from recsys.datasets.store import DatasetIntegrityError, load_dataset, save_dataset
from recsys.datasets.tabular import normalize_interactions, read_table


def test_interaction_schema_is_strict() -> None:
    assert Interaction(user_id="u", item_id=4).value == 1.0
    with pytest.raises(ValidationError):
        Interaction(user_id=True, item_id=4)
    with pytest.raises(ValidationError):
        Interaction(user_id="", item_id=4)
    with pytest.raises(ValidationError):
        Interaction(user_id="u", item_id=4, value=float("inf"))


def test_identifier_mapping_preserves_types() -> None:
    assert deterministic_mapping(["2", 2, 1, "1"], field="id") == (1, 2, "1", "2")
    encoded = encode_identifier(4)
    assert decode_identifier(encoded) == 4
    with pytest.raises(ValueError, match="boolean"):
        deterministic_mapping([True], field="id")
    with pytest.raises(ValueError, match="serialized"):
        decode_identifier({"type": "integer", "value": True})


def test_tabular_normalization_defaults_optional_fields(tmp_path: Path) -> None:
    frame = pd.DataFrame({"account": [1, 2], "product": ["a", "b"]})
    normalized = normalize_interactions(frame, ColumnsConfig(user_id="account", item_id="product"))
    assert normalized["value"].tolist() == [1.0, 1.0]
    assert "timestamp" not in normalized
    source = tmp_path / "events.csv"
    frame.to_csv(source, index=False)
    assert len(read_table(source, "csv")) == 2
    with pytest.raises(ValueError, match="format"):
        read_table(source, "json")
    with pytest.raises(ValueError, match="missing columns"):
        normalize_interactions(frame, ColumnsConfig())
    colliding = frame.assign(user_id=["shadow", "shadow"])
    with pytest.raises(ValueError, match="duplicate canonical"):
        normalize_interactions(
            colliding,
            ColumnsConfig(user_id="account", item_id="product"),
        )
    with pytest.raises(ValueError, match="overlapping source"):
        normalize_interactions(frame, ColumnsConfig(user_id="account", item_id="account"))
    with pytest.raises(ValueError, match="at least one row"):
        normalize_interactions(pd.DataFrame(columns=["user_id", "item_id"]), ColumnsConfig())
    bad = pd.DataFrame({"user_id": [1], "item_id": [2], "value": [np.nan]})
    with pytest.raises(ValueError, match="finite"):
        normalize_interactions(bad, ColumnsConfig(value="value"))
    bad_timestamp = pd.DataFrame({"user_id": [1], "item_id": [2], "timestamp": ["not-a-timestamp"]})
    with pytest.raises(ValueError, match="timestamps"):
        normalize_interactions(bad_timestamp, ColumnsConfig(timestamp="timestamp"))


def test_splits_are_deterministic_and_temporal_requires_time(
    interaction_frame: pd.DataFrame,
) -> None:
    users = np.repeat(np.arange(5), 6)
    random_config = SplitConfig(validation_ratio=0.2, test_ratio=0.2, seed=4)
    assert np.array_equal(
        split_rows(interaction_frame, users, random_config).test,
        split_rows(interaction_frame, users, random_config).test,
    )
    temporal = SplitConfig(strategy="temporal", validation_ratio=0.2, test_ratio=0.2)
    result = split_rows(interaction_frame, users, temporal)
    assert result.test.tolist() == [5, 11, 17, 23, 29]
    with pytest.raises(ValueError, match="timestamp"):
        split_rows(interaction_frame.drop(columns="timestamp"), users, temporal)
    with pytest.raises(ValueError, match="unsupported"):
        split_rows(interaction_frame, users, SplitConfig(strategy="unknown"))
    with pytest.raises(ValueError, match="less than one"):
        split_rows(
            interaction_frame,
            users,
            SplitConfig(validation_ratio=0.6, test_ratio=0.5),
        )


def test_dataset_round_trip_and_corruption(tmp_path: Path, interaction_frame: pd.DataFrame) -> None:
    config = DatasetConfig(
        name="tabular",
        columns=ColumnsConfig(value="value", timestamp="timestamp"),
        split=SplitConfig(strategy="temporal", validation_ratio=0.2, test_ratio=0.2),
    )
    dataset = prepare_frame(interaction_frame, config)
    loaded = save_dataset(dataset, tmp_path / "prepared")
    again = save_dataset(dataset, tmp_path / "prepared")
    assert loaded.root == again.root
    assert loaded.dataset_digest == dataset.dataset_digest
    assert loaded.events.equals(dataset.events)
    assert loaded.train_events.equals(dataset.train_events)
    assert np.array_equal(loaded.test_rows, dataset.test_rows)
    assert loaded.train.shape == dataset.train.shape
    (loaded.root / "users.json").write_text("[]")
    with pytest.raises(DatasetIntegrityError, match="checksum"):
        load_dataset(loaded.root)
