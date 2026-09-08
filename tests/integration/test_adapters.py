from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recsys.conf.schema import ColumnsConfig, DatasetConfig, SplitConfig
from recsys.core.paths import WorkspacePaths
from recsys.datasets import prepare_from_config


@pytest.mark.integration
@pytest.mark.parametrize("format_name", ["csv", "parquet"])
def test_generic_table_adapter_end_to_end(tmp_path: Path, format_name: str) -> None:
    frame = pd.DataFrame(
        [
            {
                "account": f"u-{user}",
                "product": f"i-{(user + offset) % 8}",
                "strength": 1.0,
                "occurred": f"2024-01-{user * 5 + offset + 1:02d}",
            }
            for user in range(3)
            for offset in range(5)
        ]
    )
    source = tmp_path / f"events.{format_name}"
    if format_name == "csv":
        frame.to_csv(source, index=False)
    else:
        frame.to_parquet(source, index=False)
    config = DatasetConfig(
        name="tabular",
        path=source.name,
        format=format_name,
        columns=ColumnsConfig(
            user_id="account",
            item_id="product",
            value="strength",
            timestamp="occurred",
        ),
        split=SplitConfig(strategy="temporal", validation_ratio=0.2, test_ratio=0.2),
    )
    prepared = prepare_from_config(config, WorkspacePaths.from_value(tmp_path).ensure())
    assert prepared.shape == (3, 7)
    assert prepared.validation.nnz == prepared.test.nnz == 3


@pytest.mark.integration
@pytest.mark.parametrize("adapter", ["movielens", "yelp"])
def test_optional_local_adapter_prepares_fixture(tmp_path: Path, adapter: str) -> None:
    records = [
        {
            "userId": user,
            "movieId": (user + offset) % 7,
            "rating": 1,
            "timestamp": 1_700_000_000 + user * 10 + offset,
        }
        if adapter == "movielens"
        else {
            "user_id": f"u-{user}",
            "business_id": f"b-{(user + offset) % 7}",
            "stars": 1,
            "date": f"2024-01-{user * 5 + offset + 1:02d}",
        }
        for user in range(3)
        for offset in range(5)
    ]
    source = tmp_path / "ratings.csv" if adapter == "movielens" else tmp_path / "reviews.csv"
    pd.DataFrame(records).to_csv(source, index=False)
    config = DatasetConfig(
        name=adapter,
        path=str(tmp_path),
        split=SplitConfig(strategy="temporal", validation_ratio=0.2, test_ratio=0.2),
    )
    prepared = prepare_from_config(config, WorkspacePaths.from_value(tmp_path).ensure())
    assert prepared.shape == (3, 7)
    assert prepared.manifest["adapter"] == adapter
