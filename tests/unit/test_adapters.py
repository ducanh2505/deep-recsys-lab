from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys.integrations.datasets import read_movielens, read_yelp


def test_movielens_local_fixture_normalizes_columns(tmp_path: Path) -> None:
    source = tmp_path / "ratings.csv"
    source.write_text("userId,movieId,rating,timestamp\n1,10,4.5,1700000000\n")
    frame = read_movielens(tmp_path)
    assert frame.loc[0, "user_id"] == 1
    assert frame.loc[0, "item_id"] == 10
    assert frame.loc[0, "value"] == 4.5
    assert str(frame["timestamp"].dtype).startswith("datetime64")


def test_yelp_local_fixture_normalizes_columns(tmp_path: Path) -> None:
    source = tmp_path / "reviews.csv"
    pd.DataFrame([{"user_id": "u", "business_id": "b", "stars": 5, "date": "2024-01-01"}]).to_csv(
        source, index=False
    )
    frame = read_yelp(tmp_path)
    assert frame.loc[0, "item_id"] == "b"
    assert frame.loc[0, "value"] == 5
