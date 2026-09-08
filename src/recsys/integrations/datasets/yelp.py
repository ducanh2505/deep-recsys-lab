"""Normalize interactions from a local Yelp export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_yelp(path: Path) -> pd.DataFrame:
    source = path
    if path.is_dir():
        candidates = [path / "reviews.parquet", path / "reviews.csv", path / "review.json"]
        source = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if source.suffix == ".parquet":
        frame = pd.read_parquet(source)
    elif source.suffix == ".json":
        frame = pd.read_json(source, lines=True)
    else:
        frame = pd.read_csv(source)
    frame = frame.rename(columns={"business_id": "item_id", "stars": "value", "date": "timestamp"})
    if not {"user_id", "item_id"}.issubset(frame.columns):
        raise ValueError("Yelp data must contain user_id and business_id/item_id")
    if "value" not in frame:
        frame["value"] = 1.0
    if "timestamp" in frame:
        raw_timestamps = frame["timestamp"]
        parsed = pd.to_datetime(raw_timestamps, utc=True, errors="coerce")
        if (raw_timestamps.notna() & parsed.isna()).any():
            raise ValueError("Yelp timestamps must be valid datetime values")
        frame["timestamp"] = parsed
    return frame
