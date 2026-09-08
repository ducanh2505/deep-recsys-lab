"""Normalize ratings from a local MovieLens export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_movielens(path: Path) -> pd.DataFrame:
    source = path
    if path.is_dir():
        candidates = [path / "ratings.csv", path / "ratings.dat"]
        source = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if source.suffix == ".dat":
        frame = pd.read_csv(
            source,
            sep="::",
            engine="python",
            names=["user_id", "item_id", "value", "timestamp"],
        )
    else:
        frame = pd.read_csv(source).rename(
            columns={"userId": "user_id", "movieId": "item_id", "rating": "value"}
        )
    required = {"user_id", "item_id"}
    if not required.issubset(frame.columns):
        raise ValueError("MovieLens ratings file must contain user and item identifiers")
    if "value" not in frame:
        frame["value"] = 1.0
    if "timestamp" in frame:
        raw_timestamps = frame["timestamp"]
        numeric = pd.to_numeric(raw_timestamps, errors="coerce")
        parsed = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
        if (raw_timestamps.notna() & parsed.isna()).any():
            raise ValueError("MovieLens timestamps must be Unix seconds")
        frame["timestamp"] = parsed
    return frame
