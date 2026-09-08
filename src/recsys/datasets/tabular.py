"""CSV and Parquet adapter for the generic interaction contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recsys.conf.schema import ColumnsConfig

from .mapping import validate_identifier


def read_table(path: Path, format_name: str | None = None) -> pd.DataFrame:
    selected = (format_name or path.suffix.lstrip(".")).lower()
    if selected == "csv":
        return pd.read_csv(path)
    if selected in {"parquet", "pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table format: {selected}")


def normalize_interactions(frame: pd.DataFrame, columns: ColumnsConfig) -> pd.DataFrame:
    required = {columns.user_id, columns.item_id}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"interaction table is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("interaction table must contain at least one row")
    mappings = [(columns.user_id, "user_id"), (columns.item_id, "item_id")]
    if columns.value is not None and columns.value in frame:
        mappings.append((columns.value, "value"))
    if columns.timestamp is not None and columns.timestamp in frame:
        mappings.append((columns.timestamp, "timestamp"))
    sources = {source for source, _ in mappings}
    if len(sources) != len(mappings):
        raise ValueError("column mapping contains overlapping source columns")
    if any(
        source != target and target in frame and target not in sources
        for source, target in mappings
    ):
        raise ValueError("column mapping would create duplicate canonical columns")
    rename = dict(mappings)

    normalized = frame.rename(columns=rename).copy()
    normalized["user_id"] = [
        validate_identifier(value, field="user_id") for value in normalized["user_id"]
    ]
    normalized["item_id"] = [
        validate_identifier(value, field="item_id") for value in normalized["item_id"]
    ]
    if "value" not in normalized:
        normalized["value"] = 1.0
    normalized["value"] = pd.to_numeric(normalized["value"], errors="raise").astype(float)
    if not np.isfinite(normalized["value"].to_numpy()).all():
        raise ValueError("interaction values must be finite")
    if "timestamp" in normalized:
        raw_timestamps = normalized["timestamp"]
        parsed_timestamps = pd.to_datetime(raw_timestamps, utc=True, errors="coerce")
        if (raw_timestamps.notna() & parsed_timestamps.isna()).any():
            raise ValueError("interaction timestamps must be valid ISO-8601 or datetime values")
        normalized["timestamp"] = parsed_timestamps
    return normalized.reset_index(drop=True)


def read_and_normalize(path: Path, format_name: str, columns: ColumnsConfig) -> pd.DataFrame:
    return normalize_interactions(read_table(path, format_name), columns)


def extract_item_metadata(events: pd.DataFrame) -> pd.DataFrame:
    excluded = {"user_id", "value", "timestamp"}
    metadata_columns = [column for column in events.columns if column not in excluded]
    if metadata_columns == ["item_id"]:
        return pd.DataFrame({"item_id": events["item_id"].drop_duplicates()})
    grouped = events.loc[:, metadata_columns].drop_duplicates(subset=["item_id"], keep="last")
    return grouped.reset_index(drop=True)
