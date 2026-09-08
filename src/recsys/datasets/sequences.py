"""Canonical ordering for training-event sequences."""

from __future__ import annotations

import pandas as pd

from .types import PreparedDataset


def ordered_train_events(data: PreparedDataset) -> pd.DataFrame:
    """Return training rows grouped by user and ordered deterministically."""

    events = data.train_events.copy()
    events["recsys_user_index"] = events["user_id"].map(data.user_index)
    events["recsys_item_index"] = events["item_id"].map(data.item_index)
    events["recsys_event_order"] = data.train_rows
    has_complete_time = "timestamp" in events and not events["timestamp"].isna().any()
    columns = ["recsys_user_index"]
    if has_complete_time:
        columns.append("timestamp")
    columns.append("recsys_event_order")
    return events.sort_values(columns, kind="mergesort").reset_index(drop=True)


def ordered_train_item_indices(data: PreparedDataset) -> tuple[tuple[int, ...], ...]:
    """Return one ordered internal-item sequence for every mapped user."""

    sequences: list[list[int]] = [[] for _ in data.user_ids]
    events = ordered_train_events(data)
    for user_index, item_index in events[["recsys_user_index", "recsys_item_index"]].itertuples(
        index=False, name=None
    ):
        sequences[int(user_index)].append(int(item_index))
    return tuple(tuple(values) for values in sequences)
