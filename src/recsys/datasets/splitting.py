"""Explicit, deterministic interaction split strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recsys.conf.schema import SplitConfig


@dataclass(frozen=True, slots=True)
class SplitRows:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _holdout_counts(size: int, validation_ratio: float, test_ratio: float) -> tuple[int, int]:
    if size < 3:
        return 0, 0
    validation = max(1, int(round(size * validation_ratio))) if validation_ratio > 0 else 0
    test = max(1, int(round(size * test_ratio))) if test_ratio > 0 else 0
    while validation + test >= size:
        if validation >= test and validation > 0:
            validation -= 1
        elif test > 0:
            test -= 1
        else:
            break
    return validation, test


def split_rows(events: pd.DataFrame, user_indices: np.ndarray, config: SplitConfig) -> SplitRows:
    if not 0 <= config.validation_ratio < 1 or not 0 <= config.test_ratio < 1:
        raise ValueError("split ratios must be in [0, 1)")
    if config.validation_ratio + config.test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be less than one")
    if config.strategy not in {"user_holdout", "temporal"}:
        raise ValueError(f"unsupported split strategy: {config.strategy}")
    if config.strategy == "temporal" and (
        "timestamp" not in events or events["timestamp"].isna().any()
    ):
        raise ValueError("temporal split requires a timestamp for every interaction")

    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for user_index in sorted(np.unique(user_indices).tolist()):
        rows = np.flatnonzero(user_indices == user_index)
        validation_count, test_count = _holdout_counts(
            len(rows), config.validation_ratio, config.test_ratio
        )
        if config.strategy == "temporal":
            timestamps = events.iloc[rows]["timestamp"].astype("int64").to_numpy()
            order = np.lexsort((rows, timestamps))
            ordered = rows[order]
        else:
            rng = np.random.default_rng(config.seed + int(user_index) * 1_000_003)
            ordered = rng.permutation(rows)
        test_rows = ordered[-test_count:] if test_count else np.empty(0, dtype=np.int64)
        validation_stop = len(ordered) - test_count
        validation_start = validation_stop - validation_count
        validation_rows = ordered[validation_start:validation_stop]
        train_rows = ordered[:validation_start]
        train.extend(int(value) for value in train_rows)
        validation.extend(int(value) for value in validation_rows)
        test.extend(int(value) for value in test_rows)
    return SplitRows(
        train=np.asarray(sorted(train), dtype=np.int64),
        validation=np.asarray(sorted(validation), dtype=np.int64),
        test=np.asarray(sorted(test), dtype=np.int64),
    )
