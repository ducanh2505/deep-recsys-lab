"""Deterministic synthetic interactions for examples and integration tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def generate_synthetic(parameters: dict[str, Any], *, seed: int) -> pd.DataFrame:
    users = int(parameters.get("users", 32))
    items = int(parameters.get("items", 64))
    per_user = int(parameters.get("interactions_per_user", 12))
    if users < 1 or items < 3 or not 3 <= per_user <= items:
        raise ValueError("synthetic data requires users>=1, items>=3, and 3<=interactions<=items")
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    base = pd.Timestamp("2020-01-01", tz="UTC")
    for user in range(users):
        preference = (np.arange(per_user) * 5 + user * 7) % items
        selected = preference[rng.permutation(per_user)]
        for offset, item in enumerate(selected.tolist()):
            records.append(
                {
                    "user_id": f"user-{user:04d}",
                    "item_id": f"item-{int(item):05d}",
                    "value": 1.0,
                    "timestamp": base + pd.Timedelta(user * per_user + offset, unit="D"),
                    "category": f"group-{int(item) % 5}",
                }
            )
    return pd.DataFrame.from_records(records)
