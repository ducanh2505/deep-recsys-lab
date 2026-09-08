"""Isolated native LightGBM trainer used to avoid mixed runtime conflicts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np


def train(input_path: Path, output_path: Path, options: dict[str, Any]) -> None:
    with np.load(input_path, allow_pickle=False) as payload:
        features = payload["features"]
        labels = payload["labels"]
        groups = payload["groups"].astype(np.int64).tolist()
    rounds = int(options.pop("num_boost_round", 50))
    dataset = lgb.Dataset(features, label=labels, group=groups, free_raw_data=False)
    booster = lgb.train(options, dataset, num_boost_round=rounds)
    booster.save_model(str(output_path))


if __name__ == "__main__":  # pragma: no cover - exercised through the parent process
    train(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        cast(dict[str, Any], json.loads(sys.argv[3])),
    )
