"""Dataset preparation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.conf.schema import DatasetConfig
from recsys.core.hashing import digest_value
from recsys.core.paths import WorkspacePaths

from .mapping import deterministic_mapping
from .splitting import SplitRows, split_rows
from .store import save_dataset
from .synthetic import generate_synthetic
from .tabular import extract_item_metadata, normalize_interactions, read_table
from .types import PreparedDataset


def _matrix(
    events: pd.DataFrame,
    rows: np.ndarray,
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    shape: tuple[int, int],
) -> sp.csr_matrix:
    selected = events.iloc[rows]
    matrix = sp.csr_matrix(
        (
            selected["value"].to_numpy(dtype=np.float32, copy=False),
            (user_indices[rows], item_indices[rows]),
        ),
        shape=shape,
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    return matrix


def prepare_frame(
    events: pd.DataFrame,
    config: DatasetConfig,
    *,
    paths: WorkspacePaths | None = None,
) -> PreparedDataset:
    normalized = normalize_interactions(events, config.columns)
    user_ids = deterministic_mapping(normalized["user_id"], field="user_id")
    item_ids = deterministic_mapping(normalized["item_id"], field="item_id")
    user_lookup = {value: index for index, value in enumerate(user_ids)}
    item_lookup = {value: index for index, value in enumerate(item_ids)}
    user_indices = normalized["user_id"].map(user_lookup).to_numpy(dtype=np.int64)
    item_indices = normalized["item_id"].map(item_lookup).to_numpy(dtype=np.int64)
    splits: SplitRows = split_rows(normalized, user_indices, config.split)
    shape = (len(user_ids), len(item_ids))
    train = _matrix(normalized, splits.train, user_indices, item_indices, shape)
    validation = _matrix(normalized, splits.validation, user_indices, item_indices, shape)
    test = _matrix(normalized, splits.test, user_indices, item_indices, shape)

    digest_basis: dict[str, Any] = {
        "schema": {"required": ["user_id", "item_id"], "optional": ["value", "timestamp"]},
        "config": config,
        "users": user_ids,
        "items": item_ids,
        "events": [
            {
                "user_id": row.user_id,
                "item_id": row.item_id,
                "value": float(row.value),
                "timestamp": None
                if not hasattr(row, "timestamp") or pd.isna(row.timestamp)
                else row.timestamp.isoformat(),
            }
            for row in normalized.itertuples(index=False)
        ],
        "split_rows": {
            "train": splits.train.tolist(),
            "validation": splits.validation.tolist(),
            "test": splits.test.tolist(),
        },
    }
    dataset_digest = digest_value(digest_basis)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_digest": dataset_digest,
        "adapter": config.name,
        "shape": list(shape),
        "event_count": len(normalized),
        "split": config.split,
    }
    prepared = PreparedDataset(
        events=normalized,
        user_ids=user_ids,
        item_ids=item_ids,
        train=train,
        validation=validation,
        test=test,
        train_rows=splits.train,
        validation_rows=splits.validation,
        test_rows=splits.test,
        dataset_digest=dataset_digest,
        manifest=manifest,
        item_metadata=extract_item_metadata(normalized),
    )
    return save_dataset(prepared, paths.prepared_data) if paths is not None else prepared


def prepare_from_config(config: DatasetConfig, paths: WorkspacePaths) -> PreparedDataset:
    if config.name == "synthetic":
        frame = generate_synthetic(config.parameters, seed=config.split.seed)
    elif config.name == "tabular":
        if config.path is None:
            raise ValueError("dataset.path is required for the tabular adapter")
        configured = Path(config.path).expanduser()
        source = configured if configured.is_absolute() else paths.root / configured
        frame = read_table(source, config.format)
    elif config.name in {"movielens", "yelp"}:
        if config.path is None:
            raise ValueError(f"dataset.path is required for the {config.name} adapter")
        from recsys.integrations.datasets import read_movielens, read_yelp

        source = (paths.root / config.path).resolve()
        frame = read_movielens(source) if config.name == "movielens" else read_yelp(source)
        config = DatasetConfig(
            name=config.name,
            path=config.path,
            format=config.format,
            split=config.split,
            parameters=config.parameters,
        )
    else:
        raise ValueError(f"unknown dataset adapter: {config.name}")
    return prepare_frame(frame, config, paths=paths)
