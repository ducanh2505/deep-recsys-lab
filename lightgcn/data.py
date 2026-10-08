"""Load processed MovieLens splits and build the selected training graph."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch


@dataclass
class Split:
    users: np.ndarray
    items: np.ndarray


@dataclass
class Dataset:
    user_ids: np.ndarray
    item_ids: np.ndarray
    train: Split
    valid: Split
    test: Split
    train_keys: np.ndarray
    train_bits: np.ndarray
    train_offsets: np.ndarray
    valid_offsets: np.ndarray
    test_offsets: np.ndarray
    train_split: str = "train"

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def _offsets(users: np.ndarray, n_users: int) -> np.ndarray:
    counts = np.bincount(users, minlength=n_users)
    return np.r_[0, counts.cumsum()]


def _mapped_split(path: Path, user_ids: np.ndarray, item_ids: np.ndarray) -> Split:
    frame = pl.read_parquet(path, columns=["userId", "movieId"])
    raw_users = frame["userId"].to_numpy()
    raw_items = frame["movieId"].to_numpy()
    users = np.searchsorted(user_ids, raw_users)
    items = np.searchsorted(item_ids, raw_items)
    if np.any(users == len(user_ids)) or np.any(user_ids[np.minimum(users, len(user_ids) - 1)] != raw_users):
        raise ValueError(f"Unknown user in {path}")
    if np.any(items == len(item_ids)) or np.any(item_ids[np.minimum(items, len(item_ids) - 1)] != raw_items):
        raise ValueError(f"Unknown item in {path}")
    order = np.lexsort((items, users))
    return Split(users[order].astype(np.int32), items[order].astype(np.int32))


def load_dataset(directory: Path, train_split: str = "train") -> Dataset:
    """Load disjoint splits, optionally merging validation into the training graph."""
    if train_split not in {"train", "train-valid"}:
        raise ValueError("train_split must be train or train-valid")
    train_frame = pl.read_parquet(directory / "train.parquet", columns=["userId", "movieId"])
    user_ids = np.unique(train_frame["userId"].to_numpy())
    item_ids = np.unique(train_frame["movieId"].to_numpy())
    train = _mapped_split(directory / "train.parquet", user_ids, item_ids)
    valid = _mapped_split(directory / "valid.parquet", user_ids, item_ids)
    test = _mapped_split(directory / "test.parquet", user_ids, item_ids)
    splits = {"train": train, "valid": valid, "test": test}
    keys = {name: split.users.astype(np.int64) * len(item_ids) + split.items
            for name, split in splits.items()}
    for name, values in keys.items():
        if np.any(np.diff(values) <= 0):
            raise ValueError(f"Duplicate {name} interactions")
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        positions = np.searchsorted(keys[left], keys[right])
        in_bounds = positions < len(keys[left])
        if np.any(keys[left][positions[in_bounds]] == keys[right][in_bounds]):
            raise ValueError(f"Overlapping {left} and {right} interactions")
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = manifest.get("final_split_counts", {})
        for name, split in splits.items():
            if name in expected and len(split.users) != expected[name]:
                raise ValueError(f"{name} interaction count does not match manifest")
    if train_split == "train-valid":
        users = np.r_[train.users, valid.users]
        items = np.r_[train.items, valid.items]
        order = np.lexsort((items, users))
        train = Split(users[order], items[order])
    train_keys = train.users.astype(np.int64) * len(item_ids) + train.items
    train_offsets = _offsets(train.users, len(user_ids))
    if np.any(np.diff(train_offsets) >= len(item_ids)):
        raise ValueError("At least one user has no unobserved item to sample")
    train_bits = np.zeros((len(user_ids), (len(item_ids) + 7) // 8), dtype=np.uint8)
    np.bitwise_or.at(
        train_bits,
        (train.users, train.items // 8),
        np.left_shift(np.uint8(1), (train.items % 8).astype(np.uint8)),
    )
    return Dataset(
        user_ids, item_ids, train, valid, test, train_keys, train_bits,
        train_offsets,
        _offsets(valid.users, len(user_ids)),
        _offsets(test.users, len(user_ids)),
        train_split,
    )


def normalized_adjacency(data: Dataset, device: torch.device) -> torch.Tensor:
    """D^-1/2 A D^-1/2 using the selected training split, without self-loops."""
    users = data.train.users.astype(np.int64)
    items = data.train.items.astype(np.int64) + data.n_users
    node_count = data.n_users + data.n_items
    degrees = np.bincount(np.r_[users, items], minlength=node_count)
    if np.any(degrees == 0):
        raise ValueError("Train graph contains isolated nodes")
    weights = (degrees[users] * degrees[items]).astype(np.float64) ** -0.5
    indices = np.stack((np.r_[users, items], np.r_[items, users]))
    values = np.r_[weights, weights].astype(np.float32)
    return torch.sparse_coo_tensor(
        torch.from_numpy(indices).to(device),
        torch.from_numpy(values).to(device),
        (node_count, node_count),
        check_invariants=False,
    ).coalesce()


def sample_negatives(
    users: np.ndarray, train_bits: np.ndarray, n_items: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniformly sample unobserved items, rejecting every train interaction."""
    negatives = rng.integers(n_items, size=len(users), dtype=np.int32)
    while True:
        invalid = ((train_bits[users, negatives // 8] >> (negatives % 8)) & 1) != 0
        if not np.any(invalid):
            return negatives
        negatives[invalid] = rng.integers(n_items, size=int(invalid.sum()), dtype=np.int32)
