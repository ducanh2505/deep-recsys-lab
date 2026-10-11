"""Read and audit existing splits; materialize only individual user batches."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


@dataclass
class Split:
    items: np.ndarray
    offsets: np.ndarray
    keys: np.ndarray

    def coordinates(self, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return row/item indices for the selected users, including empty rows."""
        lengths = self.offsets[users + 1] - self.offsets[users]
        rows = np.repeat(np.arange(len(users), dtype=np.int64), lengths)
        starts = np.cumsum(lengths) - lengths
        positions = (np.repeat(self.offsets[users], lengths)
                     + np.arange(len(rows)) - np.repeat(starts, lengths))
        return rows, self.items[positions].astype(np.int64)


@dataclass
class Dataset:
    user_ids: np.ndarray
    item_ids: np.ndarray
    train: Split
    valid: Split
    test: Split
    manifest: dict
    audit: dict

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def batch(self, users: np.ndarray, include_valid: bool = False) -> np.ndarray:
        """Build binary train histories, optionally adding validation at inference."""
        history = np.zeros((len(users), self.n_items), dtype=np.float32)
        for split in ((self.train, self.valid) if include_valid else (self.train,)):
            rows, columns = split.coordinates(users)
            history[rows, columns] = 1.0
        return history


def canonical_sha256(frame: pl.DataFrame) -> str:
    """Hash canonical rows using the exact convention in data preparation docs."""
    digest = hashlib.sha256()
    for user, movie, rating, timestamp in frame.sort(["userId", "movieId"]).iter_rows():
        digest.update(f"{user},{movie},{rating:.1f},{timestamp}\n".encode())
    return digest.hexdigest()


def sample_ranking_pairs(
    data: Dataset, users: np.ndarray, seed: int, epoch: int, negative_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Preselect one train positive and uniform unseen negatives per user/epoch.

    Negatives are sampled with replacement using train membership only. Separate
    RNG streams keep pairs identical across objective weights and KL schedules.
    """
    if negative_count < 1:
        raise ValueError("negative_count must be positive")
    lengths = np.diff(data.train.offsets)[users]
    if np.any(lengths == 0) or np.any(lengths >= data.n_items):
        raise ValueError("Ranking sampling requires positives and unseen items for every user")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 4107]))
    positions = data.train.offsets[users] + np.floor(rng.random(len(users)) * lengths).astype(np.int64)
    positives = data.train.items[positions].astype(np.int64)
    negatives = rng.integers(data.n_items, size=(len(users), negative_count), dtype=np.int64)
    # Sorted train keys provide membership checks without a dense user/item matrix.
    while True:
        keys = users[:, None].astype(np.int64) * data.n_items + negatives
        positions = np.searchsorted(data.train.keys, keys)
        invalid = ((positions < len(data.train.keys)) &
                   (data.train.keys[np.minimum(positions, len(data.train.keys) - 1)] == keys))
        if not invalid.any():
            return positives, negatives
        negatives[invalid] = rng.integers(data.n_items, size=int(invalid.sum()))


def _split(frame: pl.DataFrame, user_ids: np.ndarray, item_ids: np.ndarray) -> Split:
    raw_users = frame["userId"].to_numpy()
    raw_items = frame["movieId"].to_numpy()
    users = np.searchsorted(user_ids, raw_users)
    items = np.searchsorted(item_ids, raw_items)
    if np.any(users >= len(user_ids)) or np.any(user_ids[users] != raw_users):
        raise ValueError("Split contains a user absent from train")
    if np.any(items >= len(item_ids)) or np.any(item_ids[items] != raw_items):
        raise ValueError("Split contains an item absent from train")
    keys = users.astype(np.int64) * len(item_ids) + items
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    if np.any(np.diff(keys) == 0):
        raise ValueError("Split contains duplicate user/item pairs")
    counts = np.bincount(users, minlength=len(user_ids))
    return Split(items[order].astype(np.int32), np.r_[0, counts.cumsum()], keys)


def _overlap(left: np.ndarray, right: np.ndarray) -> bool:
    positions = np.searchsorted(left, right)
    inside = positions < len(left)
    return bool(np.any(left[positions[inside]] == right[inside]))


def load_dataset(directory: Path) -> Dataset:
    """Verify split content against the manifest without modifying prepared data."""
    manifest = json.loads((directory / "manifest.json").read_text())
    splits = {}
    hashes = {}
    user_ids = item_ids = None
    columns = ["userId", "movieId", "rating", "timestamp"]
    expected_schema = {"userId": pl.Int64, "movieId": pl.Int64,
                       "rating": pl.Float64, "timestamp": pl.Int64}
    for name in ("train", "valid", "test"):
        frame = pl.read_parquet(directory / f"{name}.parquet")
        if dict(frame.schema) != expected_schema:
            raise ValueError(f"Unexpected schema in {name}.parquet")
        frame = frame.select(columns)
        if any(frame.null_count().row(0)) or frame.filter(pl.col("rating") < 4).height:
            raise ValueError(f"Nulls or non-positive ratings in {name}.parquet")
        if frame.height != manifest["final_split_counts"][name]:
            raise ValueError(f"Row count does not match manifest for {name}")
        hashes[name] = canonical_sha256(frame)
        if hashes[name] != manifest["canonical_sha256"][name]:
            raise ValueError(f"Canonical SHA-256 does not match manifest for {name}")
        if name == "train":
            if frame.height == 0:
                raise ValueError("Train split is empty")
            user_ids = np.unique(frame["userId"].to_numpy())
            item_ids = np.unique(frame["movieId"].to_numpy())
        splits[name] = _split(frame, user_ids, item_ids)
    if len(user_ids) != manifest["ten_core"]["users"] or len(item_ids) != manifest["ten_core"]["items"]:
        raise ValueError("Train user/item counts do not match manifest")
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if _overlap(splits[left].keys, splits[right].keys):
            raise ValueError(f"Overlapping interactions in {left} and {right}")
    audit = {"canonical_sha256": hashes, "split_counts": {
        name: len(split.items) for name, split in splits.items()},
        "eligible_users": {name: int(np.count_nonzero(np.diff(split.offsets)))
                           for name, split in splits.items()},
        "n_users": len(user_ids), "n_items": len(item_ids),
        "schema_verified": True, "disjoint_splits_verified": True}
    return Dataset(user_ids, item_ids, splits["train"], splits["valid"],
                   splits["test"], manifest, audit)
