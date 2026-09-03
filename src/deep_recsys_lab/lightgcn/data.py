"""Deterministic implicit-feedback preparation for LightGCN-style experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import LightGCNDataConfig

SCHEMA_VERSION = 1
MATRIX_FILES = ("train.npz", "validation.npz", "test.npz")
ID_FILES = ("user_ids.npy", "item_ids.npy")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    """Hash JSON data with canonical formatting."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _csr_hash(matrix: sp.csr_matrix) -> str:
    return stable_hash(
        {
            "shape": matrix.shape,
            "indptr": _array_hash(matrix.indptr),
            "indices": _array_hash(matrix.indices),
            "data": _array_hash(matrix.data),
        }
    )


def _dataset_hash_basis(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key not in {"dataset_hash", "files"}}


@dataclass(frozen=True)
class LightGCNData:
    """Verified prepared matrices and their provenance manifest."""

    root: Path
    train: sp.csr_matrix
    validation: sp.csr_matrix
    test: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    manifest: dict[str, Any]

    @property
    def n_users(self) -> int:
        return int(self.train.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.train.shape[1])

    @property
    def dataset_hash(self) -> str:
        return str(self.manifest["dataset_hash"])


def _find_ratings(raw_dir: Path) -> Path:
    candidates = (raw_dir / "ratings.csv", raw_dir / "ml-20m" / "ratings.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"ratings.csv not found below {raw_dir}")


def _to_csr(
    user_values: np.ndarray,
    item_values: np.ndarray,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    shape: tuple[int, int],
) -> sp.csr_matrix:
    rows = np.searchsorted(user_ids, user_values).astype(np.int64, copy=False)
    cols = np.searchsorted(item_ids, item_values).astype(np.int64, copy=False)
    matrix = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=shape, dtype=np.float32
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _assert_expected(counts: dict[str, int], expected: dict[str, int]) -> None:
    for key, value in expected.items():
        if key in counts and counts[key] != value:
            raise ValueError(f"prepared count mismatch for {key}: {counts[key]} != {value}")


def prepare_lightgcn_from_splits(
    train: sp.csr_matrix,
    validation: sp.csr_matrix,
    test: sp.csr_matrix,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    output_dir: Path,
    *,
    dataset: str,
    protocol: str,
    seed: int,
    config: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    counts_extra: dict[str, int] | None = None,
    expected_counts: dict[str, int] | None = None,
) -> LightGCNData:
    """Persist explicit binary train/validation/test matrices as a verified artifact.

    The helper is used by datasets that already provide an official train/test
    split, such as Yelp2018.  ``validation`` is kept separate so model
    selection never needs to inspect the official test fold.
    """

    matrices = {
        "train": train.tocsr().astype(np.float32),
        "validation": validation.tocsr().astype(np.float32),
        "test": test.tocsr().astype(np.float32),
    }
    if not (matrices["train"].shape == matrices["validation"].shape == matrices["test"].shape):
        raise ValueError("train/validation/test matrices must have identical shapes")
    if matrices["train"].shape != (len(user_ids), len(item_ids)):
        raise ValueError("ID arrays do not match split matrix shape")
    for name, matrix in matrices.items():
        matrix.sum_duplicates()
        matrix.sort_indices()
        if matrix.nnz and not np.all(np.isin(matrix.data, [0.0, 1.0])):
            raise ValueError(f"{name} matrix must contain binary interactions")
        matrix.data.fill(1.0)
    if _overlap(matrices["train"], matrices["validation"]):
        raise ValueError("train/validation leakage detected")
    if _overlap(matrices["train"], matrices["test"]):
        raise ValueError("train/test leakage detected")
    if _overlap(matrices["validation"], matrices["test"]):
        raise ValueError("validation/test leakage detected")
    user_ids = np.asarray(user_ids)
    item_ids = np.asarray(item_ids)
    if len(np.unique(user_ids)) != len(user_ids) or len(np.unique(item_ids)) != len(item_ids):
        raise ValueError("user and item IDs must be unique")
    if np.any(np.diff(matrices["train"].tocsc().indptr) == 0):
        raise ValueError("catalog contains an item absent from training")

    counts_manifest = {
        "users": int(matrices["train"].shape[0]),
        "items": int(matrices["train"].shape[1]),
        "train_edges": int(matrices["train"].nnz),
        "validation_edges": int(matrices["validation"].nnz),
        "test_edges": int(matrices["test"].nnz),
        "validation_users": int(np.count_nonzero(np.diff(matrices["validation"].indptr))),
        "test_users": int(np.count_nonzero(np.diff(matrices["test"].indptr))),
    }
    if counts_extra:
        counts_manifest.update({str(key): int(value) for key, value in counts_extra.items()})
    _assert_expected(counts_manifest, expected_counts or {})

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        sp.save_npz(output_dir / f"{name}.npz", matrix, compressed=True)
    np.save(output_dir / ID_FILES[0], user_ids, allow_pickle=False)
    np.save(output_dir / ID_FILES[1], item_ids, allow_pickle=False)
    file_hashes = {name: sha256_file(output_dir / name) for name in (*MATRIX_FILES, *ID_FILES)}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "protocol": protocol,
        "seed": int(seed),
        "config": config or {},
        "counts": counts_manifest,
        "source": source or {"kind": "in-memory-test-fixture"},
        "files": file_hashes,
        "content_hashes": {
            "train": _csr_hash(matrices["train"]),
            "validation": _csr_hash(matrices["validation"]),
            "test": _csr_hash(matrices["test"]),
            "user_ids": _array_hash(user_ids),
            "item_ids": _array_hash(item_ids),
        },
    }
    manifest["dataset_hash"] = stable_hash(_dataset_hash_basis(manifest))
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return load_lightgcn_data(output_dir)


def prepare_lightgcn_from_frame(
    ratings: pd.DataFrame,
    output_dir: Path,
    *,
    config: LightGCNDataConfig,
    seed: int,
    source: dict[str, Any] | None = None,
) -> LightGCNData:
    """Create per-user random 80/10/10 implicit splits from rating rows."""

    required = {"userId", "movieId", "rating"}
    if not required.issubset(ratings.columns):
        raise ValueError(f"ratings must contain {sorted(required)}")
    positives = ratings.loc[
        ratings["rating"] >= config.positive_rating_threshold, ["userId", "movieId"]
    ].drop_duplicates(["userId", "movieId"])
    counts = positives.groupby("userId", sort=False).size()
    eligible = counts[counts >= config.min_positive_ratings].index
    positives = positives[positives["userId"].isin(eligible)].sort_values(
        ["userId", "movieId"], kind="mergesort"
    )
    if positives.empty:
        raise ValueError("no eligible positive interactions")

    user_values = positives["userId"].to_numpy(dtype=np.int64, copy=False)
    starts = np.flatnonzero(np.concatenate((np.array([True]), user_values[1:] != user_values[:-1])))
    stops = np.concatenate((starts[1:], np.array([len(positives)])))
    folds = np.zeros(len(positives), dtype=np.uint8)
    for start, stop in zip(starts, stops, strict=True):
        user_id = int(user_values[start])
        n_edges = int(stop - start)
        n_validation = max(1, int(np.floor(n_edges * config.validation_ratio)))
        n_test = max(1, int(np.floor(n_edges * config.test_ratio)))
        if n_validation + n_test >= n_edges:
            raise ValueError(f"user {user_id} cannot retain a training edge")
        rng = np.random.default_rng(seed + user_id * 1_000_003)
        order = rng.permutation(n_edges)
        folds[start + order[:n_test]] = 2
        folds[start + order[n_test : n_test + n_validation]] = 1

    train_frame = positives.iloc[np.flatnonzero(folds == 0)]
    validation_before = positives.iloc[np.flatnonzero(folds == 1)]
    test_before = positives.iloc[np.flatnonzero(folds == 2)]
    catalog = np.sort(train_frame["movieId"].unique().astype(np.int64))
    validation_frame = validation_before[validation_before["movieId"].isin(catalog)]
    test_frame = test_before[test_before["movieId"].isin(catalog)]
    users = np.sort(positives["userId"].unique().astype(np.int64))
    shape = (len(users), len(catalog))
    train = _to_csr(
        train_frame["userId"].to_numpy(),
        train_frame["movieId"].to_numpy(),
        users,
        catalog,
        shape,
    )
    validation = _to_csr(
        validation_frame["userId"].to_numpy(),
        validation_frame["movieId"].to_numpy(),
        users,
        catalog,
        shape,
    )
    test = _to_csr(
        test_frame["userId"].to_numpy(),
        test_frame["movieId"].to_numpy(),
        users,
        catalog,
        shape,
    )
    counts_manifest = {
        "users": len(users),
        "all_positive_items": int(positives["movieId"].nunique()),
        "items": len(catalog),
        "positive_edges": len(positives),
        "train_edges": int(train.nnz),
        "validation_edges_before_cold_start_filter": len(validation_before),
        "validation_edges": int(validation.nnz),
        "test_edges_before_cold_start_filter": len(test_before),
        "test_edges": int(test.nnz),
        "cold_start_items_removed": int(positives["movieId"].nunique() - len(catalog)),
        "validation_cold_start_edges_removed": int(len(validation_before) - validation.nnz),
        "test_cold_start_edges_removed": int(len(test_before) - test.nnz),
        "validation_users": int(np.count_nonzero(np.diff(validation.indptr))),
        "test_users": int(np.count_nonzero(np.diff(test.indptr))),
    }
    _assert_expected(counts_manifest, config.expected_counts)

    output_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(output_dir / MATRIX_FILES[0], train, compressed=True)
    sp.save_npz(output_dir / MATRIX_FILES[1], validation, compressed=True)
    sp.save_npz(output_dir / MATRIX_FILES[2], test, compressed=True)
    np.save(output_dir / ID_FILES[0], users, allow_pickle=False)
    np.save(output_dir / ID_FILES[1], catalog, allow_pickle=False)
    file_hashes = {name: sha256_file(output_dir / name) for name in (*MATRIX_FILES, *ID_FILES)}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": config.name,
        "protocol": "per-user-random-80-10-10",
        "seed": seed,
        "config": {
            "positive_rating_threshold": config.positive_rating_threshold,
            "min_positive_ratings": config.min_positive_ratings,
            "validation_ratio": config.validation_ratio,
            "test_ratio": config.test_ratio,
        },
        "counts": counts_manifest,
        "source": source or {"kind": "in-memory-test-fixture"},
        "files": file_hashes,
        "content_hashes": {
            "train": _csr_hash(train),
            "validation": _csr_hash(validation),
            "test": _csr_hash(test),
            "user_ids": _array_hash(users),
            "item_ids": _array_hash(catalog),
        },
    }
    manifest["dataset_hash"] = stable_hash(_dataset_hash_basis(manifest))
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return load_lightgcn_data(output_dir)


def prepare_lightgcn_data(
    raw_dir: Path, output_dir: Path, *, config: LightGCNDataConfig, seed: int
) -> LightGCNData:
    """Read GroupLens ratings and create a separate LightGCN artifact."""

    ratings_path = _find_ratings(raw_dir)
    ratings = pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int64, "movieId": np.int64, "rating": np.float32},
    )
    source = {
        "kind": "GroupLens MovieLens 20M",
        "ratings_file": ratings_path.name,
        "ratings_sha256": sha256_file(ratings_path),
    }
    return prepare_lightgcn_from_frame(ratings, output_dir, config=config, seed=seed, source=source)


def _overlap(left: sp.csr_matrix, right: sp.csr_matrix) -> int:
    return int(left.multiply(right).nnz)


def load_lightgcn_data(root: Path, *, verify: bool = True) -> LightGCNData:
    """Load and fail closed when any prepared artifact or split invariant changed."""

    manifest_path = root / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported LightGCN data schema")
    if verify:
        expected_dataset_hash = manifest.get("dataset_hash")
        if stable_hash(_dataset_hash_basis(manifest)) != expected_dataset_hash:
            raise ValueError("LightGCN manifest hash mismatch")
        for name, expected in dict(manifest["files"]).items():
            if sha256_file(root / name) != expected:
                raise ValueError(f"checksum mismatch: {name}")
    train = sp.load_npz(root / MATRIX_FILES[0]).tocsr().astype(np.float32)
    validation = sp.load_npz(root / MATRIX_FILES[1]).tocsr().astype(np.float32)
    test = sp.load_npz(root / MATRIX_FILES[2]).tocsr().astype(np.float32)
    users = np.load(root / ID_FILES[0], allow_pickle=False)
    items = np.load(root / ID_FILES[1], allow_pickle=False)
    if verify:
        actual_content = {
            "train": _csr_hash(train),
            "validation": _csr_hash(validation),
            "test": _csr_hash(test),
            "user_ids": _array_hash(users),
            "item_ids": _array_hash(items),
        }
        if actual_content != manifest.get("content_hashes"):
            raise ValueError("prepared LightGCN semantic content hash mismatch")
    if train.shape != validation.shape or train.shape != test.shape:
        raise ValueError("split matrix shapes do not match")
    if train.shape != (len(users), len(items)):
        raise ValueError("ID arrays do not match matrix shape")
    if _overlap(train, validation) or _overlap(train, test) or _overlap(validation, test):
        raise ValueError("train/validation/test leakage detected")
    if np.any(np.diff(train.tocsc().indptr) == 0):
        raise ValueError("catalog contains an item absent from training")
    counts = dict(manifest["counts"])
    if (
        int(counts["users"]) != train.shape[0]
        or int(counts["items"]) != train.shape[1]
        or int(counts["train_edges"]) != train.nnz
        or int(counts["validation_edges"]) != validation.nnz
        or int(counts["test_edges"]) != test.nnz
    ):
        raise ValueError("prepared LightGCN counts do not match matrix contents")
    return LightGCNData(root, train, validation, test, users, items, manifest)
