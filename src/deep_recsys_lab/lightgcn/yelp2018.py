"""Download and prepare the official processed Yelp2018 LightGCN split."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import scipy.sparse as sp

from .data import LightGCNData, prepare_lightgcn_from_splits, sha256_file

OFFICIAL_REPOSITORY = "https://github.com/gusye1234/LightGCN-PyTorch"
OFFICIAL_RAW_ROOT = "https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch"
OFFICIAL_DATASET_PATH = "data/yelp2018"
OFFICIAL_FILES = ("user_list.txt", "item_list.txt", "train.txt", "test.txt")
DEFAULT_REVISION = "master"
DEFAULT_VALIDATION_RATIO = 0.1
EXPECTED_COUNTS = {
    "users": 31_668,
    "items": 38_048,
    "positive_edges": 1_561_406,
}


def _official_url(filename: str, revision: str) -> str:
    return f"{OFFICIAL_RAW_ROOT}/{revision}/{OFFICIAL_DATASET_PATH}/{filename}"


def _download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "deep-recsys-lab/0.1"})
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def download_yelp2018(
    source_dir: Path,
    *,
    revision: str = DEFAULT_REVISION,
    force: bool = False,
) -> dict[str, Any]:
    """Download the four official processed Yelp2018 files and return provenance."""

    source_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for filename in OFFICIAL_FILES:
        destination = source_dir / filename
        url = _official_url(filename, revision)
        if force or not destination.exists():
            _download_file(url, destination)
        files[filename] = {
            "url": url,
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
        }
    return {
        "kind": "official LightGCN-PyTorch processed Yelp2018",
        "repository": OFFICIAL_REPOSITORY,
        "revision": revision,
        "dataset_path": OFFICIAL_DATASET_PATH,
        "files": files,
    }


def _read_mapping(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    remapped: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                remap_id = int(fields[1])
            except ValueError:
                if line_number == 1:
                    continue
                raise ValueError(f"invalid remapped ID in {path}:{line_number}") from None
            original_id = fields[0]
            if original_id in mapping or remap_id in remapped:
                raise ValueError(f"duplicate ID in mapping file {path}:{line_number}")
            if remap_id < 0:
                raise ValueError(f"negative remapped ID in {path}:{line_number}")
            mapping[original_id] = remap_id
            remapped.add(remap_id)
    if not mapping:
        raise ValueError(f"mapping file is empty: {path}")
    expected = set(range(len(mapping)))
    if remapped != expected:
        raise ValueError(f"remapped IDs in {path} are not contiguous from zero")
    # The official interaction files use remapped integer IDs, while the
    # mapping files retain original IDs in their first column.  Accept both
    # representations so the importer remains compatible with released files
    # and small fixtures that use symbolic original IDs.
    for remap_id in sorted(remapped):
        alias = str(remap_id)
        if alias in mapping and mapping[alias] != remap_id:
            raise ValueError(f"ambiguous original/remapped ID {alias!r} in {path}")
        mapping[alias] = remap_id
    return mapping


def _read_interactions(
    path: Path,
    user_mapping: dict[str, int],
    item_mapping: dict[str, int],
    *,
    n_users: int,
    n_items: int,
) -> sp.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            user_token = fields[0]
            if user_token not in user_mapping:
                raise ValueError(f"unknown user {user_token!r} in {path}:{line_number}")
            user_id = user_mapping[user_token]
            for item_token in fields[1:]:
                if item_token not in item_mapping:
                    raise ValueError(f"unknown item {item_token!r} in {path}:{line_number}")
                rows.append(user_id)
                columns.append(item_mapping[item_token])
    matrix = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    raw_edges = len(rows)
    matrix.sum_duplicates()
    if matrix.nnz != raw_edges:
        raise ValueError(f"duplicate interactions found in {path}")
    matrix.sort_indices()
    return matrix


def read_official_yelp2018_split(source_dir: Path) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Read official ``train.txt`` and ``test.txt`` into binary CSR matrices."""

    required = [source_dir / filename for filename in OFFICIAL_FILES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Yelp2018 source files: {', '.join(missing)}")
    user_mapping = _read_mapping(source_dir / "user_list.txt")
    item_mapping = _read_mapping(source_dir / "item_list.txt")
    shape = (
        max(user_mapping.values()) + 1,
        max(item_mapping.values()) + 1,
    )
    train = _read_interactions(
        source_dir / "train.txt",
        user_mapping,
        item_mapping,
        n_users=shape[0],
        n_items=shape[1],
    )
    test = _read_interactions(
        source_dir / "test.txt",
        user_mapping,
        item_mapping,
        n_users=shape[0],
        n_items=shape[1],
    )
    if train.multiply(test).nnz:
        raise ValueError("official Yelp2018 train/test split contains overlapping interactions")
    if np.any(train.getnnz(axis=0) == 0):
        raise ValueError("official Yelp2018 train split has catalog items absent from train")
    if np.any(test.getnnz(axis=0) > 0) and np.any(
        (train.getnnz(axis=0) == 0) & (test.getnnz(axis=0) > 0)
    ):
        raise ValueError("official Yelp2018 test contains cold-start catalog items")
    if np.any(train.getnnz(axis=1) == 0):
        raise ValueError("official Yelp2018 train split contains users without interactions")
    return train, test


def _derive_validation(
    official_train: sp.csr_matrix,
    *,
    seed: int,
    validation_ratio: float,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Derive validation only from official train while preserving train coverage."""

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")
    train = official_train.tocsr().astype(np.float32)
    train.sort_indices()
    item_remaining = np.asarray(train.getnnz(axis=0), dtype=np.int64).ravel()
    validation_rows: list[int] = []
    validation_columns: list[int] = []
    for user_id in range(train.shape[0]):
        start, stop = int(train.indptr[user_id]), int(train.indptr[user_id + 1])
        columns = train.indices[start:stop]
        n_edges = stop - start
        if n_edges <= 1:
            continue
        desired = max(1, int(math.floor(n_edges * validation_ratio)))
        desired = min(desired, n_edges - 1)
        rng = np.random.default_rng(seed + user_id * 1_000_003)
        shuffled = rng.permutation(columns)
        eligible = shuffled[item_remaining[shuffled] > 1]
        chosen = eligible[: min(desired, len(eligible))]
        if len(chosen):
            validation_rows.extend([user_id] * len(chosen))
            validation_columns.extend(int(item) for item in chosen)
            item_remaining[chosen] -= 1
    validation = sp.csr_matrix(
        (
            np.ones(len(validation_rows), dtype=np.float32),
            (validation_rows, validation_columns),
        ),
        shape=train.shape,
        dtype=np.float32,
    )
    validation.sum_duplicates()
    validation.sort_indices()
    derived_train = (train - validation).tocsr().astype(np.float32)
    derived_train.data[derived_train.data != 0] = 1.0
    derived_train.eliminate_zeros()
    derived_train.sort_indices()
    if np.any(derived_train.getnnz(axis=0) == 0):
        raise RuntimeError("validation derivation removed a catalog item from training")
    if derived_train.multiply(validation).nnz:
        raise RuntimeError("validation derivation produced leakage")
    return derived_train, validation


def _local_source_metadata(source_dir: Path, revision: str) -> dict[str, Any]:
    return {
        "kind": "official LightGCN-PyTorch processed Yelp2018",
        "repository": OFFICIAL_REPOSITORY,
        "revision": revision,
        "dataset_path": OFFICIAL_DATASET_PATH,
        "files": {
            filename: {
                "sha256": sha256_file(source_dir / filename),
                "bytes": (source_dir / filename).stat().st_size,
            }
            for filename in OFFICIAL_FILES
        },
    }


def prepare_yelp2018_data(
    source_dir: Path,
    output_dir: Path,
    *,
    seed: int = 2020,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    revision: str = DEFAULT_REVISION,
    source: dict[str, Any] | None = None,
    expected_counts: dict[str, int] | None = None,
) -> LightGCNData:
    """Prepare official Yelp2018 train/validation/test artifacts with checksums."""

    official_train, official_test = read_official_yelp2018_split(source_dir)
    train, validation = _derive_validation(
        official_train,
        seed=seed,
        validation_ratio=validation_ratio,
    )
    counts_extra = {
        "official_train_edges": int(official_train.nnz),
        "official_test_edges": int(official_test.nnz),
        "positive_edges": int(official_train.nnz + official_test.nnz),
        "all_positive_items": int(
            np.count_nonzero(np.asarray((official_train + official_test).getnnz(axis=0)))
        ),
    }
    expected = dict(EXPECTED_COUNTS if expected_counts is None else expected_counts)
    source_info = source or _local_source_metadata(source_dir, revision)
    return prepare_lightgcn_from_splits(
        train,
        validation,
        official_test,
        np.arange(official_train.shape[0], dtype=np.int64),
        np.arange(official_train.shape[1], dtype=np.int64),
        output_dir,
        dataset="yelp2018",
        protocol="official-train-test-within-train-validation",
        seed=seed,
        config={
            "source_revision": revision,
            "validation_ratio": validation_ratio,
            "validation_seed": seed,
            "official_train_is_untouched_for_test": True,
        },
        source=source_info,
        counts_extra=counts_extra,
        expected_counts=expected,
    )
