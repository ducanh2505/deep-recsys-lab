"""Portable prepared-dataset persistence with checksum verification."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.core.hashing import canonical_json, digest_file, digest_token
from recsys.core.io import read_json, write_json

from .mapping import decode_identifier, encode_identifier
from .types import PreparedDataset

PAYLOADS = (
    "events.parquet",
    "catalog.parquet",
    "users.json",
    "items.json",
    "train.npz",
    "validation.npz",
    "test.npz",
    "split_rows.npz",
)


class DatasetIntegrityError(ValueError):
    pass


def _encode_identifiers(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    encoded = frame.copy()
    for column in columns:
        if column in encoded:
            encoded[column] = [
                canonical_json(encode_identifier(value)) for value in encoded[column]
            ]
    return encoded


def _decode_identifiers(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    decoded = frame.copy()
    for column in columns:
        if column in decoded:
            decoded[column] = [
                decode_identifier(cast(dict[str, Any], json.loads(value)))
                for value in decoded[column]
            ]
    return decoded


def save_dataset(dataset: PreparedDataset, parent: Path) -> PreparedDataset:
    target = parent / digest_token(dataset.dataset_digest)
    if target.exists():
        return load_dataset(target)
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dataset-", dir=parent))
    try:
        _encode_identifiers(dataset.events, ("user_id", "item_id")).to_parquet(
            temporary / "events.parquet", index=False
        )
        _encode_identifiers(dataset.item_metadata, ("item_id",)).to_parquet(
            temporary / "catalog.parquet", index=False
        )
        write_json(
            temporary / "users.json", [encode_identifier(value) for value in dataset.user_ids]
        )
        write_json(
            temporary / "items.json", [encode_identifier(value) for value in dataset.item_ids]
        )
        sp.save_npz(temporary / "train.npz", dataset.train, compressed=True)
        sp.save_npz(temporary / "validation.npz", dataset.validation, compressed=True)
        sp.save_npz(temporary / "test.npz", dataset.test, compressed=True)
        np.savez_compressed(
            temporary / "split_rows.npz",
            train=dataset.train_rows.astype(np.int64, copy=False),
            validation=dataset.validation_rows.astype(np.int64, copy=False),
            test=dataset.test_rows.astype(np.int64, copy=False),
        )
        manifest = {
            **dataset.manifest,
            "payloads": {name: digest_file(temporary / name) for name in PAYLOADS},
        }
        write_json(temporary / "manifest.json", manifest)
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_dataset(target)


def load_dataset(root: Path, *, verify: bool = True) -> PreparedDataset:
    try:
        manifest = cast(dict[str, Any], read_json(root / "manifest.json"))
    except (FileNotFoundError, ValueError) as exc:
        raise DatasetIntegrityError("prepared dataset manifest is missing or invalid") from exc
    if manifest.get("schema_version") != 1:
        raise DatasetIntegrityError("unsupported prepared dataset schema")
    try:
        if root.name != digest_token(str(manifest.get("dataset_digest", ""))):
            raise DatasetIntegrityError("prepared dataset directory does not match its digest")
    except ValueError as exc:
        raise DatasetIntegrityError("prepared dataset digest is invalid") from exc
    if verify:
        try:
            payloads = manifest.get("payloads")
            valid_payloads = isinstance(payloads, dict) and all(
                payloads.get(name) == digest_file(root / name) for name in PAYLOADS
            )
        except OSError:
            valid_payloads = False
        if not valid_payloads:
            raise DatasetIntegrityError("prepared dataset checksum verification failed")
    try:
        users = tuple(decode_identifier(value) for value in read_json(root / "users.json"))
        items = tuple(decode_identifier(value) for value in read_json(root / "items.json"))
        with np.load(root / "split_rows.npz", allow_pickle=False) as split_rows:
            train_rows = np.asarray(split_rows["train"], dtype=np.int64).copy()
            validation_rows = np.asarray(split_rows["validation"], dtype=np.int64).copy()
            test_rows = np.asarray(split_rows["test"], dtype=np.int64).copy()
        result = PreparedDataset(
            events=_decode_identifiers(
                pd.read_parquet(root / "events.parquet"), ("user_id", "item_id")
            ),
            user_ids=users,
            item_ids=items,
            train=sp.load_npz(root / "train.npz").tocsr(),
            validation=sp.load_npz(root / "validation.npz").tocsr(),
            test=sp.load_npz(root / "test.npz").tocsr(),
            train_rows=train_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            dataset_digest=str(manifest["dataset_digest"]),
            manifest=manifest,
            item_metadata=_decode_identifiers(
                pd.read_parquet(root / "catalog.parquet"), ("item_id",)
            ),
            root=root,
        )
    except Exception as exc:
        raise DatasetIntegrityError("prepared dataset payload is invalid") from exc
    if result.shape != tuple(manifest.get("shape", [])) or any(
        matrix.shape != result.shape for matrix in (result.validation, result.test)
    ):
        raise DatasetIntegrityError("prepared dataset shape does not match its manifest")
    rows = (result.train_rows, result.validation_rows, result.test_rows)
    if any(value.ndim != 1 for value in rows):
        raise DatasetIntegrityError("prepared dataset split rows must be one-dimensional")
    combined = np.concatenate(rows)
    if len(combined) != len(result.events) or not np.array_equal(
        np.sort(combined), np.arange(len(result.events), dtype=np.int64)
    ):
        raise DatasetIntegrityError("prepared dataset split rows are incomplete or overlapping")
    return result
