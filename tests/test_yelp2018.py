from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deep_recsys_lab.lightgcn.yelp2018 import (
    prepare_yelp2018_data,
    read_official_yelp2018_split,
)


def _write_official_fixture(root: Path) -> None:
    (root / "user_list.txt").write_text(
        "org_id remap_id\n" + "\n".join(f"u{user} {user}" for user in range(4)) + "\n",
        encoding="utf-8",
    )
    (root / "item_list.txt").write_text(
        "org_id remap_id\n" + "\n".join(f"i{item} {item}" for item in range(5)) + "\n",
        encoding="utf-8",
    )
    (root / "train.txt").write_text(
        "0 0 1 2\n1 1 2 3\n2 2 3 4\n3 0 4\n",
        encoding="utf-8",
    )
    (root / "test.txt").write_text(
        "0 3\n1 4\n2 0\n3 1\n",
        encoding="utf-8",
    )


def test_read_official_yelp2018_split_parses_mapping_headers(tmp_path: Path) -> None:
    _write_official_fixture(tmp_path)
    train, test = read_official_yelp2018_split(tmp_path)
    assert train.shape == (4, 5)
    assert train.nnz == 11
    assert test.nnz == 4
    assert train.multiply(test).nnz == 0


def test_prepare_yelp2018_derives_validation_without_catalog_loss(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    source.mkdir()
    _write_official_fixture(source)
    prepared = prepare_yelp2018_data(
        source,
        output,
        seed=2020,
        expected_counts={"users": 4, "items": 5, "positive_edges": 15},
        source={"kind": "synthetic-official-format"},
    )
    assert prepared.manifest["dataset"] == "yelp2018"
    assert prepared.manifest["counts"]["official_train_edges"] == 11
    assert prepared.manifest["counts"]["official_test_edges"] == 4
    assert prepared.manifest["counts"]["positive_edges"] == 15
    assert prepared.train.multiply(prepared.validation).nnz == 0
    assert prepared.train.multiply(prepared.test).nnz == 0
    assert prepared.validation.multiply(prepared.test).nnz == 0
    assert np.all(prepared.train.getnnz(axis=0) > 0)
    assert prepared.train.nnz + prepared.validation.nnz == 11
    reloaded_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert reloaded_manifest["dataset_hash"] == prepared.dataset_hash
