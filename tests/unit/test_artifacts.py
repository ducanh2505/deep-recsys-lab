from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from recsys.artifacts import ArtifactIntegrityError, RuntimeSpec, load_artifact, write_artifact
from recsys.artifacts.npz import read_npz, write_npz
from recsys.artifacts.store import arrays_csr, csr_arrays
from recsys.core.hashing import digest_file
from recsys.core.types import QueryMode
from recsys.datasets.types import PreparedDataset


def test_deterministic_npz_and_sparse_round_trip(tmp_path: Path, prepared: PreparedDataset) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    arrays = {"b": np.arange(3), "a": np.eye(2, dtype=np.float32)}
    write_npz(first, arrays)
    write_npz(second, dict(reversed(list(arrays.items()))))
    assert digest_file(first) == digest_file(second)
    assert np.array_equal(read_npz(first)["a"], arrays["a"])
    encoded = csr_arrays("value", prepared.train)
    assert (arrays_csr(encoded, "value") != prepared.train).nnz == 0
    with pytest.raises(ArtifactIntegrityError, match="missing sparse"):
        arrays_csr({}, "missing")


def test_artifact_identity_is_content_derived_and_immutable(
    tmp_path: Path, prepared: PreparedDataset
) -> None:
    spec = RuntimeSpec(
        "popularity",
        "popularity",
        tuple(QueryMode),
        {"global_scores": prepared.train.sum(axis=0).A1.astype(np.float32)},
    )
    first = write_artifact(tmp_path, spec, prepared, {"alpha": 1})
    second = write_artifact(tmp_path, spec, prepared, {"alpha": 1})
    different = write_artifact(tmp_path, spec, prepared, {"alpha": 2})
    metadata_change = write_artifact(
        tmp_path,
        RuntimeSpec(
            "popularity",
            "popularity",
            tuple(QueryMode),
            spec.arrays,
            metadata={"interpretation": "different"},
        ),
        prepared,
        {"alpha": 1},
    )
    assert first.root == second.root
    assert first.manifest.artifact_id != different.manifest.artifact_id
    assert first.manifest.artifact_id != metadata_change.manifest.artifact_id
    assert first.root.name in first.manifest.artifact_id
    assert "latest" not in [path.name for path in (tmp_path / "popularity").iterdir()]


def test_artifact_fails_closed_for_checksum_schema_and_path(
    tmp_path: Path, prepared: PreparedDataset
) -> None:
    artifact = write_artifact(
        tmp_path,
        RuntimeSpec(
            "popularity",
            "popularity",
            (QueryMode.HISTORY,),
            {"global_scores": np.ones(prepared.shape[1], dtype=np.float32)},
        ),
        prepared,
        {},
    )
    (artifact.root / "catalog.json").write_text("{}")
    with pytest.raises(ArtifactIntegrityError, match="missing|checksum"):
        load_artifact(artifact.root)

    manifest_path = artifact.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactIntegrityError, match="manifest"):
        load_artifact(artifact.root, verify=False)


def test_artifact_rejects_unsafe_extra_payload(tmp_path: Path, prepared: PreparedDataset) -> None:
    extra = tmp_path / "extra.txt"
    extra.write_text("safe")
    with pytest.raises(ValueError, match="payload name"):
        write_artifact(
            tmp_path / "models",
            RuntimeSpec(
                "test",
                "popularity",
                (QueryMode.HISTORY,),
                {"global_scores": np.ones(prepared.shape[1])},
                files={"../escape": extra},
            ),
            prepared,
            {},
        )
