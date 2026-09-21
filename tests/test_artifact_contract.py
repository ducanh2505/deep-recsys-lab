import json
from pathlib import Path

import pytest

from deep_recsys_lifecycle.artifact import (
    canonical_configuration_checksum,
    deterministic_dataset_checksum,
)
from deep_recsys_lifecycle.fixture import load_movielens_fixture
from deep_recsys_lifecycle.kafka import InMemoryKafkaBoundary
from deep_recsys_lifecycle.lifecycle import run_fast_lifecycle


def test_dataset_checksum_is_independent_of_event_order() -> None:
    events = load_movielens_fixture()

    assert deterministic_dataset_checksum(events) == deterministic_dataset_checksum(
        tuple(reversed(events))
    )


def test_configuration_checksum_is_canonical() -> None:
    assert canonical_configuration_checksum({"seed": 42, "retriever": "popularity"}) == (
        canonical_configuration_checksum({"retriever": "popularity", "seed": 42})
    )


def test_lifecycle_manifest_is_complete_and_pointer_is_relative(tmp_path: Path) -> None:
    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())
    manifest = json.loads((result.artifact_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["artifact_schema_version"] == 1
    assert manifest["artifact_id"] == result.artifact_id
    assert manifest["data_cutoff_percentage"] == 50
    assert manifest["source_event_count"] == result.source_event_count
    assert manifest["dataset_checksum"]
    assert manifest["configuration_sha256"]
    assert manifest["random_seed"] == 42
    assert manifest["training_device"] == "cpu"
    assert manifest["serving_device"] == "cpu"
    assert set(manifest["timings"]) >= {"training", "export", "smoke"}
    assert set(manifest["evaluation_metrics"]) >= {"popularity", "itemknn", "rrf"}
    assert manifest["payload_inventory"] == [
        {
            "path": "model.json",
            "role": "popularity_retriever",
            "sha256": manifest["model_sha256"],
        }
    ]

    pointer = json.loads(result.active_pointer_path.read_text(encoding="utf-8"))
    assert pointer == {"artifact_id": result.artifact_id}
    assert not Path(pointer["artifact_id"]).is_absolute()


def test_manifest_json_is_stable_and_rejects_unsupported_version(tmp_path: Path) -> None:
    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())
    manifest_path = result.artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported artifact schema version"):
        # The checksum failure must not mask the typed version error.
        from deep_recsys_lifecycle.artifact import ServingArtifact

        ServingArtifact.load(result.artifact_path)
