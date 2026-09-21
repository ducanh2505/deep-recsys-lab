import hashlib
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deep_recsys_lifecycle.api import create_app
from deep_recsys_lifecycle.artifact import ArtifactStore, ServingArtifact
from deep_recsys_lifecycle.kafka import InMemoryKafkaBoundary
from deep_recsys_lifecycle.lifecycle import run_fast_lifecycle


def _lifecycle(
    store_path: Path,
    revision: str = "revision-a",
    smoke_validator=None,
):
    return run_fast_lifecycle(
        store_path,
        kafka=InMemoryKafkaBoundary(),
        source_revision=revision,
        smoke_validator=smoke_validator,
    )


def test_missing_payload_is_rejected(tmp_path: Path) -> None:
    result = _lifecycle(tmp_path / "store")
    (result.artifact_path / "model.json").unlink()

    with pytest.raises(ValueError, match="payload is missing"):
        ServingArtifact.load(result.artifact_path)


def test_corrupted_payload_is_rejected(tmp_path: Path) -> None:
    result = _lifecycle(tmp_path / "store")
    (result.artifact_path / "model.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="payload checksum"):
        ServingArtifact.load(result.artifact_path)


def test_path_traversal_payload_is_rejected(tmp_path: Path) -> None:
    result = _lifecycle(tmp_path / "store")
    manifest_path = result.artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_inventory"][0]["path"] = "../model.json"
    _resign_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="payload"):
        ServingArtifact.load(result.artifact_path)


def test_invalid_active_pointer_fails_closed(tmp_path: Path) -> None:
    result = _lifecycle(tmp_path / "store")
    result.active_pointer_path.write_text(
        json.dumps({"artifact_id": "../outside"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="pointer"):
        ArtifactStore(result.artifact_store_path).load_active()


def test_failed_smoke_test_preserves_previous_active_artifact(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    first = _lifecycle(store_path, revision="revision-a")
    old_pointer = first.active_pointer_path.read_text(encoding="utf-8")

    def fail_smoke(_artifact: ServingArtifact) -> None:
        raise RuntimeError("injected smoke failure")

    with pytest.raises(RuntimeError, match="injected smoke failure"):
        _lifecycle(store_path, revision="revision-b", smoke_validator=fail_smoke)

    assert first.active_pointer_path.read_text(encoding="utf-8") == old_pointer
    assert ArtifactStore(store_path).load_active().artifact_id == first.artifact_id


def test_failed_activation_validation_preserves_previous_active_artifact(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    first = _lifecycle(store_path)
    invalid_path = store_path / "invalid-artifact"
    shutil.copytree(first.artifact_path, invalid_path)
    (invalid_path / "model.json").unlink()

    with pytest.raises(ValueError, match="payload is missing"):
        ArtifactStore(store_path).activate(invalid_path)

    assert ArtifactStore(store_path).load_active().artifact_id == first.artifact_id


def test_immutable_artifact_cannot_be_overwritten(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    first = _lifecycle(store_path)
    staging_path = ArtifactStore(store_path).new_staging_path()
    shutil.copytree(first.artifact_path, staging_path)

    with pytest.raises(FileExistsError, match="immutable artifact already exists"):
        ArtifactStore(store_path).publish(staging_path)

    assert ArtifactStore(store_path).load_active().artifact_id == first.artifact_id


def test_incompatible_configuration_and_negative_timing_are_rejected(tmp_path: Path) -> None:
    result = _lifecycle(tmp_path / "store")
    manifest_path = result.artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration"]["serving_device"] = "gpu"
    manifest["configuration_sha256"] = _configuration_checksum(manifest["configuration"])
    _resign_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible with serving device"):
        ServingArtifact.load(result.artifact_path)

    manifest["configuration"]["serving_device"] = "cpu"
    manifest["configuration_sha256"] = _configuration_checksum(manifest["configuration"])
    manifest["timings"]["smoke"] = -1
    _resign_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be negative"):
        ServingArtifact.load(result.artifact_path)


def test_api_restart_from_same_active_pointer_is_deterministic(tmp_path: Path) -> None:
    result = _lifecycle(tmp_path / "store")
    requests = (
        {"subject_id": 1, "top_n": 5},
        {"history": [10, 12], "top_n": 5},
        {"top_n": 5},
    )

    first_client = TestClient(create_app(result.active_pointer_path))
    first_responses = [
        first_client.post("/recommendations", json=payload).json() for payload in requests
    ]
    first_health = first_client.get("/health").json()

    restarted_client = TestClient(create_app(result.artifact_store_path))
    restarted_responses = [
        restarted_client.post("/recommendations", json=payload).json() for payload in requests
    ]
    restarted_health = restarted_client.get("/health").json()

    assert restarted_responses == first_responses
    assert restarted_health == first_health
    assert first_health["artifact_id"] == result.artifact_id
    assert first_health["status"] == "ok"


def test_runtime_loader_does_not_fit_or_read_event_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _lifecycle(tmp_path / "store")

    import deep_recsys_lifecycle.event_store as event_store
    import deep_recsys_lifecycle.popularity as popularity

    monkeypatch.setattr(popularity, "fit_popularity", lambda *_args, **_kwargs: _fail())
    monkeypatch.setattr(event_store.AppendOnlyEventStore, "read_all", lambda _self: _fail())

    loaded = ServingArtifact.load(result.artifact_path)

    assert loaded.artifact_id == result.artifact_id


def _fail() -> object:
    raise AssertionError("training or Event Store code was called by the runtime loader")


def _resign_manifest(manifest: dict[str, object]) -> None:
    unsigned_bundle = {
        key: value
        for key, value in manifest.items()
        if key not in {"bundle_integrity_sha256", "manifest_sha256"}
    }
    manifest["bundle_integrity_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned_bundle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _configuration_checksum(configuration: object) -> str:
    return hashlib.sha256(
        json.dumps(
            configuration,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
