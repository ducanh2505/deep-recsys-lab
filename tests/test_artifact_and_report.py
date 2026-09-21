import json
from pathlib import Path

import pytest

from deep_recsys_lifecycle.artifact import ServingArtifact
from deep_recsys_lifecycle.kafka import InMemoryKafkaBoundary
from deep_recsys_lifecycle.lifecycle import run_fast_lifecycle


def test_artifact_manifest_is_minimal_and_artifact_cannot_be_overwritten(
    tmp_path: Path,
) -> None:
    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())
    artifact = ServingArtifact.load(result.artifact_path)

    assert {
        "artifact_version",
        "data_snapshot_fingerprint",
        "event_count",
        "positive_interaction_count",
        "implicit_signal_threshold",
        "retriever",
        "query_modes",
        "model_sha256",
    } <= artifact.manifest.keys()
    assert artifact.manifest["query_modes"] == [
        "known_user",
        "history_only",
        "empty_history",
    ]
    with pytest.raises(FileExistsError):
        artifact.save()


def test_artifact_rejects_tampered_manifest_metadata(tmp_path: Path) -> None:
    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())
    manifest_path = result.artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["event_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest checksum"):
        ServingArtifact.load(result.artifact_path)


def test_report_is_a_static_html_document_with_embedded_lifecycle_data(tmp_path: Path) -> None:
    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())
    report = result.report_path.read_text(encoding="utf-8")

    assert report.startswith("<!doctype html>")
    assert '<html lang="en">' in report
    assert "</html>" in report
    assert 'id="data-snapshot"' in report
    assert 'id="positive-interactions"' in report
    assert 'id="query-modes"' in report
    assert "query_examples" in report

    snapshot = json.loads(result.snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["fingerprint"] == result.snapshot_fingerprint
