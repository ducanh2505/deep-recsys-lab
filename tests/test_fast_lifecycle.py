from pathlib import Path

from deep_recsys_lifecycle.artifact import ServingArtifact
from deep_recsys_lifecycle.kafka import InMemoryKafkaBoundary
from deep_recsys_lifecycle.lifecycle import run_fast_lifecycle


def test_fast_lifecycle_produces_loadable_artifact_and_static_report(
    tmp_path: Path,
) -> None:
    result = run_fast_lifecycle(
        output_dir=tmp_path / "fast",
        kafka=InMemoryKafkaBoundary(),
    )

    artifact = ServingArtifact.load(result.artifact_path)

    assert artifact.manifest["retriever"] == "popularity"
    assert artifact.manifest["event_count"] > 0
    assert result.report_path.is_file()
    assert "Movie Recommender Lifecycle Showcase" in result.report_path.read_text()
    assert result.snapshot_fingerprint == result.replay_snapshot_fingerprint
