import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deep_recsys_lifecycle.api import create_app
from deep_recsys_lifecycle.artifact import ArtifactStore, ServingArtifact
from deep_recsys_lifecycle.event_store import AppendOnlyEventStore
from deep_recsys_lifecycle.fixture import load_movielens_fixture
from deep_recsys_lifecycle.kafka import InMemoryKafkaBoundary
from deep_recsys_lifecycle.lifecycle import run_rolling_lifecycle


def test_rolling_lifecycle_has_exact_six_boundaries_and_only_100_is_active(
    tmp_path: Path,
) -> None:
    result = run_rolling_lifecycle(
        output_dir=tmp_path / "rolling",
        kafka=InMemoryKafkaBoundary(),
        source_events=load_movielens_fixture(),
    )

    assert [stage.percentage for stage in result.stages] == [50, 60, 70, 80, 90, 100]
    assert [stage.snapshot_event_count for stage in result.stages] == [20, 24, 28, 32, 36, 40]
    assert result.active_artifact_id == result.stages[-1].artifact_id
    assert ArtifactStore(result.artifact_store_path).load_active().artifact_id == (
        result.stages[-1].artifact_id
    )
    for stage in result.stages[:-1]:
        assert stage.active is False
        assert ServingArtifact.load(stage.artifact_path).artifact_id == stage.artifact_id
    assert result.stages[-1].active is True
    assert result.stages[-1].evaluation is None

    materialized = AppendOnlyEventStore(result.event_store_dir).materialize_snapshot()
    source_ids = {event.event_id for event in load_movielens_fixture()}
    assert {event.event_id for event in materialized.events} == source_ids
    assert len(AppendOnlyEventStore(result.event_store_dir).read_all()) == len(source_ids)


def test_rolling_does_not_publish_the_whole_source_at_stage_50(tmp_path: Path) -> None:
    class RecordingKafka(InMemoryKafkaBoundary):
        def __init__(self) -> None:
            super().__init__()
            self.published_counts: list[int] = []

        def publish(self, topic, events) -> None:
            self.published_counts.append(len(events))
            super().publish(topic, events)

    kafka = RecordingKafka()
    run_rolling_lifecycle(
        output_dir=tmp_path / "rolling",
        kafka=kafka,
        source_events=load_movielens_fixture(),
    )

    assert kafka.published_counts == [20, 4, 4, 4, 4, 4]


def test_rolling_evaluates_each_future_window_before_ingesting_it(tmp_path: Path) -> None:
    events: list[tuple[str, int]] = []
    observed_store_counts: dict[int, int] = {}
    output_dir = tmp_path / "rolling"

    def observe(phase: str, percentage: int) -> None:
        events.append((phase, percentage))
        if phase == "evaluate":
            observed_store_counts[percentage] = (
                AppendOnlyEventStore(output_dir / "event_store").materialize_snapshot().event_count
            )

    result = run_rolling_lifecycle(
        output_dir=output_dir,
        kafka=InMemoryKafkaBoundary(),
        source_events=load_movielens_fixture(),
        phase_observer=observe,
    )

    assert result.stages[-1].evaluation is None
    for percentage in (50, 60, 70, 80, 90):
        assert events.index(("evaluate", percentage)) < events.index(("ingest_next", percentage))
        assert observed_store_counts[percentage] == 40 * percentage // 100


def test_rolling_lhf_uses_only_prior_validation_and_100_has_no_future_metrics(
    tmp_path: Path,
) -> None:
    source_events = load_movielens_fixture()
    ordered = tuple(sorted(source_events, key=lambda event: (event.event_time, event.event_id)))
    result = run_rolling_lifecycle(
        output_dir=tmp_path / "rolling",
        kafka=InMemoryKafkaBoundary(),
        source_events=source_events,
    )

    for stage in result.stages[:-1]:
        start = len(ordered) * stage.percentage // 100
        end = len(ordered) * (stage.percentage + 10) // 100
        future_ids = {event.event_id for event in ordered[start:end]}
        manifest = json.loads((stage.artifact_path / "manifest.json").read_text(encoding="utf-8"))
        for mode in ("known_user", "history_only"):
            training = manifest["fusion_training"][mode]
            assert not future_ids.intersection(training["validation_event_ids"])
            if stage.percentage == 50:
                assert training["source_snapshot_fingerprint"] != stage.snapshot_fingerprint
            else:
                assert training["validation_sources"]
                assert all(
                    source["source_stage_percentage"] < stage.percentage
                    for source in training["validation_sources"]
                )

        evaluation = json.loads(
            (result.output_dir / "evaluations" / f"stage-{stage.percentage:03d}.json").read_text(
                encoding="utf-8"
            )
        )
        assert evaluation["evaluated_before_ingest"] is True
        assert set(evaluation["future_window"]["event_ids"]) == future_ids
        assert evaluation["validation_pools"]["known_user"]
        assert evaluation["validation_pools"]["history_only"]

    final_manifest = json.loads(
        (result.stages[-1].artifact_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert final_manifest["evaluation_metrics"] == {}
    assert final_manifest["evaluation_boundary"]["future_quality_claim"] is False
    report = result.report_path.read_text(encoding="utf-8")
    assert "Stage 100 has no Future Window" in report
    assert "Known-User" in report and "History-Only" in report


def test_interrupted_rolling_reuses_completed_artifact_bytes(tmp_path: Path) -> None:
    output_dir = tmp_path / "rolling"

    def fail_at_stage_70_evaluation(phase: str, percentage: int) -> None:
        if phase == "evaluate" and percentage == 70:
            raise RuntimeError("injected rolling failure")

    with pytest.raises(RuntimeError, match="injected rolling failure"):
        run_rolling_lifecycle(
            output_dir=output_dir,
            kafka=InMemoryKafkaBoundary(),
            source_events=load_movielens_fixture(),
            phase_observer=fail_at_stage_70_evaluation,
        )

    completed_bytes = {}
    for percentage in (50, 60):
        record = json.loads(
            (output_dir / "stages" / f"stage-{percentage:03d}.json").read_text(encoding="utf-8")
        )
        artifact_path = output_dir / record["artifact_id"]
        completed_bytes[percentage] = {
            name: (artifact_path / name).read_bytes() for name in ("manifest.json", "model.json")
        }

    observed: list[tuple[str, int]] = []
    result = run_rolling_lifecycle(
        output_dir=output_dir,
        kafka=InMemoryKafkaBoundary(),
        source_events=load_movielens_fixture(),
        phase_observer=lambda phase, percentage: observed.append((phase, percentage)),
    )

    for percentage in (50, 60):
        record = json.loads(
            (output_dir / "stages" / f"stage-{percentage:03d}.json").read_text(encoding="utf-8")
        )
        artifact_path = output_dir / record["artifact_id"]
        assert {
            name: (artifact_path / name).read_bytes() for name in ("manifest.json", "model.json")
        } == completed_bytes[percentage]
    assert ("evaluate", 50) not in observed
    assert ("evaluate", 60) not in observed
    assert result.active_artifact_id == result.stages[-1].artifact_id
    assert list((output_dir / ".rolling-quarantine").iterdir())


def test_corrupt_stage_is_not_reused_and_dataset_or_config_mismatch_fails(tmp_path: Path) -> None:
    output_dir = tmp_path / "rolling"
    source_events = load_movielens_fixture()
    first = run_rolling_lifecycle(
        output_dir=output_dir,
        kafka=InMemoryKafkaBoundary(),
        source_events=source_events,
    )
    corrupt_stage = first.stages[3]
    (corrupt_stage.artifact_path / "model.json").write_text("{}\n", encoding="utf-8")

    repaired = run_rolling_lifecycle(
        output_dir=output_dir,
        kafka=InMemoryKafkaBoundary(),
        source_events=source_events,
    )
    assert ServingArtifact.load(repaired.stages[3].artifact_path).artifact_id == (
        repaired.stages[3].artifact_id
    )
    with pytest.raises(ValueError, match="dataset checksum"):
        run_rolling_lifecycle(
            output_dir=output_dir,
            kafka=InMemoryKafkaBoundary(),
            source_events=source_events[:-1],
        )
    with pytest.raises(ValueError, match="configuration checksum"):
        run_rolling_lifecycle(
            output_dir=output_dir,
            kafka=InMemoryKafkaBoundary(),
            source_events=source_events,
            random_seed=7,
        )


def test_active_100_artifact_survives_api_restart(tmp_path: Path) -> None:
    result = run_rolling_lifecycle(
        output_dir=tmp_path / "rolling",
        kafka=InMemoryKafkaBoundary(),
        source_events=load_movielens_fixture(),
    )
    first = TestClient(create_app(result.active_pointer_path))
    restarted = TestClient(create_app(result.artifact_store_path))
    payload = {"subject_id": 1, "top_n": 5}
    assert (
        first.post("/recommendations", json=payload).json()
        == restarted.post("/recommendations", json=payload).json()
    )
