import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from deep_recsys_lifecycle.api import create_app
from deep_recsys_lifecycle.artifact import ServingArtifact
from deep_recsys_lifecycle.fixture import load_movielens_fixture
from deep_recsys_lifecycle.kafka import InMemoryKafkaBoundary
from deep_recsys_lifecycle.lifecycle import run_fast_lifecycle
from deep_recsys_lifecycle.models import RatingEvent
from deep_recsys_lifecycle.temporal import split_temporal_events


def test_lhf_and_itemknn_are_exported_with_query_mode_banks_and_reloadable_api(
    tmp_path: Path,
) -> None:
    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())
    artifact = ServingArtifact.load(result.artifact_path)
    manifest = json.loads((result.artifact_path / "manifest.json").read_text(encoding="utf-8"))

    assert artifact.itemknn is not None
    assert artifact.known_user_fusion is not None
    assert artifact.history_only_fusion is not None
    assert artifact.known_user_fusion.retriever_bank == (
        "popularity",
        "itemknn",
        "multivae",
        "lightgcn",
    )
    assert artifact.history_only_fusion.retriever_bank == (
        "popularity",
        "itemknn",
        "multivae",
    )
    assert "lightgcn" not in artifact.history_only_fusion.feature_names
    assert manifest["configuration"]["fusion_enabled"] is True
    assert set(manifest["fusion_training"]) == {"known_user", "history_only"}

    client = TestClient(create_app(result.artifact_path))
    known = client.post("/recommendations", json={"subject_id": 1, "top_n": 5})
    history = client.post("/recommendations", json={"history": [10, 12], "top_n": 5})
    empty = client.post("/recommendations", json={"top_n": 5})

    assert known.status_code == history.status_code == empty.status_code == 200
    assert known.json()["retriever"] == "lhf"
    assert history.json()["retriever"] == "lhf"
    assert empty.json()["retriever"] == "popularity"
    assert {candidate["movie_id"] for candidate in known.json()["candidates"]}.isdisjoint({10, 11})
    assert {candidate["movie_id"] for candidate in history.json()["candidates"]}.isdisjoint(
        {10, 12}
    )

    restarted = TestClient(create_app(result.active_pointer_path))
    assert (
        restarted.post("/recommendations", json={"subject_id": 1, "top_n": 5}).json()
        == known.json()
    )
    assert (
        restarted.post("/recommendations", json={"history": [10, 12], "top_n": 5}).json()
        == history.json()
    )
    assert restarted.post("/recommendations", json={"top_n": 5}).json() == empty.json()


def test_fusion_training_metadata_stays_inside_snapshot_and_inner_validation(
    tmp_path: Path,
) -> None:
    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())
    split = result.temporal_split
    future_event_ids = {event.event_id for event in split.future_window_events}
    metadata = json.loads((result.artifact_path / "manifest.json").read_text(encoding="utf-8"))[
        "fusion_training"
    ]

    assert metadata["known_user"]["source_snapshot_fingerprint"] != split.data_snapshot.fingerprint
    assert metadata["known_user"]["validation_event_ids"]
    assert metadata["history_only"]["validation_event_ids"]
    assert not future_event_ids.intersection(metadata["known_user"]["validation_event_ids"])
    assert not future_event_ids.intersection(metadata["history_only"]["validation_event_ids"])
    assert result.history_only_evaluation.query_mode == "history_only"
    assert all(query.subject_id is None for query in result.history_only_evaluation.cohort)


def test_corrupt_lhf_payload_is_rejected_before_activation(tmp_path: Path) -> None:
    result = run_fast_lifecycle(tmp_path / "store", kafka=InMemoryKafkaBoundary())
    payload_path = result.artifact_path / "model.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["fusion"]["history_only"]["retriever_bank"] = ["popularity", "lightgcn"]
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="payload checksum"):
        ServingArtifact.load(result.artifact_path)
    pointer = json.loads(result.active_pointer_path.read_text(encoding="utf-8"))
    assert pointer == {"artifact_id": result.artifact_id}


def test_inner_validation_uses_no_future_gold_when_union_misses_it() -> None:
    split = split_temporal_events(load_movielens_fixture())
    assert set(event.event_id for event in split.future_window_events).isdisjoint(
        event.event_id for event in split.data_snapshot.events
    )


def test_future_window_mutation_cannot_change_lhf_training_or_serving_features(
    tmp_path: Path,
) -> None:
    events = load_movielens_fixture()
    split = split_temporal_events(events)
    changed_future_event = RatingEvent.from_movielens(
        subject_id=split.future_window_events[0].subject_id,
        movie_id=999,
        rating=5.0,
        event_time=split.future_window_events[0].event_time,
    )
    changed_events = (
        split.data_snapshot.events
        + (changed_future_event,)
        + split.future_window_events[1:]
        + events[split.future_boundary :]
    )

    first = run_fast_lifecycle(tmp_path / "first", InMemoryKafkaBoundary(), source_events=events)
    second = run_fast_lifecycle(
        tmp_path / "second",
        InMemoryKafkaBoundary(),
        source_events=changed_events,
    )
    first_payload = json.loads((first.artifact_path / "model.json").read_text(encoding="utf-8"))
    second_payload = json.loads((second.artifact_path / "model.json").read_text(encoding="utf-8"))
    first_manifest = json.loads((first.artifact_path / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads(
        (second.artifact_path / "manifest.json").read_text(encoding="utf-8")
    )

    assert first_manifest["fusion_training"] == second_manifest["fusion_training"]
    assert first_payload["fusion"] == second_payload["fusion"]
