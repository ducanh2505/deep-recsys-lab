from pathlib import Path

from fastapi.testclient import TestClient

from deep_recsys_lifecycle.api import create_app
from deep_recsys_lifecycle.kafka import InMemoryKafkaBoundary
from deep_recsys_lifecycle.lifecycle import run_fast_lifecycle


def _client(tmp_path: Path) -> TestClient:
    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())
    return TestClient(create_app(result.artifact_path))


def test_known_user_route_returns_unseen_candidates(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/recommendations", json={"subject_id": 1, "top_n": 4})

    assert response.status_code == 200
    body = response.json()
    assert body["query_mode"] == "known_user"
    assert body["retriever"] == "lhf"
    assert {candidate["movie_id"] for candidate in body["candidates"]}.isdisjoint({10, 11})


def test_history_only_route_returns_unseen_candidates(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/recommendations", json={"history": [10, 12], "top_n": 4})

    assert response.status_code == 200
    body = response.json()
    assert body["query_mode"] == "history_only"
    assert {candidate["movie_id"] for candidate in body["candidates"]}.isdisjoint({10, 12})


def test_empty_history_route_uses_global_popularity(tmp_path: Path) -> None:
    response = _client(tmp_path).post("/recommendations", json={"top_n": 4})

    assert response.status_code == 200
    body = response.json()
    assert body["query_mode"] == "empty_history"
    assert [candidate["movie_id"] for candidate in body["candidates"]][:2] == [10, 11]


def test_query_validation_defines_unknown_subject_and_mutually_exclusive_inputs(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)

    unknown_subject = client.post("/recommendations", json={"subject_id": 999})
    both_inputs = client.post("/recommendations", json={"subject_id": 1, "history": [10]})

    assert unknown_subject.status_code == 404
    assert unknown_subject.json()["detail"]["code"] == "unknown_subject"
    assert both_inputs.status_code == 422
