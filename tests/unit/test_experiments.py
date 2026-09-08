from __future__ import annotations

from pathlib import Path

import pytest

from recsys.core.io import read_json
from recsys.evaluation import ranking_metrics
from recsys.experiments.state import RunState


def test_ranking_metrics() -> None:
    metrics = ranking_metrics([[1, 2], [4, 3]], [{2}, {4, 5}], k=2)
    assert metrics["evaluated_users"] == 2
    assert metrics["recall@2"] == 0.75
    assert 0 < float(metrics["ndcg@2"]) <= 1
    assert ranking_metrics([], [], k=2)["evaluated_users"] == 0


def test_run_state_records_success_and_failure(tmp_path: Path) -> None:
    state = RunState.open(tmp_path / "state.json")
    assert state.stage("ok", lambda: 4) == 4
    assert read_json(state.path)["stages"]["ok"]["status"] == "complete"

    def fail() -> int:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        state.stage("bad", fail)
    assert read_json(state.path)["stages"]["bad"]["status"] == "failed"
    reopened = RunState.open(state.path)
    assert "ok" in reopened.value["stages"]


def test_run_state_resumes_serialized_stage(tmp_path: Path) -> None:
    state = RunState.open(tmp_path / "state.json")
    calls = 0

    def operation() -> int:
        nonlocal calls
        calls += 1
        return 5

    assert state.stage("value", operation, encode=str, decode=int) == 5
    assert state.stage("value", operation, encode=str, decode=int) == 5
    assert calls == 1
