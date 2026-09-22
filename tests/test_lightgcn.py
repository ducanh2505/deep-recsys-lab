import json
import math
from pathlib import Path

import pytest

from deep_recsys_lifecycle.artifact import ServingArtifact
from deep_recsys_lifecycle.evaluation import (
    EvaluationCohort,
    EvaluationQuery,
    evaluate_retrievers,
)
from deep_recsys_lifecycle.event_store import DataSnapshot
from deep_recsys_lifecycle.kafka import InMemoryKafkaBoundary
from deep_recsys_lifecycle.lifecycle import run_fast_lifecycle
from deep_recsys_lifecycle.lightgcn import LightGCNConfig, fit_lightgcn
from deep_recsys_lifecycle.models import Candidate, PositiveInteraction, RatingEvent
from deep_recsys_lifecycle.positive import derive_positive_interactions
from deep_recsys_lifecycle.query import (
    EmptyHistoryQuery,
    HistoryOnlyQuery,
    KnownUserQuery,
    RecommendationService,
    UnknownSubjectError,
)
from deep_recsys_lifecycle.serving import LightGCNRetriever


def _snapshot() -> DataSnapshot:
    events = (
        RatingEvent.from_movielens(1, 10, 5.0, 1),
        RatingEvent.from_movielens(1, 11, 4.0, 2),
        RatingEvent.from_movielens(1, 12, 3.0, 3),
        RatingEvent.from_movielens(2, 10, 4.0, 4),
        RatingEvent.from_movielens(2, 13, 5.0, 5),
        RatingEvent.from_movielens(3, 14, 2.0, 6),
    )
    return DataSnapshot.from_events(events)


def _config() -> LightGCNConfig:
    return LightGCNConfig(
        seed=7,
        embedding_dim=4,
        layers=1,
        epochs=3,
        batch_size=4,
        negative_samples=1,
    )


def _fit(*, interactions=None, **kwargs: object) -> LightGCNRetriever:
    snapshot = _snapshot()
    values = interactions or derive_positive_interactions(snapshot.events)
    device_preference = str(kwargs.pop("device_preference", "cpu"))
    return fit_lightgcn(
        snapshot,
        values,
        config=_config(),
        device_preference=device_preference,
        **kwargs,
    )


def test_lightgcn_graph_is_snapshot_bounded_and_positive_only() -> None:
    snapshot = _snapshot()
    snapshot_interactions = derive_positive_interactions(snapshot.events)
    future_interaction = PositiveInteraction.from_event(RatingEvent.from_movielens(1, 99, 5.0, 100))

    without_future = _fit(interactions=snapshot_interactions)
    with_future = _fit(interactions=(*snapshot_interactions, future_interaction))

    assert with_future.movie_ids == (10, 11, 12, 13, 14)
    assert with_future.subject_ids == (1, 2)
    assert with_future.positive_edges == ((1, 10), (1, 11), (2, 10), (2, 13))
    assert with_future.subject_embeddings == without_future.subject_embeddings
    assert with_future.movie_embeddings == without_future.movie_embeddings


def test_lightgcn_scores_full_catalog_deterministically_and_excludes_observed() -> None:
    model = _fit()

    first = model.candidate_pool_for_subject(1, model.subject_histories[1], limit=200)
    second = model.candidate_pool_for_subject(1, model.subject_histories[1], limit=200)

    assert first == second
    assert len(first) == len(model.movie_ids) - len(model.subject_histories[1])
    assert [candidate.rank for candidate in first] == list(range(1, len(first) + 1))
    assert {candidate.movie_id for candidate in first}.isdisjoint(model.subject_histories[1])
    assert {candidate.movie_id for candidate in first} <= set(model.movie_ids)
    assert all(math.isfinite(float(candidate.score)) for candidate in first)


def test_lightgcn_ties_use_movie_id_order_and_unknown_subject_is_rejected() -> None:
    model = _fit()
    payload = model.to_dict()
    dimension = model.configuration["embedding_dim"]
    assert isinstance(dimension, int)
    payload["subject_embeddings"] = [[0.0] * dimension for _ in model.subject_ids]
    payload["movie_embeddings"] = [[0.0] * dimension for _ in model.movie_ids]
    tied = LightGCNRetriever.from_dict(payload)

    candidates = tied.candidate_pool_for_subject(1, (10, 11), limit=200)
    assert [candidate.movie_id for candidate in candidates] == [12, 13, 14]
    with pytest.raises(KeyError, match="LightGCN has no embedding for Subject 999"):
        tied.candidate_pool_for_subject(999, (), limit=3)


def test_lightgcn_device_fallbacks_are_explicit_and_injectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = _fit(
        device_preference="mps",
        mps_probe=lambda: False,
    )
    assert unavailable.training_metadata["requested_device"] == "mps"
    assert unavailable.training_metadata["actual_device"] == "cpu"
    assert "unavailable" in str(unavailable.training_metadata["fallback_reason"])

    slower = _fit(
        device_preference="mps",
        mps_probe=lambda: True,
        device_benchmark=lambda device: 2.0 if device == "mps" else 1.0,
    )
    assert slower.training_metadata["actual_device"] == "cpu"
    assert "slower" in str(slower.training_metadata["fallback_reason"])
    assert slower.training_metadata["benchmark_seconds"] == {"cpu": 1.0, "mps": 2.0}

    import deep_recsys_lifecycle.lightgcn as lightgcn_module

    original_train_once = lightgcn_module._train_once

    def fail_mps_once(*args: object, **kwargs: object):
        if args[-1] == "mps":
            raise RuntimeError("injected unsupported MPS operator")
        return original_train_once(*args, **kwargs)

    monkeypatch.setattr(lightgcn_module, "_train_once", fail_mps_once)
    failed = _fit(
        device_preference="mps",
        mps_probe=lambda: True,
        device_benchmark=lambda _device: 1.0,
    )
    assert failed.training_metadata["actual_device"] == "cpu"
    assert "injected unsupported MPS operator" in str(failed.training_metadata["fallback_reason"])


def test_evaluation_passes_subject_identity_only_through_subject_aware_seam() -> None:
    calls: list[int] = []

    class SubjectAwareProbe:
        name = "probe"

        def candidate_pool(self, _history: tuple[int, ...], limit: int = 200):
            raise AssertionError("history-only seam was used for a subject-aware retriever")

        def candidate_pool_for_subject(
            self, subject_id: int, _history: tuple[int, ...], limit: int = 200
        ):
            calls.append(subject_id)
            return tuple(
                Candidate(movie_id=movie_id, score=1.0, rank=rank)
                for rank, movie_id in enumerate((20, 21)[:limit], start=1)
            )

    cohort = EvaluationCohort(
        queries=(EvaluationQuery(subject_id=7, history=(10,), gold_movie_id=20),)
    )
    report = evaluate_retrievers({"probe": SubjectAwareProbe()}, cohort)

    assert calls == [7]
    assert report.retrievers["probe"].metrics.coverage_at_200 == 1.0
    with pytest.raises(ValueError, match="Known-User"):
        EvaluationQuery(
            subject_id=7,
            history=(10,),
            gold_movie_id=20,
            query_mode="history_only",
        )


def test_lightgcn_serving_supports_known_user_only(tmp_path: Path) -> None:
    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())
    artifact = ServingArtifact.load(result.artifact_path)
    service = RecommendationService(artifact)

    known = service.recommend_with_retriever(KnownUserQuery(subject_id=1), "lightgcn", top_n=5)
    assert known.retriever == "lightgcn"
    assert known.query_mode == "known_user"
    assert (
        known.candidates
        == service.recommend_with_retriever(
            KnownUserQuery(subject_id=1), "lightgcn", top_n=5
        ).candidates
    )

    with pytest.raises(ValueError, match="Known-User"):
        service.recommend_with_retriever(HistoryOnlyQuery(movie_ids=(10,)), "lightgcn")
    with pytest.raises(ValueError, match="Known-User"):
        service.recommend_with_retriever(EmptyHistoryQuery(), "lightgcn")
    with pytest.raises(UnknownSubjectError):
        service.recommend_with_retriever(KnownUserQuery(subject_id=999), "lightgcn")


def test_lightgcn_payload_round_trip_and_report_resource_evidence(tmp_path: Path) -> None:
    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())
    artifact = ServingArtifact.load(result.artifact_path)
    assert artifact.lightgcn is not None
    assert set(result.evaluation.retrievers) == {
        "popularity",
        "itemknn",
        "multivae",
        "lightgcn",
    }
    assert result.evaluation.rrf.name == "rrf"
    before = (
        RecommendationService(artifact)
        .recommend_with_retriever(KnownUserQuery(subject_id=1), "lightgcn", top_n=5)
        .candidates
    )

    reloaded = ServingArtifact.load(result.artifact_path)
    assert reloaded.lightgcn is not None
    after = (
        RecommendationService(reloaded)
        .recommend_with_retriever(KnownUserQuery(subject_id=1), "lightgcn", top_n=5)
        .candidates
    )
    assert after == before
    assert reloaded.manifest["lightgcn_training"]["actual_device"] == "cpu"
    assert "lightgcn" in reloaded.manifest["evaluation_metrics"]

    report = result.report_path.read_text(encoding="utf-8")
    assert 'id="lightgcn-metrics"' in report
    assert 'id="lightgcn-resource-evidence"' in report
    assert "History-Only and Empty-History queries do not support LightGCN" in report


def test_corrupt_lightgcn_payload_is_rejected(tmp_path: Path) -> None:
    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())
    payload_path = result.artifact_path / "model.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    del payload["lightgcn"]["movie_embeddings"]
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="payload checksum"):
        ServingArtifact.load(result.artifact_path)


def test_failed_lightgcn_smoke_preserves_previous_active_artifact(tmp_path: Path) -> None:
    store_path = tmp_path / "fast"
    first = run_fast_lifecycle(store_path, kafka=InMemoryKafkaBoundary())
    old_pointer = first.active_pointer_path.read_text(encoding="utf-8")

    def fail_smoke(_artifact: ServingArtifact) -> None:
        raise RuntimeError("injected LightGCN smoke failure")

    with pytest.raises(RuntimeError, match="injected LightGCN smoke failure"):
        run_fast_lifecycle(
            store_path,
            kafka=InMemoryKafkaBoundary(),
            smoke_validator=fail_smoke,
        )

    assert first.active_pointer_path.read_text(encoding="utf-8") == old_pointer
    assert ServingArtifact.load(first.active_pointer_path).artifact_id == first.artifact_id
