import json
import math
from pathlib import Path

import pytest

from deep_recsys_lifecycle.artifact import ServingArtifact
from deep_recsys_lifecycle.event_store import DataSnapshot
from deep_recsys_lifecycle.models import PositiveInteraction, RatingEvent
from deep_recsys_lifecycle.multivae import MultVAEConfig, fit_multivae
from deep_recsys_lifecycle.positive import derive_positive_interactions


def _snapshot() -> DataSnapshot:
    events = (
        RatingEvent.from_movielens(1, 10, 5.0, 1),
        RatingEvent.from_movielens(1, 10, 4.0, 2),
        RatingEvent.from_movielens(1, 11, 3.0, 3),
        RatingEvent.from_movielens(1, 12, 4.0, 4),
        RatingEvent.from_movielens(2, 11, 5.0, 5),
        RatingEvent.from_movielens(2, 12, 4.0, 6),
        RatingEvent.from_movielens(3, 13, 5.0, 7),
    )
    return DataSnapshot.from_events(events)


def _config() -> MultVAEConfig:
    return MultVAEConfig(hidden_dim=4, latent_dim=2, epochs=2, batch_size=8)


def test_multivae_profiles_are_binary_deduplicated_and_catalog_bounded() -> None:
    snapshot = _snapshot()
    model = fit_multivae(
        snapshot,
        derive_positive_interactions(snapshot.events),
        config=_config(),
        device_preference="cpu",
    )

    assert model.catalog == (10, 11, 12, 13)
    assert model.profile_for_history((10, 10, 999, 12)) == (1, 0, 1, 0)
    assert model.profile_for_subject(1) == (1, 0, 1, 0)
    assert model.profile_for_subject(2) == (0, 1, 1, 0)


def test_multivae_candidate_contract_is_bounded_finite_excluding_observed_items() -> None:
    snapshot = _snapshot()
    model = fit_multivae(
        snapshot,
        derive_positive_interactions(snapshot.events),
        config=_config(),
        device_preference="cpu",
    )

    candidates = model.candidate_pool((10, 10, 999), limit=3)

    assert len(candidates) == 3
    assert [candidate.rank for candidate in candidates] == [1, 2, 3]
    assert [candidate.movie_id for candidate in candidates] == sorted(
        candidate.movie_id for candidate in candidates
    ) or len({candidate.score for candidate in candidates}) == len(candidates)
    assert {candidate.movie_id for candidate in candidates}.isdisjoint({10, 999})
    assert all(math.isfinite(float(candidate.score)) for candidate in candidates)
    assert model.candidate_pool((10, 10, 999), limit=3) == candidates


def test_known_user_and_history_only_use_the_same_multivae_profile() -> None:
    snapshot = _snapshot()
    model = fit_multivae(
        snapshot,
        derive_positive_interactions(snapshot.events),
        config=_config(),
        device_preference="cpu",
    )
    history = (10, 12)

    assert model.profile_for_subject(1) == model.profile_for_history(history)
    assert model.candidate_pool(model.subject_histories[1], limit=4) == model.candidate_pool(
        history, limit=4
    )


def test_multivae_ignores_interactions_outside_the_snapshot() -> None:
    snapshot = _snapshot()
    snapshot_interactions = derive_positive_interactions(snapshot.events)
    future = PositiveInteraction.from_event(RatingEvent.from_movielens(1, 13, 5.0, 100))
    with_future = fit_multivae(
        snapshot,
        (*snapshot_interactions, future),
        config=_config(),
        device_preference="cpu",
    )
    without_future = fit_multivae(
        snapshot,
        snapshot_interactions,
        config=_config(),
        device_preference="cpu",
    )

    assert with_future.to_dict()["weights"] == without_future.to_dict()["weights"]


def test_multivae_records_cpu_fallback_when_mps_training_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deep_recsys_lifecycle.multivae as multivae_module

    original_train_once = multivae_module._train_once

    def fail_mps_once(*args: object, **kwargs: object):
        if args[-1] == "mps":
            raise RuntimeError("injected unsupported MPS operator")
        return original_train_once(*args, **kwargs)

    monkeypatch.setattr(multivae_module, "_train_once", fail_mps_once)
    model = fit_multivae(
        _snapshot(),
        derive_positive_interactions(_snapshot().events),
        config=_config(),
        device_preference="mps",
        mps_probe=lambda: True,
    )

    assert model.training_metadata["actual_device"] == "cpu"
    assert "injected unsupported MPS operator" in model.training_metadata["fallback_reason"]


def test_multivae_tied_scores_use_movie_id_order() -> None:
    from deep_recsys_lifecycle.serving import MultVAERetriever

    model = fit_multivae(
        _snapshot(),
        derive_positive_interactions(_snapshot().events),
        config=_config(),
        device_preference="cpu",
    )
    payload = model.to_dict()
    weights = payload["weights"]
    assert isinstance(weights, dict)
    for key, value in weights.items():
        if isinstance(value, list):
            weights[key] = (
                [[0.0 for _ in row] for row in value]
                if value and isinstance(value[0], list)
                else [0.0 for _ in value]
            )
    tied = MultVAERetriever.from_dict(payload)

    assert [candidate.movie_id for candidate in tied.candidate_pool((10,), limit=3)] == [11, 12, 13]


def test_multivae_payload_round_trip_preserves_candidates(tmp_path: Path) -> None:
    from deep_recsys_lifecycle.kafka import InMemoryKafkaBoundary
    from deep_recsys_lifecycle.lifecycle import run_fast_lifecycle

    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())
    artifact = ServingArtifact.load(result.artifact_path)
    assert artifact.multivae is not None

    query_history = artifact.multivae.subject_histories[min(artifact.multivae.subject_histories)]
    before = artifact.multivae.candidate_pool(query_history, limit=5)
    reloaded = ServingArtifact.load(result.artifact_path)

    assert reloaded.multivae is not None
    assert reloaded.multivae.candidate_pool(query_history, limit=5) == before
    assert reloaded.manifest["multivae_training"]["actual_device"] in {"cpu", "mps"}
    assert "multivae" in reloaded.manifest["evaluation_metrics"]
    report = result.report_path.read_text(encoding="utf-8")
    assert 'id="multivae-metrics"' in report
    assert 'id="multivae-resource-evidence"' in report


def test_corrupt_multivae_payload_is_rejected(tmp_path: Path) -> None:
    from deep_recsys_lifecycle.kafka import InMemoryKafkaBoundary
    from deep_recsys_lifecycle.lifecycle import run_fast_lifecycle

    result = run_fast_lifecycle(tmp_path / "fast", kafka=InMemoryKafkaBoundary())
    payload_path = result.artifact_path / "model.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    del payload["multivae"]["weights"]
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="payload checksum"):
        ServingArtifact.load(result.artifact_path)
