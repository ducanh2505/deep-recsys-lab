import pytest

from deep_recsys_lifecycle.dense_scoring import LightGCNScorer, MultVAEScorer
from deep_recsys_lifecycle.event_store import DataSnapshot
from deep_recsys_lifecycle.fixture import load_movielens_fixture
from deep_recsys_lifecycle.lightgcn import LightGCNConfig, fit_lightgcn
from deep_recsys_lifecycle.multivae import MultVAEConfig, fit_multivae
from deep_recsys_lifecycle.positive import derive_positive_interactions


def test_dense_multivae_scoring_matches_cpu_contract() -> None:
    snapshot = DataSnapshot.from_events(load_movielens_fixture())
    model = fit_multivae(
        snapshot, derive_positive_interactions(snapshot.events),
        config=MultVAEConfig(epochs=1), device_preference="cpu",
    )
    scorer = MultVAEScorer(model.catalog, {
        "encoder_weight": model.encoder_weight, "encoder_bias": model.encoder_bias,
        "mean_weight": model.mean_weight, "mean_bias": model.mean_bias,
        "decoder_weight": model.decoder_weight, "decoder_bias": model.decoder_bias,
        "output_weight": model.output_weight, "output_bias": model.output_bias,
    })

    for history in ((10,), (10, 11), (999,)):
        expected = model.candidate_pool(history, limit=5)
        actual = scorer.candidate_pool(history, limit=5)
        assert [item.movie_id for item in actual] == [item.movie_id for item in expected]
        assert [item.score for item in actual] == pytest.approx(
            [item.score for item in expected]
        )


def test_dense_lightgcn_scoring_matches_cpu_contract() -> None:
    snapshot = DataSnapshot.from_events(load_movielens_fixture())
    model = fit_lightgcn(
        snapshot, derive_positive_interactions(snapshot.events),
        config=LightGCNConfig(epochs=1), device_preference="cpu",
    )
    scorer = LightGCNScorer(
        model.subject_ids, model.movie_ids,
        model.subject_embeddings, model.movie_embeddings,
    )
    subject_id = model.subject_ids[0]
    history = model.subject_histories[subject_id]
    expected = model.candidate_pool_for_subject(subject_id, history, limit=5)
    actual = scorer.candidate_pool(subject_id, history, limit=5)
    assert [item.movie_id for item in actual] == [item.movie_id for item in expected]
    assert [item.score for item in actual] == pytest.approx(
        [item.score for item in expected]
    )
    with pytest.raises(KeyError, match="no embedding"):
        scorer.candidate_pool(999_999, (), limit=5)
