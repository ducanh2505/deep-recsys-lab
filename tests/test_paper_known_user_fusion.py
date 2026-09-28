from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

import pytest

from deep_recsys_lifecycle.fusion import FusionFeatureBuilder, build_lhf_training_rows
from deep_recsys_lifecycle.lightgcn import LightGCNConfig
from deep_recsys_lifecycle.models import Candidate, RatingEvent
from deep_recsys_lifecycle.multivae import MultVAEConfig
from deep_recsys_lifecycle.paper_known_user_fusion import (
    KnownUserHybridConfig,
    _inner_holdout,
    evaluate_known_user_frozen_test,
    run_known_user_hybrid_benchmark,
    write_known_user_hybrid_report,
)


def _events() -> tuple[RatingEvent, ...]:
    subjects = (2, 28, 29, 34, 42, 44, 46, 53, 70, 71)
    return tuple(
        RatingEvent.from_movielens(
            subject_id=subject_id,
            movie_id=movie_id,
            rating=5.0,
            event_time=subject_id * 1_000 + movie_id,
        )
        for subject_id in subjects
        for movie_id in (*range(1, 41), 99)
    )


def _small_config() -> KnownUserHybridConfig:
    return KnownUserHybridConfig(
        seed=42,
        inner_seed=91,
        multivae=MultVAEConfig(epochs=1, hidden_dim=8, latent_dim=4),
        lightgcn=LightGCNConfig(epochs=1, embedding_dim=4, layers=1),
        capture_training_rows=True,
        capture_query_evidence=True,
    )


class _MultiGoldQuery:
    subject_id = 7
    history = (10,)
    gold_movie_ids = (20, 30, 40)


def test_multi_gold_rows_label_every_naturally_retrieved_gold_without_injection() -> None:
    builder = FusionFeatureBuilder.from_serving(
        snapshot_fingerprint="inner-fit",
        retriever_bank=("popularity", "itemknn", "multivae", "lightgcn"),
        catalog=(10, 20, 30, 40, 50),
        item_popularity={10: 2, 20: 1, 30: 1, 40: 1, 50: 2},
    )
    pools = {
        7: {
            "popularity": (
                Candidate(20, 4.0, 1),
                Candidate(50, 3.0, 2),
            ),
            "itemknn": (Candidate(30, 0.8, 1),),
            "multivae": (Candidate(10, 0.7, 1),),
            "lightgcn": (),
        }
    }

    rows = build_lhf_training_rows(
        (_MultiGoldQuery(),), pools, builder, max_negative_rows_per_query=1
    )

    assert {(row.movie_id, row.label) for row in rows} == {(20, 1), (30, 1), (50, 0)}
    assert 40 not in {row.movie_id for row in rows}
    assert 10 not in {row.movie_id for row in rows}


def test_inner_interaction_folds_are_deterministic_disjoint_and_keep_subject_ids() -> None:
    events = tuple(
        RatingEvent.from_movielens(subject_id, movie_id, 5.0, subject_id * 100 + movie_id)
        for subject_id in (1, 2)
        for movie_id in range(1, 11)
    )
    first_fit, first_holdout = _inner_holdout(events, seed=91, fold_index=0)
    second_fit, second_holdout = _inner_holdout(events, seed=91, fold_index=1)

    assert _inner_holdout(reversed(events), seed=91, fold_index=0) == (
        first_fit,
        first_holdout,
    )
    assert {event.event_id for event in first_holdout}.isdisjoint(
        event.event_id for event in second_holdout
    )
    assert {event.subject_id for event in first_fit} == {1, 2}
    assert {event.subject_id for event in second_fit} == {1, 2}


def test_public_known_user_hybrid_uses_train_only_evidence_and_final_lhf_order(tmp_path) -> None:
    events = _events()
    result = run_known_user_hybrid_benchmark(events, config=_small_config())
    report = result.to_dict()

    assert report["evaluated_partition"] == "validation"
    assert report["test_status"] == "sealed"
    assert report["retriever_bank"] == ["popularity", "itemknn", "multivae", "lightgcn"]
    assert report["configuration"]["inner_seed"] == 91
    assert report["configuration"]["inner_fold_count"] == 1
    assert "test_metrics" not in report
    assert all(
        "test_movie_ids" not in split
        for split in report["split_membership_by_subject"].values()
    )
    assert all(not split.test_movie_ids for split in result.split_by_subject.values())
    assert set(report["evaluation"]) >= {"source_pools", "oracle_union", "final_lhf"}
    assert set(report["evaluation"]["source_pools"]) == {
        "popularity",
        "itemknn",
        "multivae",
        "lightgcn",
    }
    assert report["metrics"] == report["evaluation"]["final_lhf"]
    assert report["evaluation"]["oracle_union"]["final_order"] is False
    assert report["training"]["fusion"]["training_status"] == result.fusion.training_status
    assert report["training"]["paper_validation_or_test_in_fit"] is False

    fit_pairs = {
        (subject_id, movie_id)
        for subject_id, split in result.split_by_subject.items()
        for movie_id in split.train_movie_ids
    }
    fit_event_ids = {
        event.event_id
        for event in events
        if (event.subject_id, event.movie_id) in fit_pairs
    }
    reserved_event_ids = {
        event.event_id
        for event in events
        if (event.subject_id, event.movie_id) not in fit_pairs
    }
    inner_gold_event_ids = set(result.fusion.training_metadata["validation_event_ids"])
    assert inner_gold_event_ids
    assert inner_gold_event_ids <= fit_event_ids
    assert not inner_gold_event_ids & reserved_event_ids
    assert sum(row.label for row in result.inner_training_rows) >= 2
    assert all(row.movie_id in result.training_catalog for row in result.inner_training_rows)
    expected_popularity = Counter(
        movie_id for _subject_id, movie_id in fit_pairs
    )
    assert result._feature_builder.item_popularity == expected_popularity
    assert 99 not in result.training_catalog
    assert 99 not in result._feature_builder.item_popularity
    assert set(result._retrievers) == {
        "popularity",
        "itemknn",
        "multivae",
        "lightgcn",
    }

    validation_gold = report["validation_gold_sets_by_subject"]
    assert result.validation_rankings
    for subject_id, evidence in result.validation_query_evidence.items():
        split = result.split_by_subject[subject_id]
        source_pools = evidence["source_pool_movie_ids"]
        final = evidence["final_movie_ids"]
        assert set(source_pools) == set(result._retrievers)
        assert all(
            set(pool) <= result.training_catalog - set(split.train_movie_ids)
            for pool in source_pools.values()
        )
        assert set(final) <= set(evidence["oracle_union_movie_ids"])
        assert len(final) <= 100
        assert evidence["final_ranks"] == list(range(1, len(final) + 1))
        assert tuple(final) == result.validation_rankings[subject_id]
        assert validation_gold[str(subject_id)] == evidence["gold_movie_ids"]
        hits = len(set(final[:10]) & set(evidence["gold_movie_ids"]))
        expected = hits / len(evidence["gold_movie_ids"])
        assert result.validation_subject_metrics[subject_id]["Recall@10"] == pytest.approx(expected)
    assert report["metrics"]["Recall@10"] == pytest.approx(
        sum(value["Recall@10"] for value in result.validation_subject_metrics.values())
        / len(result.validation_subject_metrics)
    )

    frozen_test = evaluate_known_user_frozen_test(result)
    assert frozen_test["evaluated_partition"] == "test"
    assert frozen_test["training_snapshot_fingerprint"] == report["training"][
        "outer_snapshot_fingerprint"
    ]
    assert result.to_dict()["test_status"] == "sealed"
    path = write_known_user_hybrid_report(result, tmp_path / "paper" / "validation.json")
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_public_known_user_hybrid_accepts_multiple_disjoint_inner_folds() -> None:
    config = replace(
        _small_config(),
        inner_fold_count=2,
        capture_training_rows=False,
        capture_query_evidence=False,
    )

    result = run_known_user_hybrid_benchmark(_events(), config=config)

    assert result.to_dict()["configuration"]["inner_fold_count"] == 2
    sources = result.fusion.training_metadata["validation_sources"]
    assert [source["fold_index"] for source in sources] == [0, 1]
    assert sources[0]["heldout_event_ids_sha256"] != sources[1]["heldout_event_ids_sha256"]
    assert result.to_dict()["counts"]["inner_training_query_count"] == 20
    assert result.validation_rankings
    assert result.inner_training_rows == ()
    assert result.fusion.training_metadata["row_count"] > 0
    assert result.validation_query_evidence == {}
