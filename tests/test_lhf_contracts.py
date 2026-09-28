import pytest

from deep_recsys_lifecycle.evaluation import EvaluationCohort, EvaluationQuery
from deep_recsys_lifecycle.event_store import DataSnapshot
from deep_recsys_lifecycle.fusion import (
    FEATURE_SCHEMA_VERSION,
    FusionFeatureBuilder,
    LearnedHybridFusion,
    build_candidate_union,
    build_lhf_training_rows,
    expected_feature_names,
    rank_lhf_union,
    train_lhf_classifier,
    untrained_fusion_classifier,
)
from deep_recsys_lifecycle.models import Candidate, RatingEvent
from deep_recsys_lifecycle.positive import derive_positive_interactions


def _builder() -> FusionFeatureBuilder:
    events = (
        RatingEvent.from_movielens(1, 10, 5.0, 1),
        RatingEvent.from_movielens(1, 11, 5.0, 2),
        RatingEvent.from_movielens(2, 10, 5.0, 3),
        RatingEvent.from_movielens(2, 12, 5.0, 4),
        RatingEvent.from_movielens(3, 13, 5.0, 5),
    )
    snapshot = DataSnapshot.from_events(events)
    return FusionFeatureBuilder.from_snapshot(
        snapshot,
        derive_positive_interactions(snapshot.events),
        retriever_bank=("popularity", "itemknn", "multivae", "lightgcn"),
    )


def test_union_is_bounded_deduplicated_and_excludes_observed_movies() -> None:
    pools = {
        "popularity": (
            Candidate(movie_id=10, score=9.0, rank=1),
            Candidate(movie_id=2, score=8.0, rank=2),
        ),
        "itemknn": (
            Candidate(movie_id=2, score=0.9, rank=1),
            Candidate(movie_id=3, score=0.8, rank=2),
        ),
        "multivae": (Candidate(movie_id=4, score=0.7, rank=1),),
    }

    union = build_candidate_union(pools, history=(10,), limit=200)

    assert [candidate.movie_id for candidate in union] == [2, 3, 4]


def test_feature_schema_has_fixed_missing_evidence_and_no_identifiers() -> None:
    builder = _builder()
    pools = {
        "popularity": (Candidate(movie_id=20, score=4.0, rank=2),),
        "itemknn": (),
        "multivae": (Candidate(movie_id=20, score=0.5, rank=1),),
    }

    features = builder.features_for(20, pools, history=(10, 99))

    assert builder.schema_version == FEATURE_SCHEMA_VERSION
    assert features == (
        1.0,
        2.0,
        4.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.5,
        0.0,
        0.0,
        0.0,
        2.0,
        1.0,
        1.5,
        2.0,
        1.0,
        0.0,
        0.0,
    )
    assert all(
        "movie" not in name.lower() and "subject" not in name.lower()
        for name in builder.feature_names
    )


def test_training_and_serving_feature_builders_have_identical_contract() -> None:
    events = (
        RatingEvent.from_movielens(1, 10, 5.0, 1),
        RatingEvent.from_movielens(1, 11, 5.0, 2),
        RatingEvent.from_movielens(2, 12, 5.0, 3),
    )
    snapshot = DataSnapshot.from_events(events)
    interactions = derive_positive_interactions(snapshot.events)
    training_builder = FusionFeatureBuilder.from_snapshot(
        snapshot,
        interactions,
        retriever_bank=("popularity", "itemknn", "multivae"),
    )
    serving_builder = FusionFeatureBuilder.from_serving(
        snapshot_fingerprint=snapshot.fingerprint,
        retriever_bank=training_builder.retriever_bank,
        catalog=tuple(event.movie_id for event in snapshot.events),
        item_popularity=training_builder.item_popularity,
        user_cold_history_threshold=training_builder.user_cold_history_threshold,
    )
    pools = {
        "popularity": (Candidate(movie_id=12, score=2.0, rank=1),),
        "itemknn": (),
        "multivae": (Candidate(movie_id=12, score=0.5, rank=1),),
    }

    assert serving_builder.feature_schema == training_builder.feature_schema
    assert serving_builder.features_for(12, pools, (10,)) == training_builder.features_for(
        12, pools, (10,)
    )


def test_gold_outside_union_is_not_injected_and_does_not_create_positive_label() -> None:
    builder = _builder()
    cohort = EvaluationCohort(
        queries=(
            EvaluationQuery(
                subject_id=1,
                history=(10,),
                gold_movie_id=999,
                query_mode="known_user",
            ),
        )
    )
    pools = {
        1: {
            "popularity": (Candidate(movie_id=20, score=1.0, rank=1),),
            "itemknn": (),
            "multivae": (),
            "lightgcn": (),
        }
    }

    rows = build_lhf_training_rows(cohort, pools, builder)
    classifier = train_lhf_classifier(
        query_mode="known_user",
        retriever_bank=builder.retriever_bank,
        feature_names=builder.feature_names,
        rows=rows,
        source_snapshot_fingerprint=builder.snapshot_fingerprint,
        seed=42,
    )

    assert len(rows) == 1
    assert rows[0].movie_id == 20
    assert rows[0].label == 0
    assert classifier.training_status == "untrained fallback"
    assert classifier.training_metadata["positive_row_count"] == 0


def test_history_only_rows_have_no_subject_identity_or_lightgcn_feature() -> None:
    builder = FusionFeatureBuilder.from_snapshot(
        DataSnapshot.from_events(
            (
                RatingEvent.from_movielens(1, 10, 5.0, 1),
                RatingEvent.from_movielens(2, 11, 5.0, 2),
            )
        ),
        (),
        retriever_bank=("popularity", "itemknn", "multivae"),
    )
    cohort = EvaluationCohort(
        queries=(
            EvaluationQuery(
                subject_id=None,
                query_id="history-query-1",
                history=(10,),
                gold_movie_id=12,
                query_mode="history_only",
            ),
        )
    )
    pools = {
        "history-query-1": {
            "popularity": (Candidate(movie_id=12, score=1.0, rank=1),),
            "itemknn": (),
            "multivae": (),
        }
    }

    rows = build_lhf_training_rows(cohort, pools, builder)

    assert rows[0].query_key == "history-query-1"
    assert "lightgcn" not in builder.retriever_bank
    assert all("lightgcn" not in name for name in builder.feature_names)


def test_fusion_row_budget_keeps_gold_and_selects_negatives_deterministically() -> None:
    builder = _builder()
    cohort = EvaluationCohort(
        queries=(EvaluationQuery(subject_id=1, history=(10,), gold_movie_id=15),)
    )
    candidates = tuple(
        Candidate(movie_id=movie_id, score=float(movie_id), rank=rank)
        for rank, movie_id in enumerate(range(11, 31), start=1)
    )
    pools = {1: {name: candidates for name in builder.retriever_bank}}

    first = build_lhf_training_rows(cohort, pools, builder, max_negative_rows_per_query=3, seed=42)
    second = build_lhf_training_rows(cohort, pools, builder, max_negative_rows_per_query=3, seed=42)

    assert first == second
    assert len(first) == 4
    assert [row.movie_id for row in first] == sorted(row.movie_id for row in first)
    assert [(row.movie_id, row.label) for row in first if row.label] == [(15, 1)]


def test_fusion_row_budget_validates_limit_and_seed() -> None:
    builder = _builder()
    with pytest.raises(ValueError, match="max_negative_rows_per_query"):
        build_lhf_training_rows((), {}, builder, max_negative_rows_per_query=0)
    with pytest.raises(ValueError, match="sampling seed"):
        build_lhf_training_rows((), {}, builder, seed=True)


def test_trained_classifier_round_trip_and_fallback_order_are_deterministic() -> None:
    builder = _builder()
    names = expected_feature_names(builder.retriever_bank)
    rows = tuple(
        build_lhf_training_rows(
            EvaluationCohort(
                queries=(
                    EvaluationQuery(
                        subject_id=query_id,
                        history=(),
                        gold_movie_id=query_id,
                    ),
                )
            ),
            {
                query_id: {
                    "popularity": (Candidate(movie_id=query_id, score=1.0, rank=1),),
                    "itemknn": (),
                    "multivae": (),
                    "lightgcn": (),
                }
            },
            builder,
        )[0]
        for query_id in (1, 2)
    ) + tuple(
        build_lhf_training_rows(
            EvaluationCohort(
                queries=(
                    EvaluationQuery(
                        subject_id=query_id,
                        history=(),
                        gold_movie_id=999,
                    ),
                )
            ),
            {
                query_id: {
                    "popularity": (Candidate(movie_id=query_id, score=0.0, rank=1),),
                    "itemknn": (),
                    "multivae": (),
                    "lightgcn": (),
                }
            },
            builder,
        )[0]
        for query_id in (3, 4)
    )
    classifier = train_lhf_classifier(
        query_mode="known_user",
        retriever_bank=builder.retriever_bank,
        feature_names=names,
        rows=rows,
        source_snapshot_fingerprint=builder.snapshot_fingerprint,
    )
    loaded = classifier.from_dict(classifier.to_dict())

    assert classifier.training_status == "trained"
    assert loaded.training_status == "trained"
    assert loaded.score(rows[0].features) == classifier.score(rows[0].features)
    assert loaded.score_many((rows[0].features, rows[1].features)) == (
        loaded.score(rows[0].features),
        loaded.score(rows[1].features),
    )

    fallback = untrained_fusion_classifier(
        "known_user", source_snapshot_fingerprint=builder.snapshot_fingerprint
    )
    pools = {
        "popularity": (
            Candidate(movie_id=8, score=1.0, rank=1),
            Candidate(movie_id=7, score=1.0, rank=2),
        ),
        "itemknn": (),
        "multivae": (),
        "lightgcn": (),
    }
    ranked = rank_lhf_union(fallback, builder, pools, (), top_n=2)
    assert [candidate.movie_id for candidate in ranked] == [8, 7]


def test_trained_union_uses_one_predict_call_and_best_ranked_duplicate_evidence() -> None:
    builder = _builder()

    class RecordingBooster:
        def __init__(self) -> None:
            self.calls: list[tuple[list[list[float]], int]] = []

        def predict(self, rows: list[list[float]], *, num_threads: int) -> list[float]:
            self.calls.append((rows, num_threads))
            return [0.4, 0.8, 0.4]

    booster = RecordingBooster()
    classifier = LearnedHybridFusion(
        query_mode="known_user",
        retriever_bank=builder.retriever_bank,
        feature_names=builder.feature_names,
        feature_schema=builder.feature_schema,
        training_status="trained",
        training_metadata={},
        _booster=booster,
    )
    pools = {
        "popularity": (
            Candidate(movie_id=20, score=1.0, rank=3),
            Candidate(movie_id=21, score=2.0, rank=1),
            Candidate(movie_id=20, score=3.0, rank=2),
        ),
        "itemknn": (Candidate(movie_id=22, score=0.5, rank=1),),
        "multivae": (),
        "lightgcn": (),
    }

    ranked = rank_lhf_union(classifier, builder, pools, (10, 10), top_n=3)

    assert [(candidate.movie_id, candidate.score, candidate.rank) for candidate in ranked] == [
        (21, 0.8, 1),
        (20, 0.4, 2),
        (22, 0.4, 3),
    ]
    assert booster.calls == [
        ([list(builder.features_for(movie_id, pools, (10, 10))) for movie_id in (20, 21, 22)], 1)
    ]
    assert booster.calls[0][0][0][:3] == [1.0, 2.0, 3.0]

    training_rows = build_lhf_training_rows(
        EvaluationCohort(
            queries=(EvaluationQuery(subject_id=1, history=(10, 10), gold_movie_id=20),)
        ),
        {1: pools},
        builder,
    )
    assert [(row.movie_id, row.features) for row in training_rows] == [
        (movie_id, builder.features_for(movie_id, pools, (10, 10))) for movie_id in (20, 21, 22)
    ]
