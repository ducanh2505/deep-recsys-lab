from deep_recsys_lifecycle.evaluation import EvaluationCohort, EvaluationQuery
from deep_recsys_lifecycle.event_store import DataSnapshot
from deep_recsys_lifecycle.fusion import (
    FEATURE_SCHEMA_VERSION,
    FusionFeatureBuilder,
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
