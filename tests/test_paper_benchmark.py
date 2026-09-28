from __future__ import annotations

import json

import pytest

from deep_recsys_lifecycle.models import RatingEvent
from deep_recsys_lifecycle.paper_benchmark import (
    run_known_user_benchmark,
    write_known_user_benchmark_report,
)

_USERS = (2, 28, 29, 34, 42, 44, 46, 53, 70, 71)
_CORE_MOVIES = tuple(range(1, 41))
_COLD_MOVIE = 99


def _known_user_events() -> tuple[RatingEvent, ...]:
    events = [
        RatingEvent.from_movielens(
            subject_id=subject_id,
            movie_id=movie_id,
            rating=5.0,
            event_time=subject_id * 1_000 + movie_id,
        )
        for subject_id in _USERS
        for movie_id in (*_CORE_MOVIES, _COLD_MOVIE)
    ]
    events.extend(
        RatingEvent.from_movielens(
            subject_id=999,
            movie_id=movie_id,
            rating=5.0,
            event_time=999_000 + movie_id,
        )
        for movie_id in range(1_000, 1_010)
    )
    events.append(
        RatingEvent.from_movielens(
            subject_id=2,
            movie_id=777,
            rating=3.0,
            event_time=2_777,
        )
    )
    return tuple(events)


def _metric_events() -> tuple[RatingEvent, ...]:
    special_subjects = (2, 28, 29, 34, 42, 53, 70, 71, 84, 86)
    other_subjects = tuple(range(10_000, 10_010))
    users = (*special_subjects, *other_subjects)
    events = []
    for profile_index, subject_id in enumerate(users):
        start = profile_index * 2
        movie_ids = tuple(((start + offset) % 40) + 1 for offset in range(20))
        if subject_id in special_subjects:
            movie_ids = (*movie_ids, _COLD_MOVIE)
        events.extend(
            RatingEvent.from_movielens(
                subject_id=subject_id,
                movie_id=movie_id,
                rating=5.0,
                event_time=subject_id * 1_000 + movie_id,
            )
            for movie_id in movie_ids
        )
    return tuple(events)


def test_known_user_run_filters_positive_10core_and_reproduces_splits() -> None:
    events = _known_user_events()

    result = run_known_user_benchmark(events)
    replay = run_known_user_benchmark(reversed(events))
    report = result.to_dict()

    counts = report["counts"]
    assert counts["rating_event_count"] == 421
    assert counts["positive_rating_event_count"] == 420
    assert counts["positive_interaction_count_before_10_core"] == 420
    assert counts["positive_interaction_count_after_10_core"] == 410
    assert counts["eligible_subject_count"] == 10
    assert counts["outer_train_interaction_count"] == 320
    assert counts["validation_interaction_count"] == 40
    assert counts["training_interaction_count"] == 280
    assert counts["test_interaction_count"] == 90

    assert result.split_by_subject == replay.split_by_subject
    assert result.gold_sets == replay.gold_sets
    assert result.to_dict() == replay.to_dict()
    subject_two = result.split_by_subject[2]
    assert subject_two.train_movie_ids == (
        1,
        2,
        3,
        4,
        6,
        7,
        8,
        9,
        11,
        12,
        13,
        16,
        17,
        18,
        19,
        21,
        22,
        23,
        24,
        25,
        27,
        30,
        32,
        33,
        34,
        36,
        37,
        39,
    )
    assert subject_two.validation_movie_ids == (5, 14, 15, 40)
    assert subject_two.test_movie_ids == (10, 20, 26, 28, 29, 31, 35, 38, 99)
    assert subject_two.outer_train_movie_ids == tuple(
        sorted((*subject_two.train_movie_ids, *subject_two.validation_movie_ids))
    )
    assert not set(subject_two.outer_train_movie_ids) & set(subject_two.test_movie_ids)
    assert all(99 not in split.train_movie_ids for split in result.split_by_subject.values())

    exclusions = report["exclusions"]
    assert exclusions["positive_interaction_count_removed_by_10_core"] == 10
    assert exclusions["test_gold_movie_count_missing_from_training_catalog"] == 19
    assert exclusions["subjects_without_eligible_test_gold"] == 0
    reproducibility = report["reproducibility"]
    assert reproducibility["split_seed"] == 42
    assert reproducibility["split_algorithm"] == "sha256-ranked-per-subject-v1"
    assert reproducibility["subject_cohort_cap"] is None
    assert reproducibility["candidate_catalog_source"] == "final_training_interactions"
    assert report["source"]["input_event_fingerprint_sha256"]
    assert report["source"] == replay.to_dict()["source"]


def test_known_user_metrics_macro_average_complete_gold_sets() -> None:
    result = run_known_user_benchmark(_metric_events())

    assert result.gold_sets[2] == (1, 10, 17, 20)
    assert result.gold_sets[28] == (8, 10, 11, 13)
    metrics = result.to_dict()["metrics"]

    assert metrics["query_count"] == 20
    assert metrics["Recall@10"] == pytest.approx(0.175)
    assert metrics["Recall@20"] == pytest.approx(0.6125)
    assert metrics["Recall@50"] == 1.0
    assert metrics["Recall@100"] == 1.0
    assert metrics["NDCG@10"] == pytest.approx(0.10389383391183157)
    assert metrics["NDCG@20"] == pytest.approx(0.2707373790281412)
    assert metrics["NDCG@50"] == pytest.approx(0.401649231407207)
    assert metrics["NDCG@100"] == pytest.approx(0.401649231407207)
    assert metrics["QueryRetrievalCoverage@100"] == 1.0


def test_known_user_report_writer_persists_structured_reproducible_results(
    tmp_path,
) -> None:
    result = run_known_user_benchmark(_known_user_events())
    path = write_known_user_benchmark_report(result, tmp_path / "nested" / "report.json")

    saved_report = json.loads(path.read_text(encoding="utf-8"))

    assert saved_report == result.to_dict()
    assert saved_report["protocol"] == "lightgcn-ngcf-known-user"
    assert saved_report["baseline"] == "popularity"
    assert saved_report["counts"]["evaluation_query_count"] == 10
    assert saved_report["exclusions"]["test_gold_movie_count_missing_from_training_catalog"] == 19
    assert saved_report["metrics"]["Recall@100"] == 1.0
    assert saved_report["training_candidate_catalog_movie_ids"] == sorted(
        result.candidate_catalog
    )
    assert saved_report["evaluation"]["sampled_negatives"] is False
    assert saved_report["evaluation"]["synthetic_gold_insertion"] is False


def test_known_user_run_has_no_temporal_cohort_cap() -> None:
    events = tuple(
        RatingEvent.from_movielens(
            subject_id=subject_id,
            movie_id=movie_id,
            rating=4.0,
            event_time=subject_id * 100 + movie_id,
        )
        for subject_id in range(1, 5_002)
        for movie_id in range(1, 11)
    )

    result = run_known_user_benchmark(events)
    report = result.to_dict()

    assert len(result.split_by_subject) == 5_001
    assert report["counts"]["eligible_subject_count"] == 5_001
    assert report["counts"]["evaluation_query_count"] == 5_001
    assert report["reproducibility"]["subject_cohort_cap"] is None
