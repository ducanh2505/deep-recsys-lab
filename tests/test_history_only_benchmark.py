from __future__ import annotations

import json
import math
from dataclasses import fields

import pytest
from typer.testing import CliRunner

from deep_recsys_lifecycle import cli
from deep_recsys_lifecycle.history_only_benchmark import (
    HistoryOnlyBenchmarkConfig,
    HistoryOnlyQuery,
    run_history_only_benchmark,
    write_history_only_benchmark_report,
)
from deep_recsys_lifecycle.models import RatingEvent

_SMALL_CONFIG = HistoryOnlyBenchmarkConfig(
    seed=42, validation_subject_count=2, test_subject_count=2
)


def _public_events(*, late_test_gold: bool = False) -> tuple[RatingEvent, ...]:
    profiles = {
        1: tuple(range(1, 61)),
        2: (1, 2, 3, 4, 5, 996),
        3: (110, 111, 112, 113, 114) if late_test_gold else (45, 110, 111, 112, 113),
        4: tuple(range(1, 131)),
        5: tuple(range(1, 131)),
        6: (1, 2, 3, 4, 5, 999),
        7: tuple(range(1, 131)),
        8: (200, 201, 202, 203),
    }
    events = [
        RatingEvent.from_movielens(subject_id, movie_id, 5.0, subject_id * 1_000 + movie_id)
        for subject_id, movie_ids in profiles.items()
        for movie_id in movie_ids
    ]
    events.append(RatingEvent.from_movielens(1, 1, 5.0, 10_001))
    events.append(RatingEvent.from_movielens(8, 204, 3.0, 8_204))
    return tuple(events)


def test_history_only_public_run_has_disjoint_subjects_and_complete_fold_in() -> None:
    events = _public_events()
    result = run_history_only_benchmark(events, config=_SMALL_CONFIG, evaluate_test=True)
    replay = run_history_only_benchmark(reversed(events), config=_SMALL_CONFIG, evaluate_test=True)
    report = result.to_dict()

    assert result.cohort_subject_ids == {
        "train": (4, 5, 7),
        "validation": (2, 6),
        "test": (1, 3),
    }
    assert result.cohort_subject_ids == replay.cohort_subject_ids
    assert result.split_by_subject == replay.split_by_subject
    assert report == replay.to_dict()
    assert report["cohort_subject_ids"] == {
        "train": [4, 5, 7],
        "validation": [2, 6],
        "test": [1, 3],
    }
    for subject_id, split in result.split_by_subject.items():
        expected_profile_size = 60 if subject_id == 1 else 6 if subject_id in (2, 6) else 5
        assert len(split.held_out_movie_ids) == math.ceil(expected_profile_size * 0.20)
        assert len(split.fold_in_movie_ids) + len(split.held_out_movie_ids) == expected_profile_size
        assert not set(split.fold_in_movie_ids) & set(split.held_out_movie_ids)
        assert result.gold_sets[subject_id] == split.gold_movie_ids
    assert result.split_by_subject[1].held_out_movie_ids == (
        9,
        13,
        25,
        26,
        29,
        32,
        39,
        44,
        46,
        55,
        56,
        60,
    )

    counts = report["counts"]
    assert counts["rating_event_count"] == 473
    assert counts["positive_rating_event_count"] == 472
    assert counts["duplicate_positive_interaction_count"] == 1
    assert counts["eligible_subject_count"] == 7
    assert counts["training_subject_count"] == 3
    assert counts["training_interaction_count"] == 390
    assert counts["validation"]["held_out_interaction_count"] == 4
    assert counts["test"]["held_out_interaction_count"] == 13
    assert report["exclusions"]["subject_count_removed_with_fewer_than_five_positives"] == 1
    assert report["exclusions"]["positive_interaction_count_removed_with_short_subjects"] == 4
    assert report["source"]["input_event_fingerprint_sha256"]
    assert report["reproducibility"]["split_seed"] == 42
    assert report["reproducibility"]["implementation_sha256"]
    assert report["reproducibility"]["validation_subject_count"] == 2


def test_history_only_public_run_records_catalog_exclusions_and_no_identity_in_query() -> None:
    result = run_history_only_benchmark(_public_events(), config=_SMALL_CONFIG)
    report = result.to_dict()

    assert result.training_catalog == frozenset(range(1, 131))
    assert result.split_by_subject[2].held_out_movie_ids == (1, 996)
    assert result.gold_sets[2] == (1,)
    assert result.split_by_subject[2].excluded_cold_gold_movie_ids == (996,)
    assert result.split_by_subject[6].excluded_cold_fold_in_movie_ids == (999,)
    assert result.queries_by_subject[6] == HistoryOnlyQuery(history_movie_ids=(3, 4, 5))
    assert [field.name for field in fields(HistoryOnlyQuery)] == ["history_movie_ids"]
    assert report["evaluation"]["query_features"] == ["fold_in_history_movie_ids"]
    assert report["evaluation"]["eligible_retrievers"] == [
        "popularity",
        "itemknn",
        "multivae",
    ]
    assert report["evaluation"]["sampled_negatives"] is False
    assert report["evaluation"]["synthetic_gold_insertion"] is False
    assert "cold held-out" in report["evaluation"]["cold_item_policy"]
    assert report["exclusions"]["validation"] == {
        "fold_in_movie_count_missing_from_training_catalog": 1,
        "held_out_gold_movie_count_missing_from_training_catalog": 1,
        "subjects_without_eligible_gold": 0,
        "subjects_without_eligible_fold_in_history": 0,
    }
    assert report["held_out_subject_splits"]["validation"]["2"]["excluded_cold_gold_movie_ids"] == [
        996
    ]
    assert report["held_out_subject_splits"]["validation"]["6"][
        "excluded_cold_fold_in_movie_ids"
    ] == [999]


def test_history_only_public_metrics_use_capped_recall_denominator_and_binary_ndcg() -> None:
    report = run_history_only_benchmark(
        _public_events(), config=_SMALL_CONFIG, evaluate_test=True
    ).to_dict()
    validation = report["metrics"]["validation"]
    test = report["metrics"]["test"]

    assert validation["query_count"] == 2
    for cutoff in (10, 20, 50, 100):
        assert validation[f"Recall@{cutoff}"] == 1.0
        assert validation[f"NDCG@{cutoff}"] == 1.0

    # Subject 1 has 12 Gold Movies and all its first ten recommendations are gold.
    # Subject 3 has one gold at rank 45. Both Subjects count equally in the macro mean.
    assert test["query_count"] == 2
    assert report["counts"]["test"]["held_out_interaction_count"] == 13
    assert test["Recall@10"] == 0.5
    assert test["Recall@20"] == 0.5
    assert test["Recall@50"] == 1.0
    assert test["Recall@100"] == 1.0
    assert test["NDCG@10"] == 0.5
    assert test["NDCG@20"] == 0.5
    expected_late_ndcg = 1.0 / math.log2(46)
    assert test["NDCG@50"] == pytest.approx((1 + expected_late_ndcg) / 2)
    assert test["NDCG@100"] == pytest.approx((1 + expected_late_ndcg) / 2)
    assert test["QueryRetrievalCoverage@100"] == 1.0


def test_history_only_does_not_insert_gold_into_top_100() -> None:
    result = run_history_only_benchmark(
        _public_events(late_test_gold=True), config=_SMALL_CONFIG, evaluate_test=True
    )

    assert result.gold_sets[3] == (111,)
    assert result.to_dict()["metrics"]["test"]["Recall@100"] == 0.5
    assert result.to_dict()["metrics"]["test"]["QueryRetrievalCoverage@100"] == 0.5


def test_history_only_seals_test_scores_until_explicit_final_evaluation(tmp_path) -> None:
    events = _public_events()
    validation = run_history_only_benchmark(events, config=_SMALL_CONFIG)
    path = write_history_only_benchmark_report(validation, tmp_path / "nested" / "report.json")
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved == validation.to_dict()
    assert saved["protocol"] == "mult-vae-history-only"
    assert saved["baseline"] == "popularity"
    assert set(saved["metrics"]) == {"validation"}
    assert saved["held_out_subject_splits"]["test"]["status"] == "sealed"
    assert saved["held_out_subject_splits"]["test"]["membership_sha256"]
    assert saved["counts"]["test"] == {"subject_count": 2, "status": "sealed"}
    assert saved["exclusions"]["test"] == {"status": "sealed"}
    assert saved["reproducibility"]["test_evaluated"] is False
    assert saved["training_catalog_movie_ids"] == list(range(1, 131))
    assert saved["cohort_subject_ids"]["test"] == [1, 3]
    assert set(validation.split_by_subject) == {2, 6}
    assert set(validation.gold_sets) == {2, 6}
    assert set(validation.queries_by_subject) == {2, 6}

    final = run_history_only_benchmark(events, config=_SMALL_CONFIG, evaluate_test=True)
    assert final.to_dict()["metrics"]["validation"] == saved["metrics"]["validation"]
    assert final.to_dict()["held_out_subject_splits"]["test"]["3"]["eligible_gold_movie_ids"] == [
        45
    ]
    assert final.to_dict()["reproducibility"]["test_evaluated"] is True


def test_history_only_cli_runs_public_benchmark_with_fixture_cohorts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "prepare_movielens_20m", lambda _cache: _public_events())
    output = tmp_path / "paper" / "report.json"

    invocation = CliRunner().invoke(
        cli.app,
        [
            "history-only-benchmark",
            "--output",
            str(output),
            "--validation-subject-count",
            "2",
            "--test-subject-count",
            "2",
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    assert json.loads(output.read_text(encoding="utf-8"))["cohort_subject_ids"]["test"] == [
        1,
        3,
    ]


def test_history_only_rejects_insufficient_subject_cohorts() -> None:
    assert HistoryOnlyBenchmarkConfig().validation_subject_count == 10_000
    assert HistoryOnlyBenchmarkConfig().test_subject_count == 10_000
    with pytest.raises(ValueError, match="at least one Subject"):
        HistoryOnlyBenchmarkConfig(validation_subject_count=0, test_subject_count=2)
    with pytest.raises(ValueError, match="nonempty training cohort"):
        run_history_only_benchmark(
            _public_events(),
            config=HistoryOnlyBenchmarkConfig(validation_subject_count=4, test_subject_count=3),
        )
