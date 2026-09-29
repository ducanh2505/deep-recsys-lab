from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deep_recsys_lifecycle import cli
from deep_recsys_lifecycle.fusion import FusionFeatureBuilder
from deep_recsys_lifecycle.history_only_benchmark import (
    HistoryOnlyBenchmarkConfig,
    HistoryOnlyFusionQuery,
    HistoryOnlyQuery,
    build_history_only_lhf_training_rows,
    run_history_only_benchmark,
    write_history_only_benchmark_report,
)
from deep_recsys_lifecycle.models import Candidate, RatingEvent
from deep_recsys_lifecycle.paper_fit_cache import FitCache
from deep_recsys_lifecycle.paper_pool_cache import PoolCache
from deep_recsys_lifecycle.paper_screening import fixture_test_access

_SMALL_CONFIG = HistoryOnlyBenchmarkConfig(
    seed=42, validation_subject_count=2, test_subject_count=2
)


def test_history_only_validation_cache_replays_exact_metrics(tmp_path: Path) -> None:
    fit = FitCache(tmp_path / "fit", max_bytes=100_000_000, min_free_bytes=0)
    pools = PoolCache(
        tmp_path / "pools", max_bytes=100_000_000, min_free_bytes=0, shard_queries=4
    )
    first = run_history_only_benchmark(
        _public_events(), config=_SMALL_CONFIG, fit_cache=fit, pool_cache=pools
    ).to_dict()
    replay = run_history_only_benchmark(
        _public_events(), config=_SMALL_CONFIG, fit_cache=fit, pool_cache=pools
    ).to_dict()
    assert first["metrics"] == replay["metrics"]
    assert first["diagnostics"] == replay["diagnostics"]
    assert replay["screen_cache"]["fit"]["hits"] == 6
    assert all(
        info["cache_hit"]
        for partition in replay["screen_cache"]["pool_partitions"]
        for info in partition["sources"].values()
    )
    assert replay["reproducibility"]["test_evaluated"] is False


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


def _trained_events() -> tuple[RatingEvent, ...]:
    events = [
        RatingEvent.from_movielens(
            subject_id,
            ((subject_id * 11 + offset) % 150) + 1,
            5.0,
            subject_id * 1_000 + offset,
        )
        for subject_id in range(1, 17)
        for offset in range(60)
    ]
    events.extend(
        RatingEvent.from_movielens(subject_id, 999, 5.0, subject_id * 1_000 + 999)
        for subject_id in (9, 10, 14, 15)
    )
    return tuple(events)


def test_history_only_public_run_has_disjoint_subjects_and_complete_fold_in() -> None:
    events = _public_events()
    result = run_history_only_benchmark(
        events,
        config=_SMALL_CONFIG,
        evaluate_test=True,
        test_access=fixture_test_access("history_only"),
    )
    replay = run_history_only_benchmark(
        reversed(events),
        config=_SMALL_CONFIG,
        evaluate_test=True,
        test_access=fixture_test_access("history_only"),
    )
    report = result.to_dict()

    assert result.cohort_subject_ids == {
        "train": (4, 5, 7),
        "validation": (2, 6),
        "test": (1, 3),
    }
    assert result.cohort_subject_ids == replay.cohort_subject_ids
    assert result.split_by_subject == replay.split_by_subject
    stable_report = deepcopy(report)
    stable_replay = deepcopy(replay.to_dict())
    for value in (stable_report, stable_replay):
        value.pop("runtime")
        for cohort in value["diagnostics"].values():
            cohort.pop("inference_latency")
        for metadata in value["fusion_training"]["retriever_training_metadata"].values():
            metadata.pop("duration_seconds", None)
    assert stable_report == stable_replay
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


def test_history_only_public_metrics_use_final_lhf_order_and_report_source_diagnostics() -> None:
    report = run_history_only_benchmark(
        _public_events(),
        config=_SMALL_CONFIG,
        evaluate_test=True,
        test_access=fixture_test_access("history_only"),
    ).to_dict()
    validation = report["diagnostics"]["validation"]["sources"]["popularity"]
    test = report["diagnostics"]["test"]["sources"]["popularity"]

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
    assert report["metrics"]["test"] == report["diagnostics"]["test"]["final_lhf"]
    assert report["metrics"]["test"]["Recall@10"] == pytest.approx(0.35)
    assert report["metrics"]["test"]["Recall@10"] != test["Recall@10"]
    assert report["diagnostics"]["test"]["oracle_union"]["retrieved_gold_movie_count"] == 13


def test_history_only_does_not_insert_gold_into_top_100() -> None:
    result = run_history_only_benchmark(
        _public_events(late_test_gold=True),
        config=replace(_SMALL_CONFIG, candidate_pool_limit=20, capture_evidence=True),
        evaluate_test=True,
        test_access=fixture_test_access("history_only"),
    )

    assert result.gold_sets[3] == (111,)
    pools = result.to_dict()["candidate_pools"]["test"]["3"]
    assert all(111 not in movie_ids for movie_ids in pools["sources"].values())
    assert 111 not in pools["oracle_union_movie_ids"]
    assert 111 not in pools["final_lhf_top100_movie_ids"]
    assert (
        result.to_dict()["diagnostics"]["test"]["oracle_union"]["retrieved_gold_movie_count"] < 13
    )


def test_history_only_fusion_rows_label_all_natural_gold_without_insertion() -> None:
    builder = FusionFeatureBuilder.from_serving(
        snapshot_fingerprint="inner-fit",
        retriever_bank=("popularity", "itemknn", "multivae"),
        catalog=(1, 2, 3, 4, 5, 6),
        item_popularity={1: 5, 2: 3, 3: 1},
    )
    query = HistoryOnlyFusionQuery(pool_key="inner-000000", history=(), gold_movie_ids=(2, 3, 4))
    pools = {
        query.pool_key: {
            "popularity": (
                Candidate(movie_id=1, score=5, rank=1),
                Candidate(movie_id=2, score=3, rank=2),
            ),
            "itemknn": (
                Candidate(movie_id=3, score=0.8, rank=1),
                Candidate(movie_id=5, score=0.3, rank=2),
            ),
            "multivae": (Candidate(movie_id=6, score=0.6, rank=1),),
        }
    }

    rows = build_history_only_lhf_training_rows(
        (query,), pools, builder, max_negative_rows_per_query=1, seed=42
    )

    assert {row.movie_id for row in rows if row.label} == {2, 3}
    assert 4 not in {row.movie_id for row in rows}
    assert sum(row.label == 0 for row in rows) == 1
    assert all(row.query_key == "inner-000000" for row in rows)
    assert "lightgcn_present" not in builder.feature_names
    assert "subject_id" not in builder.feature_names


def test_history_only_public_run_trains_disjoint_fusion_and_reports_top100() -> None:
    events = _trained_events()
    result = run_history_only_benchmark(
        events,
        config=replace(_SMALL_CONFIG, capture_evidence=True),
        evaluate_test=True,
        test_access=fixture_test_access("history_only"),
    )
    report = result.to_dict()
    cohorts = result.cohort_subject_ids
    fusion_report = report["fusion_training"]

    assert result.fusion.training_status == "trained"
    assert fusion_report["training_status"] == "trained"
    assert fusion_report["retriever_bank"] == ["popularity", "itemknn", "multivae"]
    assert fusion_report["training_metadata"]["training_boundary"] == (
        "paper_inner_pseudo_held_out_training_subjects"
    )
    assert fusion_report["training_metadata"]["inner_fold_seed"] == 42
    assert set(result.inner_fit_subject_ids).isdisjoint(result.inner_pseudo_held_out_subject_ids)
    assert set(result.inner_fit_subject_ids).isdisjoint(cohorts["validation"])
    assert set(result.inner_fit_subject_ids).isdisjoint(cohorts["test"])
    assert set(result.inner_pseudo_held_out_subject_ids) <= set(cohorts["train"])
    assert fusion_report["inner_fit_subject_count"] == len(result.inner_fit_subject_ids)
    assert fusion_report["outer_fit_subject_count"] == len(cohorts["train"])
    assert fusion_report["training_metadata"]["inner_fit_event_count"] == (
        60 * len(result.inner_fit_subject_ids)
    )
    for subject_ids, fingerprint_key in (
        (cohorts["train"], "outer_fit_event_ids_sha256"),
        (result.inner_fit_subject_ids, "inner_fit_event_ids_sha256"),
    ):
        expected_ids = (
            event.event_id
            for event in sorted(events, key=lambda event: (event.subject_id, event.movie_id))
            if event.subject_id in subject_ids
        )
        assert (
            fusion_report[fingerprint_key]
            == sha256("".join(f"{event_id}\n" for event_id in expected_ids).encode()).hexdigest()
        )
    assert fusion_report["positive_training_row_count"] >= 2
    assert fusion_report["negative_training_row_count"] >= 2
    assert len(result.fusion_training_rows) == fusion_report["training_row_count"]
    assert all(str(row.query_key).startswith("inner-") for row in result.fusion_training_rows)
    assert 999 not in result.training_catalog
    assert 999 not in result.outer_feature_builder.item_popularity
    assert result.inner_feature_builder is not None
    assert 999 not in result.inner_feature_builder.item_popularity
    assert all(row.movie_id != 999 for row in result.fusion_training_rows)
    assert "lightgcn" not in " ".join(result.outer_feature_builder.feature_names)
    assert "subject_id" not in result.outer_feature_builder.feature_names

    for cohort in ("validation", "test"):
        assert report["metrics"][cohort] == report["diagnostics"][cohort]["final_lhf"]
        assert set(report["diagnostics"][cohort]["sources"]) == {
            "popularity",
            "itemknn",
            "multivae",
        }
        assert set(report["diagnostics"][cohort]["source_full_pools"]) == {
            "popularity",
            "itemknn",
            "multivae",
        }
        for subject_id, pools in report["candidate_pools"][cohort].items():
            assert set(pools["sources"]) == {"popularity", "itemknn", "multivae"}
            assert len(pools["final_lhf_top100_movie_ids"]) == 100
            assert len(set(pools["final_lhf_top100_movie_ids"])) == 100
            assert set(pools["final_lhf_top100_movie_ids"]) <= set(pools["oracle_union_movie_ids"])
            assert 999 not in pools["final_lhf_top100_movie_ids"]
            assert subject_id in report["held_out_subject_splits"][cohort]


def test_history_only_seals_test_scores_until_explicit_final_evaluation(tmp_path) -> None:
    events = _public_events()
    validation = run_history_only_benchmark(events, config=_SMALL_CONFIG)
    path = write_history_only_benchmark_report(validation, tmp_path / "nested" / "report.json")
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved == validation.to_dict()
    assert saved["protocol"] == "mult-vae-history-only"
    assert saved["baseline"] == "history-only-lhf"
    assert set(saved["metrics"]) == {"validation"}
    assert saved["held_out_subject_splits"]["test"]["status"] == "sealed"
    assert saved["held_out_subject_splits"]["test"]["membership_sha256"]
    assert saved["counts"]["test"] == {"subject_count": 2, "status": "sealed"}
    assert saved["exclusions"]["test"] == {"status": "sealed"}
    assert set(saved["diagnostics"]) == {"validation"}
    assert saved["candidate_pools"] == {"status": "omitted"}
    assert saved["reproducibility"]["test_evaluated"] is False
    assert saved["training_catalog_movie_ids"] == list(range(1, 131))
    assert saved["cohort_subject_ids"]["test"] == [1, 3]
    assert set(validation.split_by_subject) == {2, 6}
    assert set(validation.gold_sets) == {2, 6}
    assert set(validation.queries_by_subject) == {2, 6}
    assert validation.fusion_training_rows == ()

    evidence = run_history_only_benchmark(
        events, config=replace(_SMALL_CONFIG, capture_evidence=True)
    )
    assert set(evidence.to_dict()["candidate_pools"]) == {"validation"}
    assert set(evidence.split_by_subject) == {2, 6}

    final = run_history_only_benchmark(
        events,
        config=_SMALL_CONFIG,
        evaluate_test=True,
        test_access=fixture_test_access("history_only"),
    )
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
