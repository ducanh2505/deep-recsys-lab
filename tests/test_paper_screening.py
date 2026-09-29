from __future__ import annotations

import gzip
import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from deep_recsys_lifecycle.models import RatingEvent
from deep_recsys_lifecycle.paper_fit_cache import FitCache
from deep_recsys_lifecycle.paper_pool_cache import PoolCache
from deep_recsys_lifecycle.paper_screening import (
    PaperScreenWorkspace,
    ScreenAxis,
    baseline_configuration,
    default_screen_runner,
    fixture_test_access,
    initial_axes,
)


def _rows(dropout: float, mode: str) -> dict[str, dict[str, float]]:
    if mode == "history_only":
        return {
            "11": {"Recall@100": 0.5, "NDCG@20": 0.2},
            "12": {"Recall@100": 0.5, "NDCG@20": 0.2},
        }
    if dropout == 0.2:
        return {
            "1": {"Recall@100": 0.7, "NDCG@20": 0.3},
            "2": {"Recall@100": 0.4, "NDCG@20": 0.3},
        }
    if dropout == 0.5:
        return {
            "1": {"Recall@100": 0.5, "NDCG@20": 0.1},
            "2": {"Recall@100": 0.5, "NDCG@20": 0.1},
        }
    return {
        "1": {"Recall@100": 0.5, "NDCG@20": 0.2},
        "2": {"Recall@100": 0.5, "NDCG@20": 0.2},
    }


def _runner(_events: object, mode: str, configuration: dict[str, Any]) -> dict[str, object]:
    dropout = float(configuration["multivae"]["dropout"]) if mode == "known_user" else 0.0
    rows = _rows(dropout, mode)
    metrics = {
        "Recall@100": sum(row["Recall@100"] for row in rows.values()) / len(rows),
        "NDCG@20": sum(row["NDCG@20"] for row in rows.values()) / len(rows),
        "query_count": len(rows),
    }
    source = {
        "name": "Rating Events",
        "dataset_sha256": getattr(_events, "dataset_checksum", None),
        "input_event_fingerprint_sha256": "input-fixed",
    }
    common = {
        "source": source,
        "package_source_sha256": "fixture-code-v1",
        "training_catalog_movie_ids": [1, 2, 3, 4],
        "runtime": {"total_wall_seconds": 0.1, "peak_process_rss_bytes": 1_000},
        "run_identity": {"git_revision": "fixture", "git_dirty": False},
    }
    if mode == "known_user":
        return {
            **common,
            "test_status": "sealed",
            "configuration": configuration,
            "split_membership_sha256": {
                "train": "train-fixed",
                "validation": "validation-fixed",
                "test": "test-sealed",
            },
            "validation_gold_sets_by_subject": {"1": [3], "2": [4]},
            "validation_subject_metrics": rows,
            "training": {
                "outer_fit_event_ids_sha256": "outer-fixed",
                "inner_folds": [{"fit_event_ids_sha256": "inner-fixed"}],
            },
            "metrics": metrics,
            "evaluation": {
                "source_pools": {"popularity": {"Recall@100": 0.2}},
                "oracle_union": {"OracleUnionRecall": 0.7},
                "gold_loss": {"largest_addressable_category": "outside_retriever_pools"},
                "inference_latency": {"p95_ms": 1.0},
            },
        }
    return {
        **common,
        "cohort_subject_ids": {"train": [1, 2, 3], "validation": [11, 12], "test": [21]},
        "held_out_subject_splits": {
            "validation": {
                "11": {"query_history_movie_ids": [1], "eligible_gold_movie_ids": [3]},
                "12": {"query_history_movie_ids": [2], "eligible_gold_movie_ids": [4]},
            },
            "test": {"status": "sealed", "membership_sha256": "history-test-sealed"},
        },
        "reproducibility": {
            "split_seed": configuration["seed"],
            "validation_subject_count": configuration["validation_subject_count"],
            "test_subject_count": configuration["test_subject_count"],
            "inner_held_out_fraction": configuration["inner_held_out_fraction"],
            "candidate_pool_limit_per_source": configuration["candidate_pool_limit"],
            "fusion_negative_rows_per_query": configuration["fusion_negative_rows_per_query"],
            "multivae_config": configuration["multivae_config"],
            "multivae_device_preference": configuration["multivae_device_preference"],
            "capture_evidence": configuration["capture_evidence"],
            "test_evaluated": False,
            "implementation_sha256": "fixture-code-v1",
        },
        "fusion_training": {
            "outer_fit_event_ids_sha256": "history-outer-fixed",
            "inner_fit_event_ids_sha256": "history-inner-fixed",
            "inner_pseudo_held_out_subject_ids_sha256": "history-pseudo-fixed",
        },
        "metrics": {"validation": metrics},
        "diagnostics": {
            "validation": {
                "sources": {"popularity": {"Recall@100": 0.2}},
                "oracle_union": {"gold_retrieval_fraction": 0.7},
                "gold_loss": {"largest_addressable_category": "outside_retriever_pools"},
                "inference_latency": {"p95_ms": 1.0},
                "per_subject_final_lhf": rows,
            }
        },
    }


def _screen(root: Path, sealed_test_gold: list[int]) -> tuple[PaperScreenWorkspace, dict[str, Any]]:
    axis = ScreenAxis(
        "multivae_dropout",
        ("multivae.dropout",),
        (0.2, 0.5),
        "Outside source pools is the largest addressable #47 loss.",
    )
    workspace = PaperScreenWorkspace.create(
        root,
        source_sha256=None,
        axes_by_mode={"known_user": (axis,), "history_only": ()},
    )
    # The test holder is external to the validation runner. Changing its labels cannot
    # enter the selected configuration or any screening run record.
    (root / "sealed-test.json").write_text(json.dumps(sealed_test_gold), encoding="utf-8")
    known_baseline = workspace.register_baseline("known_user")
    history_baseline = workspace.register_baseline("history_only")
    workspace.run_registered(known_baseline, (), runner=_runner)
    workspace.run_registered(history_baseline, (), runner=_runner)
    for index, dropout in enumerate((0.2, 0.5)):
        config = deepcopy(baseline_configuration("known_user"))
        config["multivae"]["dropout"] = dropout
        workspace.register_variant(
            run_id=f"known-dropout-{index}",
            mode="known_user",
            axis_name="multivae_dropout",
            alternative_index=index,
            reference_run_id=known_baseline,
            configuration=config,
        )
        record = workspace.run_registered(f"known-dropout-{index}", (), runner=_runner)
        assert (
            record["validation_cohort_sha256"]
            == workspace.completed_runs("known_user")[known_baseline]["validation_cohort_sha256"]
        )
    return workspace, workspace.select_and_freeze(bootstrap_draws=200)


def test_initial_plan_registers_exact_first_screen_and_budget_first() -> None:
    assert len(initial_axes("known_user")) == 15
    assert len(initial_axes("history_only")) == 11
    assert initial_axes("known_user")[0].name.endswith("budget_checkpoint")
    assert initial_axes("history_only")[0].name == "multivae_budget_checkpoint"
    assert next(
        axis for axis in initial_axes("history_only") if axis.name == "multivae_dropout"
    ).paths == ("multivae_config.dropout",)


def test_public_screen_matches_cohort_breaks_recall_tie_and_ignores_test_labels(
    tmp_path: Path,
) -> None:
    first, first_freeze = _screen(tmp_path / "first", [7, 8])
    second, second_freeze = _screen(tmp_path / "second", [8, 9])
    assert first_freeze["modes"]["known_user"]["selected_run_id"] == "known-dropout-0"
    assert first_freeze["modes"]["history_only"]["selected_run_id"] == "history_only-baseline"
    for mode in ("known_user", "history_only"):
        for key in (
            "selected_run_id", "selected_configuration_sha256", "validation_cohort_sha256",
            "sealed_test_membership_sha256", "selection",
        ):
            assert first_freeze["modes"][mode][key] == second_freeze["modes"][mode][key]
    assert first.require_frozen_selection() == first_freeze
    assert second.require_frozen_selection() == second_freeze
    for record in first.completed_runs("known_user").values():
        assert "test_metrics" not in record
        assert "sealed-test" not in record
    with pytest.raises(ValueError, match="frozen"):
        first.register_additional(
            run_id="after-freeze",
            mode="known_user",
            reference_run_id="known-dropout-0",
            configuration=baseline_configuration("known_user"),
            rationale="too late",
        )


def test_unplanned_factor_value_and_test_scoring_are_rejected(tmp_path: Path) -> None:
    axis = ScreenAxis("multivae_dropout", ("multivae.dropout",), (0.2, 0.5), "Measured loss")
    workspace = PaperScreenWorkspace.create(
        tmp_path,
        source_sha256=None,
        axes_by_mode={"known_user": (axis,), "history_only": ()},
    )
    baseline = workspace.register_baseline("known_user")
    workspace.run_registered(baseline, (), runner=_runner)
    config = deepcopy(baseline_configuration("known_user"))
    config["multivae"]["dropout"] = 0.3
    with pytest.raises(ValueError, match="preregistered alternative"):
        workspace.register_variant(
            run_id="not-planned",
            mode="known_user",
            axis_name="multivae_dropout",
            alternative_index=0,
            reference_run_id=baseline,
            configuration=config,
        )
    with pytest.raises(FileNotFoundError):
        workspace.require_frozen_selection()

    def leaky_runner(events: object, mode: str, configuration: dict[str, Any]) -> dict[str, object]:
        report = _runner(events, mode, configuration)
        report["test_metrics"] = {"Recall@100": 1.0}
        return report

    history = workspace.register_baseline("history_only")
    with pytest.raises(ValueError, match="test metrics"):
        workspace.run_registered(history, (), runner=leaky_runner)
    assert not workspace.completed_runs("history_only")


def test_real_known_user_pipeline_refits_fusion_on_the_same_validation_cohort(
    tmp_path: Path,
) -> None:
    events = tuple(
        RatingEvent.from_movielens(
            subject_id=subject_id,
            movie_id=movie_id,
            rating=5.0,
            event_time=subject_id * 1_000 + movie_id,
        )
        for subject_id in (2, 28, 29, 34, 42, 44, 46, 53, 70, 71)
        for movie_id in (*range(1, 41), 99)
    )
    baseline_config = baseline_configuration("known_user")
    baseline_config["multivae"]["hidden_dim"] = 8
    baseline_config["multivae"]["latent_dim"] = 4
    baseline_config["lightgcn"]["embedding_dim"] = 8
    axis = ScreenAxis(
        "multivae_dropout",
        ("multivae.dropout",),
        (0.2, 0.5),
        "Train-only representation alternative",
    )
    screen = PaperScreenWorkspace.create(
        tmp_path,
        source_sha256=None,
        axes_by_mode={"known_user": (axis,), "history_only": ()},
        baseline_configs={
            "known_user": baseline_config,
            "history_only": baseline_configuration("history_only"),
        },
    )
    runner = partial(
        default_screen_runner,
        fit_cache=FitCache(tmp_path / "fit-cache", max_bytes=100_000_000, min_free_bytes=0),
        pool_cache=PoolCache(
            tmp_path / "pool-cache",
            max_bytes=100_000_000,
            min_free_bytes=0,
            shard_queries=4,
        ),
    )
    baseline_id = screen.register_baseline("known_user")
    baseline = screen.run_registered(baseline_id, events, runner=runner)
    variant_config = screen.planned_configuration(
        mode="known_user",
        axis_name="multivae_dropout",
        alternative_index=0,
        reference_run_id=baseline_id,
    )
    screen.register_variant(
        run_id="known-dropout-0",
        mode="known_user",
        axis_name="multivae_dropout",
        alternative_index=0,
        reference_run_id=baseline_id,
        configuration=variant_config,
    )
    variant = screen.run_registered("known-dropout-0", events, runner=runner)

    assert variant["validation_cohort_sha256"] == baseline["validation_cohort_sha256"]
    assert variant["configuration"]["multivae"]["dropout"] == 0.2
    assert set(screen.validation_subjects("known-dropout-0")) == set(
        screen.validation_subjects(baseline_id)
    )
    assert variant["metrics"]["query_count"] == baseline["metrics"]["query_count"]
    assert variant["source_pools"].keys() == baseline["source_pools"].keys()
    assert variant["run_identity"] == baseline["run_identity"]
    assert "test" not in variant
    for partition in variant["screen_cache"]["pool_partitions"]:
        assert partition["sources"]["popularity"]["cache_hit"]
        assert partition["sources"]["itemknn"]["cache_hit"]
        assert partition["sources"]["lightgcn"]["cache_hit"]
        assert not partition["sources"]["multivae"]["cache_hit"]


def test_frozen_receipt_checks_mode_cohort_and_both_configuration_hashes(
    tmp_path: Path,
) -> None:
    screen, freeze = _screen(tmp_path, [7, 8])
    known = freeze["modes"]["known_user"]
    receipt = screen.test_access("known_user")
    checks = {
        "mode": "known_user",
        "source_sha256": None,
        "configuration_sha256": known["selected_configuration_sha256"],
        "validation_cohort_sha256": known["validation_cohort_sha256"],
        "test_membership_sha256": known["sealed_test_membership_sha256"],
    }
    receipt.require(**checks)
    receipt.require(**{**checks, "configuration_sha256": known["baseline_configuration_sha256"]})
    with pytest.raises(ValueError, match="configuration"):
        receipt.require(**{**checks, "configuration_sha256": "wrong"})
    with pytest.raises(ValueError, match="cohort"):
        receipt.require(**{**checks, "test_membership_sha256": "wrong"})
    with pytest.raises(ValueError, match="mode"):
        receipt.require(**{**checks, "mode": "history_only"})
    with pytest.raises(ValueError, match="fixture receipt"):
        fixture_test_access("known_user").require(
            **{**checks, "source_sha256": "MovieLens-checksum"}
        )


def test_changed_runner_code_and_effective_config_are_rejected(tmp_path: Path) -> None:
    axis = ScreenAxis("multivae_dropout", ("multivae.dropout",), (0.2, 0.5), "Measured loss")
    screen = PaperScreenWorkspace.create(
        tmp_path,
        source_sha256=None,
        axes_by_mode={"known_user": (axis,), "history_only": ()},
    )
    baseline = screen.register_baseline("known_user")
    screen.run_registered(baseline, (), runner=_runner)
    config = screen.planned_configuration(
        mode="known_user",
        axis_name="multivae_dropout",
        alternative_index=0,
        reference_run_id=baseline,
    )
    screen.register_variant(
        run_id="variant",
        mode="known_user",
        axis_name="multivae_dropout",
        alternative_index=0,
        reference_run_id=baseline,
        configuration=config,
    )

    def changed_code(events: object, mode: str, configuration: dict[str, Any]) -> dict[str, object]:
        report = _runner(events, mode, configuration)
        report["package_source_sha256"] = "fixture-code-v2"
        return report

    with pytest.raises(ValueError, match="implementation changed"):
        screen.run_registered("variant", (), runner=changed_code)

    def ignored_factor(
        events: object, mode: str, configuration: dict[str, Any]
    ) -> dict[str, object]:
        report = _runner(events, mode, configuration)
        report["configuration"] = baseline_configuration("known_user")
        return report

    with pytest.raises(ValueError, match="effective configuration"):
        screen.run_registered("variant", (), runner=ignored_factor)
    assert not screen.completed_runs("known_user").get("variant")


def test_interrupted_subject_write_resumes_only_when_evidence_matches(tmp_path: Path) -> None:
    screen = PaperScreenWorkspace.create(
        tmp_path, source_sha256=None,
        axes_by_mode={"known_user": (), "history_only": ()},
    )
    baseline = screen.register_baseline("known_user")
    orphan = tmp_path / "subjects" / f"{baseline}.json.gz"
    orphan.parent.mkdir(parents=True)
    with gzip.open(orphan, "wt", encoding="utf-8") as handle:
        json.dump(_rows(0.0, "known_user"), handle)
    record = screen.run_registered(baseline, (), runner=_runner)
    assert record["per_subject_path"] == f"subjects/{baseline}.json.gz"

    other = PaperScreenWorkspace.create(
        tmp_path / "other", source_sha256=None,
        axes_by_mode={"known_user": (), "history_only": ()},
    )
    other_baseline = other.register_baseline("known_user")
    corrupted = other.root / "subjects" / f"{other_baseline}.json.gz"
    corrupted.parent.mkdir(parents=True)
    with gzip.open(corrupted, "wt", encoding="utf-8") as handle:
        json.dump({"wrong": {}}, handle)
    with pytest.raises(ValueError, match="interrupted run"):
        other.run_registered(other_baseline, (), runner=_runner)


def test_axis_alternatives_must_share_the_same_reference(tmp_path: Path) -> None:
    axis = ScreenAxis("multivae_dropout", ("multivae.dropout",), (0.2, 0.5), "Measured loss")
    screen = PaperScreenWorkspace.create(
        tmp_path, source_sha256=None,
        axes_by_mode={"known_user": (axis,), "history_only": ()},
    )
    baseline = screen.register_baseline("known_user")
    screen.run_registered(baseline, (), runner=_runner)
    first_config = screen.planned_configuration(
        mode="known_user", axis_name="multivae_dropout",
        alternative_index=0, reference_run_id=baseline,
    )
    screen.register_variant(
        run_id="first", mode="known_user", axis_name="multivae_dropout",
        alternative_index=0, reference_run_id=baseline, configuration=first_config,
    )
    screen.run_registered("first", (), runner=_runner)
    second_config = screen.planned_configuration(
        mode="known_user", axis_name="multivae_dropout",
        alternative_index=1, reference_run_id="first",
    )
    with pytest.raises(ValueError, match="same completed reference"):
        screen.register_variant(
            run_id="second", mode="known_user", axis_name="multivae_dropout",
            alternative_index=1, reference_run_id="first", configuration=second_config,
        )


def test_archived_baseline_requires_exact_subject_and_diagnostic_replay(tmp_path: Path) -> None:
    class FixtureSource(list[RatingEvent]):
        dataset_checksum = "fixture-sha256"

    source = FixtureSource()
    screen = PaperScreenWorkspace.create(
        tmp_path / "screen", source_sha256=source.dataset_checksum,
        axes_by_mode={"known_user": (), "history_only": ()},
    )
    baseline = screen.register_baseline("known_user")
    reference = _runner(source, "known_user", baseline_configuration("known_user"))
    archived = tmp_path / "archived-report.json"

    changed_union = deepcopy(reference)
    changed_union["evaluation"]["oracle_union"]["OracleUnionRecall"] = 0.9
    archived.write_text(json.dumps(changed_union), encoding="utf-8")
    with pytest.raises(ValueError, match="archived"):
        screen.run_registered(baseline, source, runner=_runner, reference_report_path=archived)
    assert not (screen.root / "cohorts" / "known_user.json").exists()

    changed_subject = deepcopy(reference)
    changed_subject["validation_subject_metrics"]["1"]["Recall@100"] = 0.9
    archived.write_text(json.dumps(changed_subject), encoding="utf-8")
    with pytest.raises(ValueError, match="archived"):
        screen.run_registered(baseline, source, runner=_runner, reference_report_path=archived)
    assert not (screen.root / "cohorts" / "known_user.json").exists()

    archived.write_text(json.dumps(reference), encoding="utf-8")
    accepted = screen.run_registered(
        baseline, source, runner=_runner, reference_report_path=archived
    )
    assert accepted["metrics"] == reference["metrics"]
