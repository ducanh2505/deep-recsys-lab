from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from deep_recsys_lifecycle.paper_screening import (
    PaperScreenWorkspace,
    ScreenAxis,
    baseline_configuration,
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
        "dataset_sha256": None,
        "input_event_fingerprint_sha256": "input-fixed",
    }
    common = {
        "source": source,
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
                "train": "train-fixed", "validation": "validation-fixed", "test": "test-sealed"
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
        "reproducibility": {"split_seed": configuration["seed"], "test_evaluated": False},
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
        "multivae_dropout", ("multivae.dropout",), (0.2, 0.5),
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
        assert record["validation_cohort_sha256"] == workspace.completed_runs("known_user")[
            known_baseline
        ]["validation_cohort_sha256"]
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
    assert first_freeze["modes"] == second_freeze["modes"]
    assert first.require_frozen_selection() == first_freeze
    assert second.require_frozen_selection() == second_freeze
    for record in first.completed_runs("known_user").values():
        assert "test_metrics" not in record
        assert "sealed-test" not in record
    with pytest.raises(ValueError, match="frozen"):
        first.register_additional(
            run_id="after-freeze", mode="known_user", reference_run_id="known-dropout-0",
            configuration=baseline_configuration("known_user"), rationale="too late",
        )


def test_unplanned_factor_value_and_test_scoring_are_rejected(tmp_path: Path) -> None:
    axis = ScreenAxis("multivae_dropout", ("multivae.dropout",), (0.2, 0.5), "Measured loss")
    workspace = PaperScreenWorkspace.create(
        tmp_path, source_sha256=None,
        axes_by_mode={"known_user": (axis,), "history_only": ()},
    )
    baseline = workspace.register_baseline("known_user")
    workspace.run_registered(baseline, (), runner=_runner)
    config = deepcopy(baseline_configuration("known_user"))
    config["multivae"]["dropout"] = 0.3
    with pytest.raises(ValueError, match="preregistered alternative"):
        workspace.register_variant(
            run_id="not-planned", mode="known_user", axis_name="multivae_dropout",
            alternative_index=0, reference_run_id=baseline, configuration=config,
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
