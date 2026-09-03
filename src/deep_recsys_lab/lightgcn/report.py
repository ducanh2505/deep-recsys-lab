"""Verified aggregate-only report export for the research site."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

from .config import to_lightgcn_config
from .data import LightGCNData, stable_hash
from .experiment import environment_manifest

PAPER_RESULTS = {
    "citation": "He et al., LightGCN, SIGIR 2020 / arXiv:2002.02126",
    "datasets": ["Gowalla", "Yelp2018", "Amazon-Book"],
    "rows": [
        {
            "model": "NGCF",
            "gowalla": [0.1570, 0.1327],
            "yelp2018": [0.0579, 0.0477],
            "amazon_book": [0.0344, 0.0263],
        },
        {
            "model": "Mult-VAE",
            "gowalla": [0.1641, 0.1335],
            "yelp2018": [0.0584, 0.0450],
            "amazon_book": [0.0407, 0.0315],
        },
        {
            "model": "GRMF",
            "gowalla": [0.1477, 0.1205],
            "yelp2018": [0.0571, 0.0462],
            "amazon_book": [0.0354, 0.0270],
        },
        {
            "model": "GRMF-norm",
            "gowalla": [0.1557, 0.1261],
            "yelp2018": [0.0561, 0.0454],
            "amazon_book": [0.0352, 0.0269],
        },
        {
            "model": "LightGCN",
            "gowalla": [0.1830, 0.1554],
            "yelp2018": [0.0649, 0.0530],
            "amazon_book": [0.0411, 0.0315],
        },
    ],
    "metric_order": ["Recall@20", "NDCG@20"],
}


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        cast(dict[str, Any], json.loads(line)) for line in path.read_text().splitlines() if line
    ]


def export_verified_report(run_dir: Path, data: LightGCNData, output_path: Path) -> dict[str, Any]:
    """Validate the run and emit only aggregate, deployment-safe information."""

    status = _read_json(run_dir / "run_status.json")
    if status.get("status") != "verified" or status.get("phase") != "complete":
        raise ValueError("LightGCN run is not verified and complete")
    if status.get("dataset_hash") != data.dataset_hash:
        raise ValueError("run and prepared-data hashes do not match")
    selection = _read_json(run_dir / "selection.json")
    test_metrics = _read_json(run_dir / "test_metrics.json")
    access = _read_json(run_dir / "test_access.json")
    resolved_config = _read_json(run_dir / "resolved_config.json")
    provenance = _read_json(run_dir / "provenance.json")
    evaluation_k = int(resolved_config["evaluation"]["k"])
    if status.get("config_hash") != stable_hash(resolved_config):
        raise ValueError("run status and resolved config hashes do not match")
    if status.get("test_metrics") != test_metrics:
        raise ValueError("run status and test metric artifact do not match")
    if provenance.get("split_files") != data.manifest["files"]:
        raise ValueError("run provenance and prepared split checksums do not match")
    if not access.get("after_model_selection") or access.get("selection_hash") != stable_hash(
        selection
    ):
        raise ValueError("test was not proven to occur after model selection")
    for key in (f"recall@{evaluation_k}", f"ndcg@{evaluation_k}"):
        if key not in test_metrics or not math.isfinite(float(test_metrics[key])):
            raise ValueError(f"missing or non-finite verified metric: {key}")
    training_environment = _read_json(run_dir / "environment.json")
    environment = {
        **training_environment,
        **environment_manifest(to_lightgcn_config(resolved_config)),
        "training_started_environment_captured_at": training_environment.get("captured_at"),
    }
    checkpoint = cast(
        dict[str, Any],
        torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False),
    )
    expected_final_hash = stable_hash(
        {"base_dataset_hash": data.dataset_hash, "phase": "train-plus-validation"}
    )
    if checkpoint.get("dataset_hash") != expected_final_hash:
        raise ValueError("final checkpoint dataset hash mismatch")
    if int(checkpoint.get("step", -1)) != int(selection["best_step"]):
        raise ValueError("final checkpoint step does not match model selection")
    checkpoint_config = dict(checkpoint.get("config", {}))
    if checkpoint.get("config_hash") != stable_hash(checkpoint_config):
        raise ValueError("final checkpoint config hash mismatch")
    winner = dict(selection["winner"])
    for key in ("embedding_dim", "layers", "learning_rate", "l2"):
        if key not in winner:
            continue
        expected = float(winner[key]) if key in {"learning_rate", "l2"} else int(winner[key])
        actual = checkpoint_config.get(key, math.nan)
        actual = float(actual) if key in {"learning_rate", "l2"} else int(actual)
        if actual != expected:
            raise ValueError("final checkpoint winner configuration mismatch")
    sweep = _read_jsonl(run_dir / "sweep.jsonl")
    curve = _read_jsonl(run_dir / "training_curve.jsonl")
    completed_at = str(status["completed_at"])
    started_at = str(status["started_at"])
    duration_seconds = (
        datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)
    ).total_seconds()
    counts = dict(data.manifest["counts"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "verified",
        "experiment": "LightGCN transfer experiment on MovieLens-20M",
        "scope_note": (
            "Local MovieLens-20M results are not a direct replication of the paper's "
            "cross-dataset numbers."
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": duration_seconds,
        "test_metrics": test_metrics,
        "validation_metrics": selection["best_validation_metrics"],
        "winner": selection["winner"],
        "best_step": selection["best_step"],
        "trained_through_step": selection["trained_through_step"],
        "stopped_early": selection["stopped_early"],
        "dataset": {
            "name": "MovieLens-20M",
            "protocol": data.manifest["protocol"],
            "seed": data.manifest["seed"],
            "counts": counts,
        },
        "graph": {
            "nodes": int(counts["users"] + counts["items"]),
            "selection_undirected_edges": int(counts["train_edges"]),
            "selection_directed_adjacency_entries": int(2 * counts["train_edges"]),
            "final_undirected_edges": int(counts["train_edges"] + counts["validation_edges"]),
            "final_directed_adjacency_entries": int(
                2 * (counts["train_edges"] + counts["validation_edges"])
            ),
            "self_loops": 0,
            "normalization": "D^-1/2 A D^-1/2",
        },
        "config": resolved_config,
        "environment": environment,
        "sweep": sweep,
        "training_curve": curve,
        "provenance": {
            "dataset_hash": data.dataset_hash,
            "aggregate_report_hash_basis": stable_hash(
                {"status": status, "selection": selection, "test": test_metrics}
            ),
            "source": data.manifest["source"],
            "split_file_sha256": data.manifest["files"],
            "test_access": access,
        },
        "paper_results": PAPER_RESULTS,
        "reproduce": [
            "deep-recsys lightgcn prepare",
            "deep-recsys lightgcn reproduce",
            "deep-recsys lightgcn report",
        ],
        "limitations": [
            (
                "The paper did not evaluate MovieLens-20M; local and published numbers "
                "must not be subtracted."
            ),
            (
                "CPU FP32 with batch 262,144 is a resource adaptation from the paper's "
                "batch 1,024 and epoch schedule."
            ),
            "The local search is bounded by an eight-hour wall-time budget.",
        ],
    }
    report["report_hash"] = stable_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
