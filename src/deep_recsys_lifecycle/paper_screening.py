"""Validation-only controlled screening for the two paper benchmark modes.

The plan and each run registration are written before a runner sees validation outcomes.
Test labels never enter a run record or the selection input. A later test command must call
``require_frozen_selection`` before opening the sealed test cohort.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from .history_only_benchmark import HistoryOnlyBenchmarkConfig, run_history_only_benchmark
from .lightgcn import LightGCNConfig
from .models import RatingEvent
from .multivae import MultVAEConfig
from .paper_known_user_fusion import KnownUserHybridConfig, run_known_user_hybrid_benchmark

ScreenMode = Literal["known_user", "history_only"]
_MODES: tuple[ScreenMode, ScreenMode] = ("known_user", "history_only")
ScreenRunner = Callable[
    [Iterable[RatingEvent], ScreenMode, Mapping[str, object]], Mapping[str, object]
]


@dataclass(frozen=True, slots=True)
class ScreenAxis:
    name: str
    paths: tuple[str, ...]
    alternatives: tuple[object, object]
    reason: str
    first: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "paths": list(self.paths),
            "alternatives": list(self.alternatives),
            "reason": self.reason,
            "first": self.first,
        }


# Values are fixed after the #47 validation loss diagnosis and before variant outcomes.
# Later tickets may add handlers without changing these preregistered alternatives.
_SHARED_AXES = (
    ScreenAxis(
        "popularity_ranking",
        ("popularity_ranking",),
        (
            {
                "kind": "head_cap",
                "cap": 500,
                "score": "min(train_positive_count,500)",
                "tie": "movie_id_asc",
            },
            {
                "kind": "time_decay",
                "half_life_days": 365,
                "score": "sum(2**(-(max_train_event_time-event_time)/31536000))",
                "time_reference": "max_train_event_time",
                "tie": "movie_id_asc",
            },
        ),
        "Gold Movies absent from every source pool; time decay is a random-split "
        "project extension.",
    ),
    ScreenAxis(
        "itemknn_shrinkage",
        ("itemknn_shrinkage",),
        (
            {"lambda": 10, "formula": "cosine*cooccurrence/(cooccurrence+lambda)"},
            {"lambda": 100, "formula": "cosine*cooccurrence/(cooccurrence+lambda)"},
        ),
        "Source-pool misses.",
    ),
    ScreenAxis(
        "itemknn_neighbor_cap",
        ("itemknn_neighbor_cap",),
        (
            {"per_history_movie": 100, "tie": "movie_id_asc"},
            {"per_history_movie": 500, "tie": "movie_id_asc"},
        ),
        "Source-pool misses.",
    ),
    ScreenAxis(
        "itemknn_history_weighting",
        ("itemknn_history_weighting",),
        (
            {
                "kind": "inverse_history_popularity",
                "exponent": 0.5,
                "formula": "sum(cosine(candidate,h)/(1+train_positive_count(h))**exponent)",
            },
            {
                "kind": "inverse_history_popularity",
                "exponent": 1.0,
                "formula": "sum(cosine(candidate,h)/(1+train_positive_count(h))**exponent)",
            },
        ),
        "Source-pool misses across history-length segments.",
    ),
    ScreenAxis(
        "multivae_budget_checkpoint",
        ("multivae.epochs", "multivae_checkpoint"),
        (
            {"epochs": 8, "checkpoint": "last"},
            {"epochs": 20, "checkpoint": "validation_best_recall100_ndcg20_earliest_epoch"},
        ),
        "One-epoch baseline and the largest loss outside source pools.",
        first=True,
    ),
    ScreenAxis("multivae_kl_beta", ("multivae.kl_beta",), (0.05, 0.5), "Source-pool misses."),
    ScreenAxis("multivae_dropout", ("multivae.dropout",), (0.2, 0.5), "Source-pool misses."),
    ScreenAxis(
        "multivae_dimensions",
        ("multivae.hidden_dim", "multivae.latent_dim"),
        (
            {"hidden_dim": 64, "latent_dim": 32},
            {"hidden_dim": 128, "latent_dim": 64},
        ),
        "Source-pool misses.",
    ),
    ScreenAxis(
        "fusion_pool_depth",
        ("pool_limit_per_source",),
        (100, 400),
        "Union Gold Movies below final Top-100.",
    ),
    ScreenAxis(
        "fusion_source_allocation",
        ("fusion_source_allocation",),
        (
            {
                "kind": "per_source_quota",
                "quotas": {"popularity": 100, "itemknn": 100, "multivae": 100, "lightgcn": 100},
                "maximum_unique_input": 400,
                "dedup": "source_order_then_rank",
                "source_order": ["popularity", "itemknn", "multivae", "lightgcn"],
                "unused_quota_backfill": False,
            },
            {
                "kind": "per_source_quota",
                "quotas": {"popularity": 50, "itemknn": 100, "multivae": 200, "lightgcn": 50},
                "maximum_unique_input": 400,
                "dedup": "source_order_then_rank",
                "source_order": ["popularity", "itemknn", "multivae", "lightgcn"],
                "unused_quota_backfill": False,
            },
        ),
        "Union Gold Movies below final Top-100.",
    ),
    ScreenAxis(
        "fusion_ordering",
        ("fusion_ordering",),
        (
            {
                "kind": "rank_aware_prefilter",
                "maximum_unique_input": 400,
                "priority": "max_source(1/(60+source_rank))",
                "tie": "movie_id_asc",
                "final_order": "lhf_score_desc_movie_id_asc",
            },
            {
                "kind": "round_robin_prefilter",
                "maximum_unique_input": 400,
                "source_order": ["multivae", "itemknn", "popularity", "lightgcn"],
                "dedup": "first_occurrence",
                "tie": "movie_id_asc",
                "final_order": "lhf_score_desc_movie_id_asc",
            },
        ),
        "Union Gold Movies below final Top-100.",
    ),
)
_KNOWN_ONLY_AXES = (
    ScreenAxis(
        "lightgcn_budget_checkpoint",
        ("lightgcn.epochs", "lightgcn_checkpoint"),
        (
            {"epochs": 8, "checkpoint": "last"},
            {"epochs": 20, "checkpoint": "validation_best_recall100_ndcg20_earliest_epoch"},
        ),
        "One-epoch baseline, low standalone Recall, and 6,438 exclusive union Gold occurrences.",
        first=True,
    ),
    ScreenAxis("lightgcn_layers", ("lightgcn.layers",), (1, 3), "Low LightGCN source Recall."),
    ScreenAxis(
        "lightgcn_embedding_dim",
        ("lightgcn.embedding_dim",),
        (32, 64),
        "Low LightGCN source Recall.",
    ),
    ScreenAxis(
        "lightgcn_negative_samples",
        ("lightgcn.negative_samples",),
        (2, 4),
        "Low LightGCN source Recall.",
    ),
)


def initial_axes(mode: ScreenMode) -> tuple[ScreenAxis, ...]:
    shared = _SHARED_AXES
    axes: tuple[ScreenAxis, ...] = shared + _KNOWN_ONLY_AXES if mode == "known_user" else shared
    if mode == "history_only":
        converted: list[ScreenAxis] = []
        for axis in axes:
            alternatives = axis.alternatives
            if axis.name == "fusion_source_allocation":
                alternatives = (
                    {
                        "kind": "per_source_quota",
                        "quotas": {"popularity": 100, "itemknn": 100, "multivae": 100},
                        "maximum_unique_input": 300,
                        "dedup": "source_order_then_rank",
                        "source_order": ["popularity", "itemknn", "multivae"],
                        "unused_quota_backfill": False,
                    },
                    {
                        "kind": "per_source_quota",
                        "quotas": {"popularity": 50, "itemknn": 100, "multivae": 150},
                        "maximum_unique_input": 300,
                        "dedup": "source_order_then_rank",
                        "source_order": ["popularity", "itemknn", "multivae"],
                        "unused_quota_backfill": False,
                    },
                )
            elif axis.name == "fusion_ordering":
                alternatives = (
                    {
                        "kind": "rank_aware_prefilter",
                        "maximum_unique_input": 300,
                        "priority": "max_source(1/(60+source_rank))",
                        "tie": "movie_id_asc",
                        "final_order": "lhf_score_desc_movie_id_asc",
                    },
                    {
                        "kind": "round_robin_prefilter",
                        "maximum_unique_input": 300,
                        "source_order": ["multivae", "itemknn", "popularity"],
                        "dedup": "first_occurrence",
                        "tie": "movie_id_asc",
                        "final_order": "lhf_score_desc_movie_id_asc",
                    },
                )
            converted.append(
                ScreenAxis(
                    axis.name,
                    tuple(
                        path.replace("multivae.", "multivae_config.").replace(
                            "pool_limit_per_source", "candidate_pool_limit"
                        )
                        for path in axis.paths
                    ),
                    alternatives,
                    axis.reason,
                    axis.first,
                )
            )
        axes = tuple(converted)
    return tuple(sorted(axes, key=lambda axis: (not axis.first, axis.name)))


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _commit_temp(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _commit_temp(temp_path: Path, path: Path) -> None:
    """Atomically publish one complete artifact without replacing an earlier result."""

    os.link(temp_path, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _validation_seal(report: Mapping[str, object], mode: ScreenMode) -> dict[str, object]:
    """Fingerprint only training and validation evidence; test labels have no selector path."""

    source = _mapping(report["source"], "source")
    catalog = report["training_catalog_movie_ids"]
    if mode == "known_user":
        membership = _mapping(report["split_membership_sha256"], "split membership")
        validation = report["validation_gold_sets_by_subject"]
        train = _mapping(report["training"], "training")
        protocol = "lightgcn-ngcf-known-user"
        seal = {
            "protocol": protocol,
            "source": dict(source),
            "split_seed": _mapping(report["configuration"], "configuration")["seed"],
            "train_membership_sha256": membership["train"],
            "validation_membership_sha256": membership["validation"],
            "train_fit_sha256": train["outer_fit_event_ids_sha256"],
            "inner_fold_membership_sha256": _digest(train["inner_folds"]),
            "validation_gold_sha256": _digest(validation),
            "catalog_sha256": _digest(catalog),
            "validation_subject_count": _mapping(report["metrics"], "metrics")["query_count"],
            "test_membership_sha256": membership["test"],
        }
    else:
        cohorts = _mapping(report["cohort_subject_ids"], "cohorts")
        splits = _mapping(report["held_out_subject_splits"], "held-out splits")
        training = _mapping(report["fusion_training"], "fusion training")
        reproduction = _mapping(report["reproducibility"], "reproducibility")
        seal = {
            "protocol": "mult-vae-history-only",
            "source": dict(source),
            "split_seed": reproduction["split_seed"],
            "train_membership_sha256": _digest(cohorts["train"]),
            "validation_membership_sha256": _digest(cohorts["validation"]),
            "train_fit_sha256": training["outer_fit_event_ids_sha256"],
            "inner_fit_event_ids_sha256": training["inner_fit_event_ids_sha256"],
            "inner_pseudo_held_out_subject_ids_sha256": training[
                "inner_pseudo_held_out_subject_ids_sha256"
            ],
            "validation_gold_sha256": _digest(splits["validation"]),
            "catalog_sha256": _digest(catalog),
            "validation_subject_count": len(cohorts["validation"]),
            "test_membership_sha256": _mapping(splits["test"], "sealed test")["membership_sha256"],
        }
    return {**seal, "validation_cohort_sha256": _digest(seal)}


def _subject_metrics(report: Mapping[str, object], mode: ScreenMode) -> Mapping[str, Any]:
    if mode == "known_user":
        return _mapping(report["validation_subject_metrics"], "Known-User Subject metrics")
    diagnostic = _mapping(report["diagnostics"], "diagnostics")
    validation = _mapping(diagnostic["validation"], "validation diagnostics")
    return _mapping(validation["per_subject_final_lhf"], "History-Only Subject metrics")


def _summary(report: Mapping[str, object], mode: ScreenMode) -> dict[str, object]:
    if mode == "known_user":
        evaluation = _mapping(report["evaluation"], "evaluation")
        metrics = report["metrics"]
        source = evaluation["source_pools"]
        union = evaluation["oracle_union"]
        loss = evaluation["gold_loss"]
        latency = evaluation["inference_latency"]
    else:
        metrics = _mapping(report["metrics"], "metrics")["validation"]
        diagnostic = _mapping(report["diagnostics"], "diagnostics")["validation"]
        source = _mapping(diagnostic, "validation diagnostic")["sources"]
        union = _mapping(diagnostic, "validation diagnostic")["oracle_union"]
        loss = _mapping(diagnostic, "validation diagnostic")["gold_loss"]
        latency = _mapping(diagnostic, "validation diagnostic")["inference_latency"]
    return {
        "metrics": metrics,
        "source_pools": source,
        "oracle_union": union,
        "gold_loss": loss,
        "inference_latency": latency,
        "runtime": report["runtime"],
        "run_identity": report["run_identity"],
        "source": report["source"],
    }


def _nested_changes(left: object, right: object, prefix: str = "") -> set[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = set(left) | set(right)
        changes: set[str] = set()
        for key in keys:
            child = f"{prefix}.{key}" if prefix else str(key)
            changes.update(_nested_changes(left.get(key), right.get(key), child))
        return changes
    return set() if left == right else {prefix}


def _at_path(value: Mapping[str, object], path: str) -> object:
    current: object = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"registered configuration is missing {path}")
        current = current[part]
    return current


def _axis_value(configuration: Mapping[str, object], paths: tuple[str, ...]) -> object:
    if len(paths) == 1:
        return _at_path(configuration, paths[0])
    return {
        ("checkpoint" if path.endswith("_checkpoint") else path.rsplit(".", 1)[-1]): _at_path(
            configuration, path
        )
        for path in paths
    }


def default_screen_runner(
    events: Iterable[RatingEvent], mode: ScreenMode, configuration: Mapping[str, object]
) -> Mapping[str, object]:
    """Run the complete existing train-only fusion pipeline on validation only."""

    if mode == "known_user":
        allowed = {
            "seed",
            "inner_seed",
            "inner_fold_count",
            "pool_limit_per_source",
            "final_top_n",
            "max_negative_rows_per_query",
            "multivae",
            "multivae_device_preference",
            "lightgcn",
            "lightgcn_device_preference",
            "capture_training_rows",
        }
        extra = set(configuration) - allowed
        if extra:
            raise NotImplementedError(f"factor handler is not installed: {sorted(extra)}")
        multivae = MultVAEConfig(**cast(dict[str, Any], configuration["multivae"]))
        lightgcn = LightGCNConfig(**cast(dict[str, Any], configuration["lightgcn"]))
        known_config = KnownUserHybridConfig(
            seed=cast(int, configuration["seed"]),
            inner_seed=cast(int, configuration["inner_seed"]),
            inner_fold_count=cast(int, configuration["inner_fold_count"]),
            pool_limit=cast(int, configuration["pool_limit_per_source"]),
            max_negative_rows_per_query=cast(
                int | None, configuration["max_negative_rows_per_query"]
            ),
            multivae=multivae,
            lightgcn=lightgcn,
        )
        return run_known_user_hybrid_benchmark(events, config=known_config).to_dict()
    allowed = {
        "seed",
        "validation_subject_count",
        "test_subject_count",
        "inner_held_out_fraction",
        "candidate_pool_limit",
        "fusion_negative_rows_per_query",
        "multivae_config",
        "multivae_device_preference",
        "capture_evidence",
    }
    extra = set(configuration) - allowed
    if extra:
        raise NotImplementedError(f"factor handler is not installed: {sorted(extra)}")
    multivae = MultVAEConfig(**cast(dict[str, Any], configuration["multivae_config"]))
    history_config = HistoryOnlyBenchmarkConfig(
        seed=cast(int, configuration["seed"]),
        validation_subject_count=cast(int, configuration["validation_subject_count"]),
        test_subject_count=cast(int, configuration["test_subject_count"]),
        inner_held_out_fraction=cast(float, configuration["inner_held_out_fraction"]),
        candidate_pool_limit=cast(int, configuration["candidate_pool_limit"]),
        fusion_negative_rows_per_query=cast(
            int | None, configuration["fusion_negative_rows_per_query"]
        ),
        multivae_config=multivae,
        multivae_device_preference=cast(str, configuration["multivae_device_preference"]),
    )
    return run_history_only_benchmark(events, config=history_config, evaluate_test=False).to_dict()


def baseline_configuration(mode: ScreenMode) -> dict[str, object]:
    if mode == "known_user":
        return KnownUserHybridConfig().to_dict()
    value = HistoryOnlyBenchmarkConfig()
    return {
        "seed": value.seed,
        "validation_subject_count": value.validation_subject_count,
        "test_subject_count": value.test_subject_count,
        "inner_held_out_fraction": value.inner_held_out_fraction,
        "candidate_pool_limit": value.candidate_pool_limit,
        "fusion_negative_rows_per_query": value.fusion_negative_rows_per_query,
        "multivae_config": value.multivae_config.to_dict(),
        "multivae_device_preference": value.multivae_device_preference,
        "capture_evidence": False,
    }


class PaperScreenWorkspace:
    """Write-once registrations and compact validation evidence for one controlled screen."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        source_sha256: str | None,
        seed: int = 42,
        axes_by_mode: Mapping[ScreenMode, tuple[ScreenAxis, ...]] | None = None,
        baseline_configs: Mapping[ScreenMode, Mapping[str, object]] | None = None,
    ) -> PaperScreenWorkspace:
        workspace = cls(root)
        resolved_axes = axes_by_mode or {mode: initial_axes(mode) for mode in _MODES}
        axes = {mode: [axis.to_dict() for axis in resolved_axes[mode]] for mode in _MODES}
        plan: dict[str, object] = {
            "schema_version": 1,
            "source_sha256": source_sha256,
            "seed": seed,
            "baseline_configuration": {
                mode: dict(baseline_configs[mode])
                if baseline_configs is not None
                else baseline_configuration(mode)
                for mode in _MODES
            },
            "axes": axes,
            "maximum_initial_settings": {mode: 1 + 2 * len(resolved_axes[mode]) for mode in _MODES},
            "largest_addressable_loss": "outside_retriever_pools",
            "diagnosis": (
                "#47: 295253 Known-User and 52082 History-Only eligible Gold occurrences "
                "absent from all source pools"
            ),
        }
        plan["plan_sha256"] = _digest(plan)
        _write_once(root / "plan.json", plan)
        return workspace

    def plan(self) -> dict[str, Any]:
        plan = _read_object(self.root / "plan.json")
        digest = plan.pop("plan_sha256")
        if _digest(plan) != digest:
            raise ValueError("screen plan fingerprint mismatch")
        plan["plan_sha256"] = digest
        return plan

    def register_baseline(self, mode: ScreenMode) -> str:
        plan = self.plan()
        run_id = f"{mode}-baseline"
        config = _mapping(plan["baseline_configuration"], "baseline configurations")[mode]
        self._register(run_id, mode, "baseline", None, None, config)
        return run_id

    def register_variant(
        self,
        *,
        run_id: str,
        mode: ScreenMode,
        axis_name: str,
        alternative_index: int,
        reference_run_id: str,
        configuration: Mapping[str, object],
    ) -> None:
        plan = self.plan()
        if (self.root / "freeze.json").exists():
            raise ValueError("screen is frozen")
        axes = _mapping(plan["axes"], "axes")[mode]
        axis = next((item for item in axes if item["name"] == axis_name), None)
        if axis is None:
            raise ValueError(f"unknown axis: {axis_name}")
        if alternative_index not in (0, 1):
            raise ValueError("alternative index must be zero or one")
        reference = self._registration(reference_run_id)
        if reference["mode"] != mode:
            raise ValueError("reference mode differs")
        if not self._run_path(reference_run_id).exists():
            raise ValueError("reference run must complete before registering a variant")
        paths = set(axis["paths"])
        changes = _nested_changes(reference["configuration"], configuration)
        if not changes or not changes <= paths:
            raise ValueError(f"variant must change only {sorted(paths)}, got {sorted(changes)}")
        expected = axis["alternatives"][alternative_index]
        actual = _axis_value(configuration, tuple(axis["paths"]))
        if actual != expected:
            raise ValueError(f"{axis_name} must use preregistered alternative {alternative_index}")
        self._register(run_id, mode, axis_name, alternative_index, reference_run_id, configuration)

    def register_additional(
        self,
        *,
        run_id: str,
        mode: ScreenMode,
        reference_run_id: str,
        configuration: Mapping[str, object],
        rationale: str,
    ) -> None:
        """Preregister an adaptive combination before its validation run, with no count cap."""

        if not rationale.strip():
            raise ValueError("adaptive setting needs a rationale")
        if (self.root / "freeze.json").exists():
            raise ValueError("screen is frozen")
        reference = self._registration(reference_run_id)
        if reference["mode"] != mode or not self._run_path(reference_run_id).exists():
            raise ValueError("adaptive reference must be a completed run in the same mode")
        if not _nested_changes(reference["configuration"], configuration):
            raise ValueError("adaptive setting must change its reference configuration")
        self._register(run_id, mode, "adaptive", None, reference_run_id, configuration, rationale)

    def skip_axis(self, mode: ScreenMode, axis_name: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("skipped axis needs a measured-loss explanation")
        axes = _mapping(self.plan()["axes"], "axes")[mode]
        if not any(axis["name"] == axis_name for axis in axes):
            raise ValueError(f"unknown axis: {axis_name}")
        if (self.root / "freeze.json").exists():
            raise ValueError("screen is frozen")
        _write_once(
            self.root / "skips" / f"{mode}-{axis_name}.json",
            {
                "mode": mode,
                "axis": axis_name,
                "reason": reason,
                "plan_sha256": self.plan()["plan_sha256"],
            },
        )

    def _register(
        self,
        run_id: str,
        mode: ScreenMode,
        axis: str,
        alternative_index: int | None,
        reference_run_id: str | None,
        configuration: Mapping[str, object],
        rationale: str | None = None,
    ) -> None:
        if not run_id or not all(char.isalnum() or char in "-_" for char in run_id):
            raise ValueError("run ID must use letters, digits, hyphens or underscores")
        if (self.root / "freeze.json").exists():
            raise ValueError("screen is frozen")
        plan = self.plan()
        _write_once(
            self.root / "registrations" / f"{run_id}.json",
            {
                "run_id": run_id,
                "mode": mode,
                "axis": axis,
                "alternative_index": alternative_index,
                "reference_run_id": reference_run_id,
                "configuration": dict(configuration),
                "configuration_sha256": _digest(configuration),
                "plan_sha256": plan["plan_sha256"],
                "rationale": rationale,
            },
        )

    def _registration(self, run_id: str) -> dict[str, Any]:
        return _read_object(self.root / "registrations" / f"{run_id}.json")

    def _run_path(self, run_id: str) -> Path:
        return self.root / "runs" / f"{run_id}.json"

    def run_registered(
        self,
        run_id: str,
        events: Iterable[RatingEvent],
        *,
        runner: ScreenRunner = default_screen_runner,
        reference_report_path: Path | None = None,
    ) -> dict[str, Any]:
        if (self.root / "freeze.json").exists():
            raise ValueError("screen is frozen")
        registration = self._registration(run_id)
        mode = cast(ScreenMode, registration["mode"])
        if self._run_path(run_id).exists():
            raise FileExistsError(f"run already completed: {run_id}")
        if registration["plan_sha256"] != self.plan()["plan_sha256"]:
            raise ValueError("run registration plan fingerprint mismatch")
        config = _mapping(registration["configuration"], "configuration")
        if _digest(config) != registration["configuration_sha256"]:
            raise ValueError("registered configuration fingerprint mismatch")
        report = runner(events, mode, config)
        if report.get("test_status") not in (None, "sealed"):
            raise ValueError("runner exposed test status")
        reproduction = report.get("reproducibility")
        if isinstance(reproduction, Mapping) and reproduction.get("test_evaluated") is True:
            raise ValueError("runner opened test")
        if "test_metrics" in report:
            raise ValueError("runner exposed test metrics")
        if mode == "history_only" and "test" in _mapping(report["metrics"], "metrics"):
            raise ValueError("runner exposed History-Only test metrics")
        seal = _validation_seal(report, mode)
        if _mapping(seal["source"], "source")["dataset_sha256"] != self.plan()["source_sha256"]:
            raise ValueError("run source checksum differs from plan")
        if seal["split_seed"] != self.plan()["seed"]:
            raise ValueError("run seed differs from plan")
        summary = _summary(report, mode)
        if (
            registration["axis"] == "baseline"
            and _mapping(seal["source"], "source")["dataset_sha256"] is not None
        ):
            if reference_report_path is None:
                raise ValueError("full-data baseline needs the archived #47 validation report")
            reference = _read_object(reference_report_path)
            if (
                _validation_seal(reference, mode) != seal
                or _summary(reference, mode)["metrics"] != summary["metrics"]
            ):
                raise ValueError("baseline differs from the archived #47 cohort or macro metrics")
            verification_path = self.root / "baseline_verification" / f"{mode}.json"
            verification = {
                "reference_report": str(reference_report_path.resolve()),
                "reference_report_sha256": _file_sha256(reference_report_path),
                "cohort_sha256": seal["validation_cohort_sha256"],
                "macro_metrics": summary["metrics"],
            }
            if verification_path.exists():
                if _read_object(verification_path) != verification:
                    raise ValueError("baseline verification receipt changed")
            else:
                _write_once(verification_path, verification)
        seal_path = self.root / "cohorts" / f"{mode}.json"
        if seal_path.exists():
            if _read_object(seal_path) != seal:
                raise ValueError("training or validation cohort changed")
        elif registration["axis"] == "baseline":
            _write_once(seal_path, seal)
        else:
            raise ValueError("baseline must establish the frozen cohort first")
        subjects = _subject_metrics(report, mode)
        subject_path = self.root / "subjects" / f"{run_id}.json.gz"
        subject_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{run_id}.", dir=subject_path.parent)
        temp_path = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as raw:
                with gzip.open(raw, "wt", encoding="utf-8") as handle:
                    json.dump(subjects, handle, sort_keys=True, separators=(",", ":"))
                raw.flush()
                os.fsync(raw.fileno())
            _commit_temp(temp_path, subject_path)
        finally:
            temp_path.unlink(missing_ok=True)
        record = {
            "run_id": run_id,
            "mode": mode,
            "axis": registration["axis"],
            "reference_run_id": registration["reference_run_id"],
            "plan_sha256": registration["plan_sha256"],
            "configuration": dict(config),
            "configuration_sha256": registration["configuration_sha256"],
            "validation_cohort_sha256": seal["validation_cohort_sha256"],
            "per_subject_path": str(subject_path.relative_to(self.root)),
            "per_subject_sha256": _file_sha256(subject_path),
            **summary,
        }
        _write_once(self._run_path(run_id), record)
        return record

    def completed_runs(self, mode: ScreenMode) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for path in sorted((self.root / "runs").glob("*.json")):
            record = _read_object(path)
            if record["mode"] == mode:
                records[cast(str, record["run_id"])] = record
        return records

    def validation_subjects(self, run_id: str) -> dict[str, dict[str, float]]:
        record = _read_object(self._run_path(run_id))
        path = self.root / cast(str, record["per_subject_path"])
        if _file_sha256(path) != record["per_subject_sha256"]:
            raise ValueError("per-Subject result fingerprint mismatch")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("per-Subject result must be an object")
        return cast(dict[str, dict[str, float]], value)

    def select_and_freeze(
        self, *, bootstrap_seed: int = 42, bootstrap_draws: int = 2_000
    ) -> dict[str, Any]:
        """Select each mode from validation only, then atomically seal both choices."""

        from .paper_selection import select_validation_runs

        plan = self.plan()
        if (self.root / "freeze.json").exists():
            raise FileExistsError("selection is already frozen")
        frozen_modes: dict[str, object] = {}
        for mode in _MODES:
            records = self.completed_runs(mode)
            baseline_id = f"{mode}-baseline"
            if baseline_id not in records:
                raise ValueError(f"{mode} baseline has not completed")
            for axis in _mapping(plan["axes"], "axes")[mode]:
                axis_name = axis["name"]
                skip_path = self.root / "skips" / f"{mode}-{axis_name}.json"
                resolved = {
                    self._registration(run_id)["alternative_index"]
                    for run_id, record in records.items()
                    if record["axis"] == axis_name
                }
                if skip_path.exists():
                    if resolved:
                        raise ValueError(f"{mode}/{axis_name} is both screened and skipped")
                    skip = _read_object(skip_path)
                    if skip["plan_sha256"] != plan["plan_sha256"]:
                        raise ValueError("skip record plan fingerprint mismatch")
                elif resolved != {0, 1}:
                    raise ValueError(f"{mode}/{axis_name} needs both alternatives or a skip reason")
            subject_rows = {run_id: self.validation_subjects(run_id) for run_id in records}
            selection = select_validation_runs(
                baseline_id, subject_rows, seed=bootstrap_seed, draws=bootstrap_draws
            )
            selected_id = cast(str, selection["selected_id"])
            selected = records[selected_id]
            baseline = records[baseline_id]
            cohort = _read_object(self.root / "cohorts" / f"{mode}.json")
            frozen_modes[mode] = {
                "baseline_run_id": baseline_id,
                "baseline_configuration_sha256": baseline["configuration_sha256"],
                "selected_run_id": selected_id,
                "selected_configuration": selected["configuration"],
                "selected_configuration_sha256": selected["configuration_sha256"],
                "validation_cohort_sha256": cohort["validation_cohort_sha256"],
                "sealed_test_membership_sha256": cohort["test_membership_sha256"],
                "source": cohort["source"],
                "selection": selection,
                "run_identity": selected["run_identity"],
            }
        freeze: dict[str, Any] = {
            "schema_version": 1,
            "plan_sha256": plan["plan_sha256"],
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_draws": bootstrap_draws,
            "modes": frozen_modes,
            "test_status": "sealed",
        }
        freeze["freeze_sha256"] = _digest(freeze)
        _write_once(self.root / "freeze.json", freeze)
        return freeze

    def require_frozen_selection(self) -> dict[str, Any]:
        frozen = _read_object(self.root / "freeze.json")
        signature = frozen.pop("freeze_sha256")
        if _digest(frozen) != signature:
            raise ValueError("selection freeze fingerprint mismatch")
        frozen["freeze_sha256"] = signature
        if frozen["plan_sha256"] != self.plan()["plan_sha256"]:
            raise ValueError("selection freeze plan fingerprint mismatch")
        modes = _mapping(frozen["modes"], "frozen modes")
        if set(modes) != {"known_user", "history_only"}:
            raise ValueError("both paper modes must be frozen before test access")
        for mode, value in modes.items():
            entry = _mapping(value, f"{mode} freeze")
            selected = _read_object(self._run_path(cast(str, entry["selected_run_id"])))
            baseline = _read_object(self._run_path(cast(str, entry["baseline_run_id"])))
            cohort = _read_object(self.root / "cohorts" / f"{mode}.json")
            if (
                selected["configuration_sha256"] != entry["selected_configuration_sha256"]
                or baseline["configuration_sha256"] != entry["baseline_configuration_sha256"]
                or cohort["validation_cohort_sha256"] != entry["validation_cohort_sha256"]
                or cohort["test_membership_sha256"] != entry["sealed_test_membership_sha256"]
            ):
                raise ValueError("frozen run or cohort changed")
        return frozen


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
