"""Bounded successive-halving reproduction and final test protocol."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy
import scipy.sparse as sp
import torch

from ..reproducibility import seed_everything
from .checkpoint import load_checkpoint, save_checkpoint
from .config import LightGCNConfig
from .data import LightGCNData, stable_hash
from .evaluation import BPRSampler, evaluate_all_ranking
from .model import LightGCN, build_normalized_adjacency


class DeadlineExceeded(RuntimeError):
    """Raised after saving resumable state when the wall-time budget expires."""


@dataclass(frozen=True)
class Candidate:
    layers: int
    l2: float

    @property
    def identifier(self) -> str:
        return f"layers-{self.layers}_l2-{self.l2:.0e}"


@dataclass(frozen=True)
class TrainOutcome:
    step: int
    metrics: dict[str, float | int] | None
    best_step: int
    best_metrics: dict[str, float | int] | None
    stopped_early: bool


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _sysctl(name: str) -> str | None:
    try:
        result = subprocess.run(["sysctl", "-n", name], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _physical_memory_bytes() -> int | None:
    memory = _sysctl("hw.memsize")
    if memory is not None:
        return int(memory)
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, TypeError, ValueError):
        return None


def environment_manifest(config: LightGCNConfig) -> dict[str, Any]:
    """Capture enough runtime context to interpret the local numbers."""

    processor = (
        os.environ.get("LIGHTGCN_PROCESSOR")
        or _sysctl("machdep.cpu.brand_string")
        or platform.processor()
        or "Apple Silicon"
    )
    return {
        "captured_at": _utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "hardware_model": os.environ.get("LIGHTGCN_HARDWARE_MODEL") or _sysctl("hw.model"),
        "memory_bytes": _physical_memory_bytes(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": "cpu",
        "precision": "fp32",
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "torch_threads": config.training.threads,
        "batch_size": config.training.batch_size,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
    }


def _candidate_config(
    config: LightGCNConfig, candidate: Candidate, dataset_hash: str
) -> dict[str, Any]:
    return {
        "dataset_hash": dataset_hash,
        "seed": config.seed,
        "embedding_dim": config.model.embedding_dim,
        "layers": candidate.layers,
        "l2": candidate.l2,
        "learning_rate": config.training.learning_rate,
        "batch_size": config.training.batch_size,
        "device": "cpu",
        "precision": "fp32",
    }


def _is_better(
    metrics: dict[str, float | int],
    incumbent: dict[str, float | int] | None,
    k: int,
) -> bool:
    if incumbent is None:
        return True
    return (float(metrics[f"recall@{k}"]), float(metrics[f"ndcg@{k}"])) > (
        float(incumbent[f"recall@{k}"]),
        float(incumbent[f"ndcg@{k}"]),
    )


def _training_state(
    best_step: int, best_metrics: dict[str, float | int] | None, stale: int
) -> dict[str, Any]:
    return {"best_step": best_step, "best_metrics": best_metrics, "stale_validations": stale}


def _build_model(observed: sp.csr_matrix, config: LightGCNConfig, candidate: Candidate) -> LightGCN:
    adjacency = build_normalized_adjacency(observed)
    return LightGCN(
        observed.shape[0],
        observed.shape[1],
        config.model.embedding_dim,
        candidate.layers,
        adjacency,
    )


def _train_to_step(
    observed: sp.csr_matrix,
    *,
    validation: sp.csr_matrix | None,
    config: LightGCNConfig,
    candidate: Candidate,
    dataset_hash: str,
    target_steps: int,
    checkpoint_path: Path,
    log_path: Path,
    deadline: float,
    early_stopping: bool,
    best_path: Path | None = None,
) -> tuple[LightGCN, TrainOutcome]:
    candidate_config = _candidate_config(config, candidate, dataset_hash)
    config_hash = stable_hash(candidate_config)
    seed_everything(config.seed)
    model = _build_model(observed, config, candidate)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    sampler = BPRSampler(observed, config.seed)
    step = 0
    best_step = 0
    best_metrics: dict[str, float | int] | None = None
    stale = 0
    if checkpoint_path.exists():
        payload = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            sampler,
            dataset_hash=dataset_hash,
            config_hash=config_hash,
        )
        step = int(payload["step"])
        state = dict(payload.get("training_state", {}))
        best_step = int(state.get("best_step", 0))
        loaded_metrics = state.get("best_metrics")
        best_metrics = dict(loaded_metrics) if isinstance(loaded_metrics, dict) else None
        stale = int(state.get("stale_validations", 0))
    if early_stopping and stale >= config.training.patience:
        return model, TrainOutcome(
            step=step,
            metrics=best_metrics,
            best_step=best_step,
            best_metrics=best_metrics,
            stopped_early=True,
        )
    model.train()
    stopped_early = False
    latest_metrics: dict[str, float | int] | None = None
    while step < target_steps:
        if time.time() >= deadline:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                sampler,
                step=step,
                dataset_hash=dataset_hash,
                config_hash=config_hash,
                config=candidate_config,
                training_state=_training_state(best_step, best_metrics, stale),
            )
            raise DeadlineExceeded("eight-hour reproduction budget exhausted")
        users, positives, negatives = sampler.sample(config.training.batch_size)
        user_tensor = torch.from_numpy(users)
        positive_tensor = torch.from_numpy(positives)
        negative_tensor = torch.from_numpy(negatives)
        optimizer.zero_grad(set_to_none=True)
        loss, ranking_loss, ego_l2 = model.bpr_loss(
            user_tensor, positive_tensor, negative_tensor, candidate.l2
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        step += 1
        if step == 1 or step % 10 == 0 or step == target_steps:
            _append_jsonl(
                log_path,
                {
                    "event": "train",
                    "step": step,
                    "loss": float(loss.detach()),
                    "bpr_loss": float(ranking_loss.detach()),
                    "ego_l2": float(ego_l2.detach()),
                    "timestamp": _utc_now(),
                },
            )
        should_validate = (
            validation is not None
            and early_stopping
            and step >= config.training.minimum_steps
            and step % config.training.validation_every == 0
        )
        if should_validate:
            latest_metrics = evaluate_all_ranking(
                model,
                observed,
                validation,
                k=config.evaluation.k,
                user_batch_size=config.evaluation.user_batch_size,
            )
            _append_jsonl(
                log_path,
                {"event": "validation", "step": step, **latest_metrics, "timestamp": _utc_now()},
            )
            if _is_better(latest_metrics, best_metrics, config.evaluation.k):
                best_metrics = latest_metrics
                best_step = step
                stale = 0
                if best_path is not None:
                    save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        sampler,
                        step=step,
                        dataset_hash=dataset_hash,
                        config_hash=config_hash,
                        config=candidate_config,
                        training_state=_training_state(best_step, best_metrics, stale),
                    )
            else:
                stale += 1
            model.train()
            if stale >= config.training.patience:
                stopped_early = True
        state = _training_state(best_step, best_metrics, stale)
        if step % config.training.checkpoint_every == 0 or step == target_steps or stopped_early:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                sampler,
                step=step,
                dataset_hash=dataset_hash,
                config_hash=config_hash,
                config=candidate_config,
                training_state=state,
            )
        if stopped_early:
            break
    if validation is not None and not early_stopping:
        if time.time() >= deadline:
            raise DeadlineExceeded("eight-hour reproduction budget exhausted before validation")
        latest_metrics = evaluate_all_ranking(
            model,
            observed,
            validation,
            k=config.evaluation.k,
            user_batch_size=config.evaluation.user_batch_size,
        )
        _append_jsonl(
            log_path,
            {"event": "validation", "step": step, **latest_metrics, "timestamp": _utc_now()},
        )
    if early_stopping and best_metrics is None:
        raise RuntimeError("no validation occurred; minimum_steps exceeds the training horizon")
    return model, TrainOutcome(
        step=step,
        metrics=latest_metrics,
        best_step=best_step,
        best_metrics=best_metrics,
        stopped_early=stopped_early,
    )


def _rank_candidates(
    results: list[tuple[Candidate, dict[str, float | int]]], k: int
) -> list[tuple[Candidate, dict[str, float | int]]]:
    def key(value: tuple[Candidate, dict[str, float | int]]) -> tuple[float, float, int, float]:
        candidate, metrics = value
        return (
            -float(metrics[f"recall@{k}"]),
            -float(metrics[f"ndcg@{k}"]),
            candidate.layers,
            abs(math.log10(candidate.l2) + 4.0),
        )

    return sorted(results, key=key)


def _run_sweep_stage(
    candidates: list[Candidate],
    *,
    stage: str,
    target_steps: int,
    data: LightGCNData,
    config: LightGCNConfig,
    run_dir: Path,
    deadline: float,
) -> list[tuple[Candidate, dict[str, float | int]]]:
    results: list[tuple[Candidate, dict[str, float | int]]] = []
    for candidate in candidates:
        candidate_dir = run_dir / "candidates" / candidate.identifier
        stage_result_path = candidate_dir / f"{stage}.json"
        if stage_result_path.exists():
            record = _read_json(stage_result_path)
            results.append((candidate, dict(record["metrics"])))
            continue
        _model, outcome = _train_to_step(
            data.train,
            validation=data.validation,
            config=config,
            candidate=candidate,
            dataset_hash=data.dataset_hash,
            target_steps=target_steps,
            checkpoint_path=candidate_dir / "last.pt",
            log_path=candidate_dir / "metrics.jsonl",
            deadline=deadline,
            early_stopping=False,
        )
        if outcome.metrics is None:
            raise RuntimeError("sweep candidate did not produce validation metrics")
        record = {
            "stage": stage,
            "candidate": {"layers": candidate.layers, "l2": candidate.l2},
            "step": outcome.step,
            "metrics": outcome.metrics,
        }
        _write_json(stage_result_path, record)
        _append_jsonl(run_dir / "sweep.jsonl", record)
        results.append((candidate, outcome.metrics))
    return _rank_candidates(results, config.evaluation.k)


def _initial_status(config: LightGCNConfig, data: LightGCNData) -> dict[str, Any]:
    return {
        "status": "incomplete",
        "phase": "initializing",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "dataset_hash": data.dataset_hash,
        "config_hash": stable_hash(config.as_dict()),
        "test_accessed": False,
    }


def _deadline(status: dict[str, Any], config: LightGCNConfig) -> float:
    started = datetime.fromisoformat(str(status["started_at"])).timestamp()
    return started + config.training.wall_time_hours * 3600.0


def _set_phase(status_path: Path, status: dict[str, Any], phase: str) -> None:
    status["phase"] = phase
    status["updated_at"] = _utc_now()
    _write_json(status_path, status)


def reproduce_lightgcn(data: LightGCNData, config: LightGCNConfig, run_dir: Path) -> dict[str, Any]:
    """Run sweep, winner training, clean retrain, and exactly one test evaluation."""

    torch.set_num_threads(config.training.threads)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "run_status.json"
    status = _read_json(status_path) if status_path.exists() else _initial_status(config, data)
    if status.get("dataset_hash") != data.dataset_hash:
        raise ValueError("run directory belongs to a different prepared dataset")
    if status.get("config_hash") != stable_hash(config.as_dict()):
        raise ValueError("run directory belongs to a different resolved config")
    if status.get("status") == "verified":
        return status
    _write_json(status_path, status)
    _write_json(run_dir / "resolved_config.json", config.as_dict())
    _write_json(run_dir / "environment.json", environment_manifest(config))
    _write_json(
        run_dir / "provenance.json",
        {
            "dataset_hash": data.dataset_hash,
            "split_files": data.manifest["files"],
            "source": data.manifest["source"],
            "protocol": data.manifest["protocol"],
        },
    )
    deadline = _deadline(status, config)
    candidates = [
        Candidate(layers, l2) for layers in config.sweep.layers for l2 in config.sweep.l2_values
    ]
    try:
        _set_phase(status_path, status, "sweep-round-1")
        round1 = _run_sweep_stage(
            candidates,
            stage="round1",
            target_steps=config.sweep.round1_steps,
            data=data,
            config=config,
            run_dir=run_dir,
            deadline=deadline,
        )
        finalists = [candidate for candidate, _metrics in round1[: config.sweep.finalists]]
        _set_phase(status_path, status, "sweep-round-2")
        round2 = _run_sweep_stage(
            finalists,
            stage="round2",
            target_steps=config.sweep.round2_steps,
            data=data,
            config=config,
            run_dir=run_dir,
            deadline=deadline,
        )
        _write_jsonl(
            run_dir / "sweep.jsonl",
            [
                {
                    "stage": stage,
                    "candidate": {"layers": candidate.layers, "l2": candidate.l2},
                    "step": target_steps,
                    "metrics": metrics,
                }
                for stage, target_steps, stage_results in (
                    ("round1", config.sweep.round1_steps, round1),
                    ("round2", config.sweep.round2_steps, round2),
                )
                for candidate, metrics in stage_results
            ],
        )
        winner = round2[0][0]
        _write_json(
            run_dir / "selection.json",
            {
                "selected_at": _utc_now(),
                "criterion": [f"recall@{config.evaluation.k}", f"ndcg@{config.evaluation.k}"],
                "winner": {"layers": winner.layers, "l2": winner.l2},
                "round1_finalists": [item.identifier for item in finalists],
            },
        )
        _set_phase(status_path, status, "winner-training")
        winner_dir = run_dir / "candidates" / winner.identifier
        _winner_model, winner_outcome = _train_to_step(
            data.train,
            validation=data.validation,
            config=config,
            candidate=winner,
            dataset_hash=data.dataset_hash,
            target_steps=config.training.max_steps,
            checkpoint_path=winner_dir / "last.pt",
            best_path=winner_dir / "best.pt",
            log_path=run_dir / "training_curve.jsonl",
            deadline=deadline,
            early_stopping=True,
        )
        if winner_outcome.best_metrics is None or winner_outcome.best_step <= 0:
            raise RuntimeError("winner training did not select a valid step")
        selection = _read_json(run_dir / "selection.json")
        selection.update(
            {
                "best_step": winner_outcome.best_step,
                "best_validation_metrics": winner_outcome.best_metrics,
                "stopped_early": winner_outcome.stopped_early,
                "trained_through_step": winner_outcome.step,
            }
        )
        _write_json(run_dir / "selection.json", selection)

        _set_phase(status_path, status, "final-retrain")
        combined = (data.train + data.validation).tocsr().astype(np.float32)
        combined.data.fill(1.0)
        final_hash = stable_hash(
            {"base_dataset_hash": data.dataset_hash, "phase": "train-plus-validation"}
        )
        final_dir = run_dir / "final"
        final_model, final_outcome = _train_to_step(
            combined,
            validation=None,
            config=config,
            candidate=winner,
            dataset_hash=final_hash,
            target_steps=winner_outcome.best_step,
            checkpoint_path=final_dir / "last.pt",
            log_path=final_dir / "metrics.jsonl",
            deadline=deadline,
            early_stopping=False,
        )
        if final_outcome.step != winner_outcome.best_step:
            raise RuntimeError("final retrain did not reach the selected step")
        shutil.copy2(final_dir / "last.pt", run_dir / "best.pt")
        shutil.copy2(final_dir / "last.pt", run_dir / "last.pt")

        test_metrics_path = run_dir / "test_metrics.json"
        access_path = run_dir / "test_access.json"
        if access_path.exists() and not test_metrics_path.exists():
            raise RuntimeError("test access was recorded without metrics; refusing a second access")
        if not test_metrics_path.exists():
            if time.time() >= deadline:
                raise DeadlineExceeded(
                    "eight-hour reproduction budget exhausted before test evaluation"
                )
            _set_phase(status_path, status, "test-evaluation")
            access = {
                "accessed_at": _utc_now(),
                "after_model_selection": True,
                "selection_hash": stable_hash(selection),
                "test_edges": int(data.test.nnz),
            }
            _write_json(access_path, access)
            status["test_accessed"] = True
            _write_json(status_path, status)
            test_metrics = evaluate_all_ranking(
                final_model,
                combined,
                data.test,
                k=config.evaluation.k,
                user_batch_size=config.evaluation.user_batch_size,
            )
            if not all(
                math.isfinite(float(value))
                for key, value in test_metrics.items()
                if key.startswith(("recall@", "ndcg@"))
            ):
                raise RuntimeError("test metrics are not finite")
            _write_json(test_metrics_path, test_metrics)
        else:
            test_metrics = _read_json(test_metrics_path)
        status.update(
            {
                "status": "verified",
                "phase": "complete",
                "completed_at": _utc_now(),
                "updated_at": _utc_now(),
                "winner": {"layers": winner.layers, "l2": winner.l2},
                "best_step": winner_outcome.best_step,
                "validation_metrics": winner_outcome.best_metrics,
                "test_metrics": test_metrics,
                "test_accessed": True,
            }
        )
        _write_json(status_path, status)
        return status
    except DeadlineExceeded as error:
        status.update(
            {
                "status": "incomplete",
                "reason": str(error),
                "updated_at": _utc_now(),
            }
        )
        _write_json(status_path, status)
        return status
