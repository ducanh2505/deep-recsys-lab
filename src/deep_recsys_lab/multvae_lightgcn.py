"""Multi-VAE training and reporting on the repository's LightGCN split."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy
import scipy.sparse as sp
import torch
from omegaconf import DictConfig, OmegaConf

from .config import compose_hydra_config
from .data.dataset import load_prepared_data
from .device import select_device
from .evaluation.evaluator import evaluate_model
from .lightgcn.data import LightGCNData, load_lightgcn_data, sha256_file, stable_hash
from .model import build_model
from .model.base import BaseRecommender
from .reproducibility import capture_rng_state, restore_rng_state, seed_everything
from .training.loss import multivae_loss
from .training.schedule import beta_at_step

CHECKPOINT_SCHEMA = "multvae-lightgcn-1"


@dataclass(frozen=True)
class MultiVAESweepConfig:
    """Finite hyper-parameter grid used by the successive-halving run."""

    architectures: tuple[tuple[int, int, int], ...] = ((600, 200, 600), (400, 200, 400))
    dropout: tuple[float, ...] = (0.0, 0.5)
    beta_cap: tuple[float, ...] = (0.2, 0.8)
    learning_rate: tuple[float, ...] = (5e-4, 1e-3)
    weight_decay: tuple[float, ...] = (0.0, 1e-5)


@dataclass(frozen=True)
class MultiVAELightGCNConfig:
    """Resolved configuration for one resumable Multi-VAE experiment."""

    seed: int = 98_765
    device: str = "auto"
    data_dir: str = "data/lightgcn/ml-20m-v1"
    run_dir: str = "outputs/multvae-lightgcn/ml-20m-v1"
    baseline_checkpoint: str | None = "outputs/multvae-20260818-224228/best.pt"
    baseline_data_dir: str | None = "data/processed"
    lightgcn_metrics_path: str | None = "outputs/lightgcn/ml-20m-v1/test_metrics.json"
    batch_size: int = 500
    num_workers: int = 0
    latent_dim: int = 200
    input_normalization: str = "l2"
    total_anneal_steps: int = 200_000
    round1_epochs: int = 20
    round2_epochs: int = 60
    max_epochs: int = 200
    round2_finalists: int = 8
    final_candidates: int = 2
    validation_every: int = 1
    patience: int = 10
    checkpoint_every: int = 5
    selection_k: int = 20
    evaluation_ks: tuple[int, ...] = (10, 20, 50, 100)
    report_output: str | None = None
    sweep: MultiVAESweepConfig = MultiVAESweepConfig()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe resolved configuration."""

        return asdict(self)

    def validate(self) -> None:
        """Reject invalid schedules before creating a run directory."""

        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers cannot be negative")
        if self.latent_dim <= 0 or self.total_anneal_steps < 0:
            raise ValueError("latent_dim must be positive and total_anneal_steps non-negative")
        if not 0 < self.round1_epochs <= self.round2_epochs <= self.max_epochs:
            raise ValueError("epoch stages must satisfy 0 < round1 <= round2 <= max_epochs")
        if self.round2_finalists <= 0 or self.final_candidates <= 0:
            raise ValueError("successive-halving finalist counts must be positive")
        if self.validation_every <= 0 or self.patience <= 0 or self.checkpoint_every <= 0:
            raise ValueError("validation, patience, and checkpoint intervals must be positive")
        if (
            self.selection_k <= 0
            or not self.evaluation_ks
            or len(set(self.evaluation_ks)) != len(self.evaluation_ks)
            or any(k <= 0 for k in self.evaluation_ks)
            or self.selection_k not in self.evaluation_ks
        ):
            raise ValueError("selection_k must be one of evaluation_ks")
        sweep = self.sweep
        if not all(
            (
                sweep.architectures,
                sweep.dropout,
                sweep.beta_cap,
                sweep.learning_rate,
                sweep.weight_decay,
            )
        ):
            raise ValueError("the sweep dimensions cannot be empty")
        if self.round2_finalists > self.trial_count:
            raise ValueError("round2_finalists cannot exceed the number of trials")
        if self.final_candidates > self.round2_finalists:
            raise ValueError("final_candidates cannot exceed round2_finalists")
        for architecture in sweep.architectures:
            if len(architecture) != 3 or architecture[1] != self.latent_dim:
                raise ValueError("each architecture must be [hidden, latent_dim, hidden]")
        if any(value < 0 or value >= 1 for value in sweep.dropout):
            raise ValueError("dropout values must be in [0, 1)")
        if any(value < 0 for value in sweep.beta_cap):
            raise ValueError("beta caps cannot be negative")
        if any(value <= 0 for value in sweep.learning_rate):
            raise ValueError("learning rates must be positive")
        if any(value < 0 for value in sweep.weight_decay):
            raise ValueError("weight decay values cannot be negative")

    @property
    def trial_count(self) -> int:
        sweep = self.sweep
        return (
            len(sweep.architectures)
            * len(sweep.dropout)
            * len(sweep.beta_cap)
            * len(sweep.learning_rate)
            * len(sweep.weight_decay)
        )


@dataclass(frozen=True)
class MultiVAETrial:
    """One deterministic Multi-VAE sweep candidate."""

    hidden_dims: tuple[int, int, int]
    dropout: float
    beta_cap: float
    learning_rate: float
    weight_decay: float

    @property
    def identifier(self) -> str:
        hidden = "-".join(str(value) for value in self.hidden_dims)
        return (
            f"h-{hidden}_drop-{self.dropout:g}_beta-{self.beta_cap:g}_"
            f"lr-{self.learning_rate:.0e}_wd-{self.weight_decay:.0e}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "beta_cap": self.beta_cap,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }


@dataclass(frozen=True)
class MultiVAELightGCNViews:
    """Matrix views required by selection, final retraining, and testing."""

    train: sp.csr_matrix
    validation: sp.csr_matrix
    test: sp.csr_matrix
    combined: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    dataset_hash: str
    manifest: dict[str, Any]

    @property
    def n_users(self) -> int:
        return int(self.train.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.train.shape[1])


def _binary_csr(matrix: sp.csr_matrix) -> sp.csr_matrix:
    result = matrix.tocsr(copy=True).astype(np.float32)
    result.sum_duplicates()
    if result.nnz and not np.all(result.data == 1.0):
        raise ValueError("LightGCN split matrices must contain binary interactions")
    result.data.fill(1.0)
    result.sort_indices()
    return result


def build_multvae_views(data: LightGCNData) -> MultiVAELightGCNViews:
    """Convert a verified LightGCN artifact into Multi-VAE matrix views."""

    train = _binary_csr(data.train)
    validation = _binary_csr(data.validation)
    test = _binary_csr(data.test)
    if train.shape != validation.shape or train.shape != test.shape:
        raise ValueError("LightGCN matrices must have identical shapes")
    if train.multiply(validation).nnz or train.multiply(test).nnz:
        raise ValueError("LightGCN train and held-out matrices overlap")
    if validation.multiply(test).nnz:
        raise ValueError("LightGCN validation and test matrices overlap")
    combined = (train + validation).tocsr().astype(np.float32)
    combined.sum_duplicates()
    combined.data.fill(1.0)
    combined.sort_indices()
    return MultiVAELightGCNViews(
        train=train,
        validation=validation,
        test=test,
        combined=combined,
        user_ids=np.asarray(data.user_ids).copy(),
        item_ids=np.asarray(data.item_ids).copy(),
        dataset_hash=data.dataset_hash,
        manifest=dict(data.manifest),
    )


def _tuple_architectures(value: Any, latent_dim: int) -> tuple[tuple[int, int, int], ...]:
    architectures: list[tuple[int, int, int]] = []
    for architecture in value:
        if len(architecture) != 3 or int(architecture[1]) != latent_dim:
            continue
        parts = tuple(int(part) for part in architecture)
        architectures.append((parts[0], parts[1], parts[2]))
    return tuple(architectures)


def to_multvae_lightgcn_config(config: DictConfig | dict[str, Any]) -> MultiVAELightGCNConfig:
    """Convert a Hydra config into the typed experiment schema."""

    raw = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    assert isinstance(raw, dict)
    sweep_raw = dict(raw.get("sweep", {}))
    latent_dim = int(raw.get("latent_dim", 200))
    sweep = MultiVAESweepConfig(
        architectures=_tuple_architectures(
            sweep_raw.get("architectures", ((600, 200, 600), (400, 200, 400))), latent_dim
        ),
        dropout=tuple(float(value) for value in sweep_raw.get("dropout", (0.0, 0.5))),
        beta_cap=tuple(float(value) for value in sweep_raw.get("beta_cap", (0.2, 0.8))),
        learning_rate=tuple(
            float(value) for value in sweep_raw.get("learning_rate", (5e-4, 1e-3))
        ),
        weight_decay=tuple(float(value) for value in sweep_raw.get("weight_decay", (0.0, 1e-5))),
    )
    return MultiVAELightGCNConfig(
        seed=int(raw.get("seed", 98_765)),
        device=str(raw.get("device", "auto")),
        data_dir=str(raw.get("data_dir", "data/lightgcn/ml-20m-v1")),
        run_dir=str(raw.get("run_dir", "outputs/multvae-lightgcn/ml-20m-v1")),
        baseline_checkpoint=(
            str(raw["baseline_checkpoint"]) if raw.get("baseline_checkpoint") else None
        ),
        baseline_data_dir=(str(raw["baseline_data_dir"]) if raw.get("baseline_data_dir") else None),
        lightgcn_metrics_path=(
            str(raw["lightgcn_metrics_path"]) if raw.get("lightgcn_metrics_path") else None
        ),
        batch_size=int(raw.get("batch_size", 500)),
        num_workers=int(raw.get("num_workers", 0)),
        latent_dim=latent_dim,
        input_normalization=str(raw.get("input_normalization", "l2")),
        total_anneal_steps=int(raw.get("total_anneal_steps", 200_000)),
        round1_epochs=int(raw.get("round1_epochs", 20)),
        round2_epochs=int(raw.get("round2_epochs", 60)),
        max_epochs=int(raw.get("max_epochs", 200)),
        round2_finalists=int(raw.get("round2_finalists", 8)),
        final_candidates=int(raw.get("final_candidates", 2)),
        validation_every=int(raw.get("validation_every", 1)),
        patience=int(raw.get("patience", 10)),
        checkpoint_every=int(raw.get("checkpoint_every", 5)),
        selection_k=int(raw.get("selection_k", 20)),
        evaluation_ks=tuple(int(value) for value in raw.get("evaluation_ks", (10, 20, 50, 100))),
        report_output=(str(raw["report_output"]) if raw.get("report_output") else None),
        sweep=sweep,
    )


def compose_multvae_lightgcn_config(
    config_name: str = "multvae_lightgcn", overrides: list[str] | None = None
) -> MultiVAELightGCNConfig:
    """Compose and validate a dedicated Multi-VAE LightGCN-split config."""

    config = to_multvae_lightgcn_config(compose_hydra_config(config_name, overrides or ()))
    config.validate()
    return config


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(cast(dict[str, Any], json.loads(line)))
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _environment(device: torch.device, config: MultiVAELightGCNConfig) -> dict[str, Any]:
    return {
        "captured_at": _utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": str(device),
        "precision": "fp32",
        "torch_threads": torch.get_num_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ),
        "batch_size": config.batch_size,
        "seed": config.seed,
        "cpu_count": os.cpu_count(),
    }


def _trial_list(config: MultiVAELightGCNConfig) -> list[MultiVAETrial]:
    sweep = config.sweep
    return [
        MultiVAETrial(architecture, dropout, beta, learning_rate, weight_decay)
        for architecture, dropout, beta, learning_rate, weight_decay in product(
            sweep.architectures,
            sweep.dropout,
            sweep.beta_cap,
            sweep.learning_rate,
            sweep.weight_decay,
        )
    ]


def _build_model(
    trial: MultiVAETrial, n_items: int, config: MultiVAELightGCNConfig
) -> BaseRecommender:
    return build_model(
        "multvae",
        n_items=n_items,
        hidden_dims=trial.hidden_dims,
        latent_dim=config.latent_dim,
        dropout=trial.dropout,
        input_normalization=config.input_normalization,
    )


class _SparseBatchIterable:
    """Convert CSR rows in batches, avoiding one Python conversion per user."""

    def __init__(
        self,
        matrix: sp.csr_matrix,
        config: MultiVAELightGCNConfig,
        *,
        shuffle: bool,
        fold_in: sp.csr_matrix | None = None,
        fold_out: sp.csr_matrix | None = None,
    ) -> None:
        if (fold_in is None) != (fold_out is None):
            raise ValueError("fold_in and fold_out must be provided together")
        self.matrix = matrix.tocsr()
        self.config = config
        self.shuffle = shuffle
        self.fold_in = fold_in.tocsr() if fold_in is not None else None
        self.fold_out = fold_out.tocsr() if fold_out is not None else None
        if (
            self.fold_in is not None
            and self.fold_out is not None
            and (
                self.fold_in.shape != self.matrix.shape or self.fold_out.shape != self.matrix.shape
            )
        ):
            raise ValueError("fold-in and fold-out matrices must match the input shape")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.shuffle:
            order = torch.randperm(self.matrix.shape[0]).numpy()
        else:
            order = np.arange(self.matrix.shape[0], dtype=np.int64)
        for start in range(0, self.matrix.shape[0], self.config.batch_size):
            rows = order[start : start + self.config.batch_size]
            data = self.matrix[rows].toarray().astype(np.float32, copy=False)
            batch = {"data": torch.from_numpy(data)}
            if self.fold_in is not None and self.fold_out is not None:
                truth = self.fold_out[rows].toarray().astype(np.float32, copy=False)
                batch["ground_truth"] = torch.from_numpy(truth)
            yield batch


def _loader(
    matrix: sp.csr_matrix,
    config: MultiVAELightGCNConfig,
    *,
    shuffle: bool,
    fold_in: sp.csr_matrix | None = None,
    fold_out: sp.csr_matrix | None = None,
) -> Iterable[dict[str, torch.Tensor]]:
    return _SparseBatchIterable(
        matrix,
        config,
        shuffle=shuffle,
        fold_in=fold_in,
        fold_out=fold_out,
    )


def _evaluate(
    model: BaseRecommender,
    observed: sp.csr_matrix,
    truth: sp.csr_matrix,
    config: MultiVAELightGCNConfig,
    device: torch.device,
) -> dict[str, float]:
    loader = _loader(
        observed,
        config,
        shuffle=False,
        fold_in=observed,
        fold_out=truth,
    )
    return evaluate_model(
        model,
        loader,
        device=device,
        ks=config.evaluation_ks,
        mask_seen=True,
        ranking_protocol="lightgcn",
    )


def _selection_key(metrics: dict[str, float], k: int) -> tuple[float, float]:
    return (float(metrics[f"recall@{k}"]), float(metrics[f"ndcg@{k}"]))


def _rank_results(
    results: list[dict[str, Any]], k: int, metric_field: str = "validation_metrics"
) -> list[dict[str, Any]]:
    def key(result: dict[str, Any]) -> tuple[float, float, str]:
        metrics = cast(dict[str, float], result[metric_field])
        score = _selection_key(metrics, k)
        return (-score[0], -score[1], str(result["trial"]["id"]))

    return sorted(results, key=key)


def _trial_from_result(result: dict[str, Any]) -> MultiVAETrial:
    trial_data = dict(result["trial"])
    parts = tuple(int(value) for value in trial_data["hidden_dims"])
    if len(parts) != 3:
        raise ValueError("trial hidden_dims must have length three")
    return MultiVAETrial(
        (parts[0], parts[1], parts[2]),
        float(trial_data["dropout"]),
        float(trial_data["beta_cap"]),
        float(trial_data["learning_rate"]),
        float(trial_data["weight_decay"]),
    )


def _save_checkpoint(
    path: Path,
    model: BaseRecommender,
    optimizer: torch.optim.Optimizer,
    *,
    trial: MultiVAETrial,
    stage: str,
    epoch: int,
    global_step: int,
    data_hash: str,
    config_hash: str,
    best_epoch: int = 0,
    best_metrics: dict[str, float] | None = None,
    stale: int = 0,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "model_state": model.state_dict(),
        "model_config": model.architecture_config(),
        "optimizer_state": optimizer.state_dict(),
        "trial": {"id": trial.identifier, **trial.as_dict()},
        "stage": stage,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "dataset_hash": data_hash,
        "config_hash": config_hash,
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics or {},
        "stale": int(stale),
        "rng_state": capture_rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    model: BaseRecommender,
    optimizer: torch.optim.Optimizer,
    *,
    trial: MultiVAETrial,
    data_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported Multi-VAE LightGCN checkpoint: {path}")
    if payload.get("dataset_hash") != data_hash:
        raise ValueError(f"checkpoint dataset hash mismatch: {path}")
    if payload.get("config_hash") != config_hash:
        raise ValueError(f"checkpoint config hash mismatch: {path}")
    recorded_trial = dict(payload.get("trial", {}))
    if recorded_trial.get("id") != trial.identifier:
        raise ValueError(f"checkpoint trial mismatch: {path}")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    if payload.get("rng_state"):
        restore_rng_state(payload["rng_state"])
    return payload


def _train_epoch(
    model: BaseRecommender,
    optimizer: torch.optim.Optimizer,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    global_step: int,
    config: MultiVAELightGCNConfig,
    trial: MultiVAETrial,
) -> tuple[dict[str, float], int]:
    model.train()
    accum = {"loss": 0.0, "nll": 0.0, "kl": 0.0, "beta": 0.0}
    batches = 0
    for batch in loader:
        data = batch["data"].to(device, dtype=torch.float32)
        output = model(data, sample=True)
        beta = beta_at_step(global_step, config.total_anneal_steps, trial.beta_cap)
        terms = multivae_loss(output.logits, data, output.mu, output.logvar, beta)
        optimizer.zero_grad(set_to_none=True)
        terms["loss"].backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        global_step += 1
        batches += 1
        for name in accum:
            if name == "beta":
                accum[name] += beta
            else:
                accum[name] += float(terms[name].detach().item())
    if batches == 0:
        raise RuntimeError("training loader produced no batches")
    return {name: value / batches for name, value in accum.items()}, global_step


def _run_stage(
    views: MultiVAELightGCNViews,
    trial: MultiVAETrial,
    config: MultiVAELightGCNConfig,
    *,
    run_dir: Path,
    stage: str,
    target_epoch: int,
    train_matrix: sp.csr_matrix,
    data_hash: str,
    device: torch.device,
    validation_observed: sp.csr_matrix | None,
    validation_truth: sp.csr_matrix | None,
    previous_checkpoint: Path | None = None,
    early_stopping: bool = False,
) -> dict[str, Any]:
    candidate_dir = run_dir / "trials" / trial.identifier
    candidate_dir.mkdir(parents=True, exist_ok=True)
    result_path = candidate_dir / f"{stage}.json"
    checkpoint_path = candidate_dir / f"{stage}_last.pt"
    if result_path.exists():
        return _read_json(result_path)

    print(
        f"[{stage}] starting {trial.identifier} through epoch {target_epoch}",
        flush=True,
    )

    config_hash = stable_hash(config.as_dict())
    seed_everything(config.seed)
    model = _build_model(trial, views.n_items, config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=trial.learning_rate, weight_decay=trial.weight_decay
    )
    source_checkpoint = checkpoint_path if checkpoint_path.exists() else previous_checkpoint
    completed_epoch = 0
    global_step = 0
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    stale = 0
    stopped_early = False
    if source_checkpoint is not None and source_checkpoint.exists():
        payload = _load_checkpoint(
            source_checkpoint,
            model,
            optimizer,
            trial=trial,
            data_hash=data_hash,
            config_hash=config_hash,
        )
        completed_epoch = int(payload.get("epoch", 0))
        global_step = int(payload.get("global_step", 0))
        best_epoch = int(payload.get("best_epoch", 0))
        loaded_metrics = payload.get("best_metrics")
        if isinstance(loaded_metrics, dict) and loaded_metrics:
            best_metrics = {str(key): float(value) for key, value in loaded_metrics.items()}
        stale = int(payload.get("stale", 0))
    if early_stopping and stale >= config.patience:
        stopped_early = True

    train_loader = _loader(train_matrix, config, shuffle=True)
    curve_path = candidate_dir / f"{stage}_curve.jsonl"
    if curve_path.exists():
        existing_curve = _read_jsonl(curve_path)
        retained_curve = [
            record
            for record in existing_curve
            if int(record.get("epoch", 0)) <= completed_epoch
        ]
        if len(retained_curve) != len(existing_curve):
            _write_jsonl(curve_path, retained_curve)
    last_train_metrics: dict[str, float] = {}
    while completed_epoch < target_epoch and not stopped_early:
        train_metrics, global_step = _train_epoch(
            model,
            optimizer,
            train_loader,
            device=device,
            global_step=global_step,
            config=config,
            trial=trial,
        )
        completed_epoch += 1
        last_train_metrics = train_metrics
        validation_metrics: dict[str, float] | None = None
        should_validate = (
            validation_observed is not None
            and validation_truth is not None
            and early_stopping
            and completed_epoch % config.validation_every == 0
        )
        if should_validate:
            validation_metrics = _evaluate(
                model,
                validation_observed,
                validation_truth,
                config,
                device,
            )
            if early_stopping:
                if best_metrics is None or _selection_key(
                    validation_metrics, config.selection_k
                ) > _selection_key(best_metrics, config.selection_k):
                    best_metrics = validation_metrics
                    best_epoch = completed_epoch
                    stale = 0
                else:
                    stale += 1
                if stale >= config.patience:
                    stopped_early = True
        _append_jsonl(
            curve_path,
            {
                "epoch": completed_epoch,
                "global_step": global_step,
                "train": train_metrics,
                "validation": validation_metrics or {},
            },
        )
        if (
            completed_epoch % config.checkpoint_every == 0
            or completed_epoch >= target_epoch
            or stopped_early
        ):
            _save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                trial=trial,
                stage=stage,
                epoch=completed_epoch,
                global_step=global_step,
                data_hash=data_hash,
                config_hash=config_hash,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                stale=stale,
            )
        if stopped_early:
            break

    if validation_observed is not None and validation_truth is not None:
        if early_stopping:
            if best_metrics is None:
                raise RuntimeError(f"stage {stage} did not produce validation metrics")
            selected_metrics = best_metrics
        else:
            selected_metrics = _evaluate(
                model,
                validation_observed,
                validation_truth,
                config,
                device,
            )
    else:
        selected_metrics = {}
    result: dict[str, Any] = {
        "stage": stage,
        "trial": {"id": trial.identifier, **trial.as_dict()},
        "target_epoch": target_epoch,
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "train_metrics": last_train_metrics,
        "stopped_early": stopped_early,
    }
    if validation_observed is not None and validation_truth is not None:
        result.update(
            {
                "validation_metrics": selected_metrics,
                "best_epoch": best_epoch if early_stopping else completed_epoch,
                "best_validation_metrics": best_metrics if early_stopping else selected_metrics,
                "trained_through_epoch": completed_epoch,
            }
        )
    _write_json(result_path, result)
    metric_text = (
        f" recall@{config.selection_k}={selected_metrics[f'recall@{config.selection_k}']:.6f}"
        if selected_metrics
        else ""
    )
    print(
        f"[{stage}] completed {trial.identifier} at epoch {completed_epoch}.{metric_text}",
        flush=True,
    )
    return result


def _run_stage_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one independent trial in an isolated process with its own RNG state."""

    config = to_multvae_lightgcn_config(cast(dict[str, Any], payload["config"]))
    torch.set_num_threads(int(payload["torch_threads"]))
    data = load_lightgcn_data(Path(str(payload["data_dir"])))
    views = build_multvae_views(data)
    trial_data = cast(dict[str, Any], payload["trial"])
    hidden_values = tuple(int(value) for value in trial_data["hidden_dims"])
    if len(hidden_values) != 3:
        raise ValueError("worker trial hidden_dims must have length three")
    trial = MultiVAETrial(
        (hidden_values[0], hidden_values[1], hidden_values[2]),
        float(trial_data["dropout"]),
        float(trial_data["beta_cap"]),
        float(trial_data["learning_rate"]),
        float(trial_data["weight_decay"]),
    )
    matrix_name = str(payload["train_matrix"])
    train_matrix = views.train if matrix_name == "train" else views.combined
    validation_enabled = bool(payload["validation_enabled"])
    previous_value = payload.get("previous_checkpoint")
    return _run_stage(
        views,
        trial,
        config,
        run_dir=Path(str(payload["run_dir"])),
        stage=str(payload["stage"]),
        target_epoch=int(payload["target_epoch"]),
        train_matrix=train_matrix,
        data_hash=str(payload["data_hash"]),
        device=torch.device("cpu"),
        validation_observed=views.train if validation_enabled else None,
        validation_truth=views.validation if validation_enabled else None,
        previous_checkpoint=Path(str(previous_value)) if previous_value else None,
        early_stopping=bool(payload["early_stopping"]),
    )


def _run_stage_group(
    views: MultiVAELightGCNViews,
    trials: list[MultiVAETrial],
    config: MultiVAELightGCNConfig,
    *,
    run_dir: Path,
    stage: str,
    target_epoch: int,
    train_matrix_name: str,
    data_hash: str,
    device: torch.device,
    validation_enabled: bool,
    early_stopping: bool,
    previous_checkpoints: dict[str, Path | None] | None = None,
) -> list[dict[str, Any]]:
    """Run independent candidates concurrently while preserving deterministic results."""

    if len(trials) <= 1 or device.type != "cpu":
        train_matrix = views.train if train_matrix_name == "train" else views.combined
        return [
            _run_stage(
                views,
                trial,
                config,
                run_dir=run_dir,
                stage=stage,
                target_epoch=target_epoch,
                train_matrix=train_matrix,
                data_hash=data_hash,
                device=device,
                validation_observed=views.train if validation_enabled else None,
                validation_truth=views.validation if validation_enabled else None,
                previous_checkpoint=(previous_checkpoints or {}).get(trial.identifier),
                early_stopping=early_stopping,
            )
            for trial in trials
        ]

    worker_count = min(4, len(trials))
    torch_threads = max(1, min(4, (os.cpu_count() or 1) // worker_count))
    payloads: list[dict[str, Any]] = []
    for trial in trials:
        previous = (previous_checkpoints or {}).get(trial.identifier)
        payloads.append(
            {
                "config": config.as_dict(),
                "data_dir": str(Path(config.data_dir).resolve()),
                "run_dir": str(run_dir),
                "stage": stage,
                "target_epoch": target_epoch,
                "train_matrix": train_matrix_name,
                "data_hash": data_hash,
                "trial": trial.as_dict(),
                "validation_enabled": validation_enabled,
                "early_stopping": early_stopping,
                "previous_checkpoint": str(previous) if previous else None,
                "torch_threads": torch_threads,
            }
        )
    try:
        executor = ProcessPoolExecutor(max_workers=worker_count)
    except (NotImplementedError, OSError, PermissionError):
        # Restricted CI/container environments may not expose POSIX semaphores.
        # Keep the workflow usable there; normal desktop runs use the process pool.
        train_matrix = views.train if train_matrix_name == "train" else views.combined
        return [
            _run_stage(
                views,
                trial,
                config,
                run_dir=run_dir,
                stage=stage,
                target_epoch=target_epoch,
                train_matrix=train_matrix,
                data_hash=data_hash,
                device=device,
                validation_observed=views.train if validation_enabled else None,
                validation_truth=views.validation if validation_enabled else None,
                previous_checkpoint=(previous_checkpoints or {}).get(trial.identifier),
                early_stopping=early_stopping,
            )
            for trial in trials
        ]
    with executor:
        futures = [executor.submit(_run_stage_worker, payload) for payload in payloads]
        return [future.result() for future in futures]


def _initial_status(data_hash: str, config_hash: str) -> dict[str, Any]:
    now = _utc_now()
    return {
        "status": "incomplete",
        "phase": "initializing",
        "started_at": now,
        "updated_at": now,
        "dataset_hash": data_hash,
        "config_hash": config_hash,
        "test_accessed": False,
    }


def _set_phase(status_path: Path, status: dict[str, Any], phase: str) -> None:
    status["phase"] = phase
    status["updated_at"] = _utc_now()
    _write_json(status_path, status)


def _final_data_hash(data_hash: str) -> str:
    return stable_hash({"base_dataset_hash": data_hash, "phase": "train-plus-validation"})


def _device_only_config_change(
    previous: MultiVAELightGCNConfig, current: MultiVAELightGCNConfig
) -> bool:
    """Allow the explicit auto→MPS continuation requested by the user."""

    previous_values = previous.as_dict()
    current_values = current.as_dict()
    previous_values.pop("device", None)
    current_values.pop("device", None)
    return (
        previous.device == "auto"
        and current.device == "mps"
        and previous_values == current_values
    )


def _migrate_device_config(
    run_dir: Path,
    status_path: Path,
    status: dict[str, Any],
    previous: MultiVAELightGCNConfig,
    current: MultiVAELightGCNConfig,
    previous_hash: str,
    current_hash: str,
) -> None:
    """Rewrite only checkpoint metadata for an intentional execution-device change."""

    migrated_checkpoints = 0
    for checkpoint in sorted((run_dir / "trials").glob("*/*.pt")):
        payload = cast(
            dict[str, Any], torch.load(checkpoint, map_location="cpu", weights_only=False)
        )
        if payload.get("config_hash") != previous_hash:
            continue
        payload["config_hash"] = current_hash
        payload["config_migration"] = {
            "from_device": previous.device,
            "to_device": current.device,
            "migrated_at": _utc_now(),
        }
        temporary = checkpoint.with_suffix(checkpoint.suffix + ".migration.tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint)
        migrated_checkpoints += 1
    migration = {
        "from_config_hash": previous_hash,
        "to_config_hash": current_hash,
        "from_device": previous.device,
        "to_device": current.device,
        "checkpoint_count": migrated_checkpoints,
        "migrated_at": _utc_now(),
    }
    status["config_hash"] = current_hash
    status.setdefault("config_migrations", []).append(migration)
    status["updated_at"] = _utc_now()
    _write_json(run_dir / "resolved_config.json", current.as_dict())
    _write_json(status_path, status)


def reproduce_multvae_lightgcn(
    config: MultiVAELightGCNConfig,
) -> dict[str, Any]:
    """Run/resume sweep, final retraining, and exactly one test evaluation."""

    config.validate()
    prepared = load_lightgcn_data(Path(config.data_dir).resolve())
    views = build_multvae_views(prepared)
    run_dir = Path(config.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "run_status.json"
    config_hash = stable_hash(config.as_dict())
    status = (
        _read_json(status_path)
        if status_path.exists()
        else _initial_status(views.dataset_hash, config_hash)
    )
    if status.get("dataset_hash") != views.dataset_hash:
        raise ValueError("run directory belongs to a different LightGCN dataset")
    if status.get("config_hash") != config_hash:
        previous_config_path = run_dir / "resolved_config.json"
        if not previous_config_path.exists():
            raise ValueError("run directory belongs to a different resolved config")
        previous_config = to_multvae_lightgcn_config(_read_json(previous_config_path))
        previous_hash = stable_hash(previous_config.as_dict())
        if status.get("config_hash") != previous_hash or not _device_only_config_change(
            previous_config, config
        ):
            raise ValueError("run directory belongs to a different resolved config")
        _migrate_device_config(
            run_dir,
            status_path,
            status,
            previous_config,
            config,
            previous_hash,
            config_hash,
        )
    if status.get("status") == "verified":
        if "reason" in status:
            status.pop("reason", None)
            status["updated_at"] = _utc_now()
            _write_json(status_path, status)
        return status

    _write_json(run_dir / "resolved_config.json", config.as_dict())
    device = select_device(config.device)
    _write_json(run_dir / "environment.json", _environment(device, config))
    _write_json(
        run_dir / "provenance.json",
        {
            "dataset_hash": views.dataset_hash,
            "split_files": views.manifest.get("files", {}),
            "source": views.manifest.get("source", {}),
            "protocol": views.manifest.get("protocol"),
            "counts": views.manifest.get("counts", {}),
        },
    )
    _write_json(run_dir / "run_status.json", status)
    candidates = _trial_list(config)
    try:
        _set_phase(status_path, status, "sweep-round-1")
        round1 = _run_stage_group(
            views,
            candidates,
            config,
            run_dir=run_dir,
            stage="round1",
            target_epoch=config.round1_epochs,
            train_matrix_name="train",
            data_hash=views.dataset_hash,
            device=device,
            validation_enabled=True,
            early_stopping=False,
        )
        round1 = _rank_results(round1, config.selection_k)
        round1_finalists = round1[: config.round2_finalists]

        _set_phase(status_path, status, "sweep-round-2")
        round2_trials = [_trial_from_result(record) for record in round1_finalists]
        round2 = _run_stage_group(
            views,
            round2_trials,
            config,
            run_dir=run_dir,
            stage="round2",
            target_epoch=config.round2_epochs,
            train_matrix_name="train",
            data_hash=views.dataset_hash,
            device=device,
            validation_enabled=True,
            early_stopping=False,
            previous_checkpoints={
                trial.identifier: run_dir / "trials" / trial.identifier / "round1_last.pt"
                for trial in round2_trials
            },
        )
        round2 = _rank_results(round2, config.selection_k)
        round2_finalists = round2[: config.final_candidates]

        _set_phase(status_path, status, "sweep-finalists")
        full_trials = [_trial_from_result(record) for record in round2_finalists]
        full = _run_stage_group(
            views,
            full_trials,
            config,
            run_dir=run_dir,
            stage="full",
            target_epoch=config.max_epochs,
            train_matrix_name="train",
            data_hash=views.dataset_hash,
            device=device,
            validation_enabled=True,
            early_stopping=True,
            previous_checkpoints={
                trial.identifier: run_dir / "trials" / trial.identifier / "round2_last.pt"
                for trial in full_trials
            },
        )
        full = _rank_results(full, config.selection_k, metric_field="best_validation_metrics")
        winner = _trial_from_result(full[0])
        winner_full = full[0]
        best_epoch = int(winner_full["best_epoch"])
        selection = {
            "selected_at": _utc_now(),
            "criterion": [f"recall@{config.selection_k}", f"ndcg@{config.selection_k}"],
            "winner": {"id": winner.identifier, **winner.as_dict()},
            "best_epoch": best_epoch,
            "best_validation_metrics": winner_full["best_validation_metrics"],
            "round1_finalists": [record["trial"]["id"] for record in round1_finalists],
            "round2_finalists": [record["trial"]["id"] for record in round2_finalists],
            "stopped_early": bool(winner_full["stopped_early"]),
            "trained_through_epoch": int(winner_full["trained_through_epoch"]),
        }
        _write_json(run_dir / "selection.json", selection)

        _set_phase(status_path, status, "final-retrain")
        final_hash = _final_data_hash(views.dataset_hash)
        final = _run_stage(
            views,
            winner,
            config,
            run_dir=run_dir,
            stage="final",
            target_epoch=best_epoch,
            train_matrix=views.combined,
            data_hash=final_hash,
            device=device,
            validation_observed=None,
            validation_truth=None,
        )
        if int(final["completed_epoch"]) != best_epoch:
            raise RuntimeError("final retrain did not reach the selected epoch")
        final_checkpoint = run_dir / "trials" / winner.identifier / "final_last.pt"
        shutil.copy2(final_checkpoint, run_dir / "best.pt")
        shutil.copy2(final_checkpoint, run_dir / "last.pt")
        full_curve = run_dir / "trials" / winner.identifier / "full_curve.jsonl"
        final_curve = run_dir / "trials" / winner.identifier / "final_curve.jsonl"
        if full_curve.exists():
            shutil.copy2(full_curve, run_dir / "training_curve.jsonl")
        if final_curve.exists():
            shutil.copy2(final_curve, run_dir / "final_training_curve.jsonl")

        test_metrics_path = run_dir / "test_metrics.json"
        access_path = run_dir / "test_access.json"
        if access_path.exists() and not test_metrics_path.exists():
            raise RuntimeError("test access was recorded without metrics; refusing a second access")
        if not test_metrics_path.exists():
            _set_phase(status_path, status, "test-evaluation")
            access = {
                "accessed_at": _utc_now(),
                "after_model_selection": True,
                "selection_hash": stable_hash(selection),
                "test_edges": int(views.test.nnz),
            }
            _write_json(access_path, access)
            status["test_accessed"] = True
            _write_json(status_path, status)
            model = _build_model(winner, views.n_items, config).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=winner.learning_rate, weight_decay=winner.weight_decay
            )
            _load_checkpoint(
                final_checkpoint,
                model,
                optimizer,
                trial=winner,
                data_hash=final_hash,
                config_hash=config_hash,
            )
            test_metrics = _evaluate(model, views.combined, views.test, config, device)
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
                "winner": selection["winner"],
                "best_epoch": best_epoch,
                "validation_metrics": selection["best_validation_metrics"],
                "test_metrics": test_metrics,
                "test_accessed": True,
            }
        )
        status.pop("reason", None)
        _write_json(status_path, status)
        return status
    except Exception as error:
        status.update({"status": "incomplete", "reason": str(error), "updated_at": _utc_now()})
        _write_json(status_path, status)
        raise


def _checkpoint_model(
    checkpoint: Path,
    n_items: int,
    device: torch.device,
) -> BaseRecommender:
    payload = cast(dict[str, Any], torch.load(checkpoint, map_location=device, weights_only=False))
    model_config = dict(payload.get("model_config", {}))
    model_name = str(model_config.pop("name", "multvae"))
    model_config["n_items"] = n_items
    model = build_model(model_name, **model_config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def evaluate_existing_checkpoint(
    checkpoint: Path,
    data_dir: Path,
    *,
    config: MultiVAELightGCNConfig,
    device: torch.device,
) -> dict[str, float]:
    """Re-score the previous Multi-VAE baseline with the LightGCN metric protocol."""

    prepared = load_prepared_data(data_dir)
    model = _checkpoint_model(checkpoint, prepared.n_items, device)
    return _evaluate(model, prepared.test_fold_in, prepared.test_fold_out, config, device)


def _read_optional_json(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value).resolve()
    if not path.exists():
        return None
    return _read_json(path)


def _metric_value(metrics: dict[str, Any] | None, key: str) -> str:
    if not metrics or key not in metrics:
        return "—"
    value = metrics[key]
    return f"{float(value):.6f}" if isinstance(value, (int, float)) else str(value)


def _metric_deltas(
    metrics: dict[str, Any], reference: dict[str, Any] | None
) -> dict[str, float]:
    """Return ranking-metric deltas against an optional comparison run."""

    if reference is None:
        return {}
    return {
        key: float(metrics[key]) - float(reference[key])
        for key in sorted(metrics)
        if key in reference and key.startswith(("recall@", "ndcg@"))
    }


def _paper_reference(dataset: str) -> dict[str, Any] | None:
    if dataset != "yelp2018":
        return None
    return {
        "source": {
            "title": (
                "LightGCN: Simplifying and Powering Graph Convolution Network for "
                "Recommendation"
            ),
            "url": "https://arxiv.org/abs/2002.02126",
        },
        "dataset": "Yelp2018",
        "metrics": {
            "multvae": {"recall@20": 0.0584, "ndcg@20": 0.0450},
            "lightgcn": {"recall@20": 0.0649, "ndcg@20": 0.0530},
        },
        "note": (
            "Published Table 4 values are contextual references only; no LightGCN baseline "
            "was retrained in this workflow."
        ),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    test = cast(dict[str, Any], report["test_metrics"])
    validation = cast(dict[str, Any], report["validation_metrics"])
    selection = cast(dict[str, Any], report["selection"])
    provenance = cast(dict[str, Any], report["provenance"])
    source = cast(dict[str, Any], provenance.get("source", {}))
    artifact_sha256 = cast(dict[str, Any], provenance.get("artifact_sha256", {}))
    environment = cast(dict[str, Any], report["environment"])
    dataset = cast(dict[str, Any], report["dataset"])
    ks = [int(k) for k in report["config"]["evaluation_ks"]]
    lines = [
        f"# Multi-VAE on {dataset['name']}",
        "",
        str(report["scope_note"]),
        "",
        "## Test metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for k in ks:
        lines.append(f"| Recall@{k} | {_metric_value(test, f'recall@{k}')} |")
        lines.append(f"| NDCG@{k} | {_metric_value(test, f'ndcg@{k}')} |")
    paper_reference = report.get("paper_reference")
    if paper_reference:
        paper_metrics = cast(dict[str, Any], paper_reference["metrics"])
        lines.extend(
            [
                "",
                "## Paper reference (context only)",
                "",
                "| Method | Recall@20 | NDCG@20 |",
                "| --- | ---: | ---: |",
                f"| Mult-VAE · paper | {_metric_value(paper_metrics['multvae'], 'recall@20')} | "
                f"{_metric_value(paper_metrics['multvae'], 'ndcg@20')} |",
                f"| LightGCN · paper | {_metric_value(paper_metrics['lightgcn'], 'recall@20')} | "
                f"{_metric_value(paper_metrics['lightgcn'], 'ndcg@20')} |",
                "",
                str(paper_reference["note"]),
            ]
        )
    comparison = cast(dict[str, Any], report.get("comparison", {}))
    if any(
        comparison.get(name) is not None
        for name in ("multvae_previous_split", "lightgcn_local")
    ):
        baseline = cast(dict[str, Any] | None, comparison.get("multvae_previous_split"))
        lightgcn = cast(dict[str, Any] | None, comparison.get("lightgcn_local"))
        lines.extend(
            [
                "",
                "## Local comparison",
                "",
                "| Run | Recall@20 | NDCG@20 |",
                "| --- | ---: | ---: |",
                f"| Multi-VAE · current | {_metric_value(test, 'recall@20')} | "
                f"{_metric_value(test, 'ndcg@20')} |",
                f"| Multi-VAE · previous split | {_metric_value(baseline, 'recall@20')} | "
                f"{_metric_value(baseline, 'ndcg@20')} |",
                f"| LightGCN · same local split | {_metric_value(lightgcn, 'recall@20')} | "
                f"{_metric_value(lightgcn, 'ndcg@20')} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Validation metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for k in ks:
        lines.append(f"| Recall@{k} | {_metric_value(validation, f'recall@{k}')} |")
        lines.append(f"| NDCG@{k} | {_metric_value(validation, f'ndcg@{k}')} |")
    lines.extend(
        [
            f"| Evaluated users | {_metric_value(validation, 'evaluated_users')} |",
            f"| Truth edges | {_metric_value(validation, 'truth_edges')} |",
            "",
            "## Selection and sweep",
            "",
            f"- Selection criterion: `{selection['criterion'][0]}`, then "
            f"`{selection['criterion'][1]}`",
            f"- Best epoch: `{selection['best_epoch']}`; full-stage training reached "
            f"epoch `{selection['trained_through_epoch']}`",
            f"- Early stopping: `{selection['stopped_early']}`",
            "",
        ]
    )
    metric_headers = " | ".join(
        header for k in ks for header in (f"Recall@{k}", f"NDCG@{k}")
    )
    separator = " | ".join("---:" for _ in range(len(ks) * 2))
    lines.append(f"| Stage | Trial | Best epoch | Through | {metric_headers} |")
    lines.append(f"| --- | --- | ---: | ---: | {separator} |")
    for stage in ("round1", "round2", "full"):
        for record in report["sweep"][stage]:
            trial = cast(dict[str, Any], record["trial"])
            metrics = cast(
                dict[str, Any],
                record.get("best_validation_metrics") or record.get("validation_metrics", {}),
            )
            values = " | ".join(
                _metric_value(metrics, key)
                for k in ks
                for key in (f"recall@{k}", f"ndcg@{k}")
            )
            lines.append(
                f"| {stage} | `{trial['id']}` | {record.get('best_epoch', '—')} | "
                f"{record.get('trained_through_epoch', record.get('completed_epoch', '—'))} | "
                f"{values} |"
            )
    split_checksums = cast(dict[str, Any], provenance.get("split_file_sha256", {}))
    lines.extend(
        [
            "",
            "## Training curves",
            "",
            f"- Winner full-stage records: `{len(report.get('training_curve', []))}`",
            f"- Final retrain records: `{len(report.get('final_training_curve', []))}`",
            "",
            "## Selected configuration",
            "",
            f"```json\n{json.dumps(selection['winner'], indent=2, sort_keys=True)}\n```",
            "",
            "## Provenance",
            "",
            f"- Dataset hash: `{provenance['dataset_hash']}`",
            f"- Config hash: `{report['config_hash']}`",
            f"- Protocol: `{dataset['protocol']}`",
            f"- Dataset counts: `{json.dumps(dataset['counts'], sort_keys=True)}`",
            f"- Evaluated users: `{test.get('evaluated_users', '—')}`",
            f"- Truth edges: `{test.get('truth_edges', '—')}`",
            f"- Device: `{environment.get('device', '—')}`",
            f"- Environment: Python `{environment.get('python', '—')}`, "
            f"PyTorch `{environment.get('torch', '—')}`, NumPy `{environment.get('numpy', '—')}`, "
            f"SciPy `{environment.get('scipy', '—')}`, MPS available "
            f"`{environment.get('mps_available', '—')}`",
            f"- Source: `{json.dumps(source, sort_keys=True)}`",
            f"- Split checksums: `{json.dumps(split_checksums, sort_keys=True)}`",
            f"- Final checkpoint SHA-256: `{artifact_sha256.get('final_checkpoint', '—')}`",
            f"- Test access proof: after selection "
            f"`{provenance['test_access'].get('after_model_selection', '—')}`, selection hash "
            f"`{provenance['test_access'].get('selection_hash', '—')}`",
            f"- Duration: `{float(report.get('duration_seconds', 0.0)):.2f}` seconds",
        ]
    )
    return "\n".join(lines) + "\n"


def export_multvae_report(
    run_dir: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Verify a completed run and export aggregate JSON/Markdown reports."""

    run_dir = run_dir.resolve()
    status = _read_json(run_dir / "run_status.json")
    if status.get("status") != "verified" or status.get("phase") != "complete":
        raise ValueError("Multi-VAE LightGCN run is not verified and complete")
    resolved = _read_json(run_dir / "resolved_config.json")
    config = to_multvae_lightgcn_config(resolved)
    config.validate()
    data = load_lightgcn_data(Path(config.data_dir).resolve())
    views = build_multvae_views(data)
    if status.get("dataset_hash") != views.dataset_hash:
        raise ValueError("run and prepared-data hashes do not match")
    if status.get("config_hash") != stable_hash(config.as_dict()):
        raise ValueError("run status and resolved config hashes do not match")
    selection = _read_json(run_dir / "selection.json")
    test_metrics = _read_json(run_dir / "test_metrics.json")
    access = _read_json(run_dir / "test_access.json")
    if not access.get("after_model_selection") or access.get("selection_hash") != stable_hash(
        selection
    ):
        raise ValueError("test was not proven to occur after model selection")
    if status.get("test_metrics") != test_metrics:
        raise ValueError("run status and test metrics do not match")
    metric_keys = [key for key in test_metrics if key.startswith(("recall@", "ndcg@"))]
    if not all(math.isfinite(float(test_metrics[key])) for key in metric_keys):
        raise ValueError("test metrics are not finite")
    final_hash = _final_data_hash(views.dataset_hash)
    final_checkpoint = cast(
        dict[str, Any], torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    )
    if final_checkpoint.get("dataset_hash") != final_hash:
        raise ValueError("final checkpoint dataset hash mismatch")
    if final_checkpoint.get("config_hash") != stable_hash(config.as_dict()):
        raise ValueError("final checkpoint config hash mismatch")
    if dict(final_checkpoint.get("trial", {})).get("id") != selection["winner"]["id"]:
        raise ValueError("final checkpoint trial does not match model selection")
    if final_checkpoint.get("stage") != "final":
        raise ValueError("best checkpoint is not the final retrain artifact")
    if int(final_checkpoint.get("epoch", -1)) != int(selection["best_epoch"]):
        raise ValueError("final checkpoint epoch does not match model selection")

    started_at = datetime.fromisoformat(str(status["started_at"]))
    completed_at = datetime.fromisoformat(str(status["completed_at"]))
    device = select_device(config.device)
    baseline: dict[str, Any] | None = None
    if config.baseline_checkpoint and config.baseline_data_dir:
        baseline_checkpoint = Path(config.baseline_checkpoint).resolve()
        baseline_data_dir = Path(config.baseline_data_dir).resolve()
        if baseline_checkpoint.exists() and baseline_data_dir.exists():
            baseline_metrics = evaluate_existing_checkpoint(
                baseline_checkpoint,
                baseline_data_dir,
                config=config,
                device=device,
            )
            baseline = {key: value for key, value in baseline_metrics.items()}
    lightgcn = _read_optional_json(config.lightgcn_metrics_path)
    dataset_key = str(views.manifest.get("dataset", "unknown")).lower()
    dataset_name = "Yelp2018" if dataset_key == "yelp2018" else "MovieLens-20M"
    if dataset_key == "yelp2018":
        experiment = "Multi-VAE retraining on the official Yelp2018 LightGCN split"
        scope_note = (
            "Official LightGCN-PyTorch train/test files are used. Validation is derived only "
            "from official train; the official test fold is accessed once after selection."
        )
        limitations = [
            "The paper-reference values are contextual and are not a fresh LightGCN baseline.",
            "The internal validation fold is a deterministic 10% carve-out from official train.",
            "Runtime and device measurements are specific to this host.",
        ]
    else:
        experiment = "Multi-VAE transfer experiment on the LightGCN MovieLens-20M split"
        scope_note = (
            "This is a MovieLens-20M transfer experiment using the repository's per-user "
            "random 80/10/10 LightGCN split; it is not a direct replication of the paper's "
            "Gowalla, Yelp2018, or Amazon-Book experiments."
        )
        limitations = [
            "MovieLens-20M is not one of the three datasets in the LightGCN paper.",
            "The local comparison uses different catalogs and user populations when included.",
            "Runtime and device measurements are specific to this host.",
        ]
    winner_id = str(selection["winner"]["id"])
    training_curve = _read_jsonl(run_dir / "training_curve.jsonl")
    final_training_curve = _read_jsonl(run_dir / "final_training_curve.jsonl")
    if not training_curve:
        training_curve = _read_jsonl(run_dir / "trials" / winner_id / "full_curve.jsonl")
    if not final_training_curve:
        final_training_curve = _read_jsonl(run_dir / "trials" / winner_id / "final_curve.jsonl")
    counts = dict(views.manifest.get("counts", {}))
    report: dict[str, Any] = {
        "schema_version": 2,
        "status": "verified",
        "experiment": experiment,
        "scope_note": scope_note,
        "generated_at": _utc_now(),
        "started_at": status["started_at"],
        "completed_at": status["completed_at"],
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "config_hash": stable_hash(config.as_dict()),
        "config_migrations": status.get("config_migrations", []),
        "test_metrics": test_metrics,
        "validation_metrics": selection["best_validation_metrics"],
        "selection": selection,
        "training_curve": training_curve,
        "final_training_curve": final_training_curve,
        "dataset": {
            "name": dataset_name,
            "key": dataset_key,
            "protocol": views.manifest.get("protocol"),
            "seed": views.manifest.get("seed"),
            "counts": counts,
        },
        "config": config.as_dict(),
        "environment": _read_json(run_dir / "environment.json"),
        "provenance": {
            "dataset_hash": views.dataset_hash,
            "split_file_sha256": views.manifest.get("files", {}),
            "source": views.manifest.get("source", {}),
            "test_access": access,
            "artifact_sha256": {
                "final_checkpoint": sha256_file(run_dir / "best.pt"),
                "resolved_config": sha256_file(run_dir / "resolved_config.json"),
                "selection": sha256_file(run_dir / "selection.json"),
            },
        },
        "comparison": {
            "multvae_previous_split": baseline,
            "lightgcn_local": lightgcn,
            "deltas": {
                "vs_multvae_previous_split": _metric_deltas(test_metrics, baseline),
                "vs_lightgcn_local": _metric_deltas(test_metrics, lightgcn),
            },
        },
        "paper_reference": _paper_reference(dataset_key),
        "sweep": {
            "round1": [
                _read_json(path)
                for path in sorted((run_dir / "trials").glob("*/round1.json"))
            ],
            "round2": [
                _read_json(path)
                for path in sorted((run_dir / "trials").glob("*/round2.json"))
            ],
            "full": [
                _read_json(path)
                for path in sorted((run_dir / "trials").glob("*/full.json"))
            ],
        },
        "limitations": limitations,
    }
    report["report_hash"] = stable_hash(report)
    _write_json(run_dir / "report.json", report)
    (run_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    destination = output_path
    if destination is None and config.report_output:
        destination = Path(config.report_output)
    if destination is not None:
        destination = destination.resolve()
        _write_json(destination, report)
    return report
