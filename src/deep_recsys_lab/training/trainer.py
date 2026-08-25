"""Small deterministic trainer for Multi-VAE."""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from ..config import AppConfig, resolved_config_dict
from ..evaluation.evaluator import evaluate_model
from ..model.base import BaseRecommender
from ..reproducibility import seed_everything
from .checkpoint import load_checkpoint, save_checkpoint
from .loss import multivae_loss
from .schedule import beta_at_step


def _environment_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        ),
    }


class Trainer:
    """Train a registered recommender and persist resumable run artifacts."""

    def __init__(
        self,
        model: BaseRecommender,
        train_loader: Iterable[Any],
        validation_loader: Iterable[Any],
        *,
        output_dir: str | Path,
        device: torch.device | str = "cpu",
        config: AppConfig | dict[str, Any] | None = None,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        beta_cap: float = 0.2,
        total_anneal_steps: int = 200_000,
        validation_ks: tuple[int, ...] = (20, 50, 100),
        max_train_batches: int | None = None,
        max_eval_batches: int | None = None,
        gradient_clip_norm: float | None = None,
        resume_from: str | Path | None = None,
        seed: int = 98_765,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.config = config
        self.epochs = int(epochs)
        self.beta_cap = float(beta_cap)
        self.total_anneal_steps = int(total_anneal_steps)
        self.validation_ks = tuple(validation_ks)
        self.max_train_batches = max_train_batches
        self.max_eval_batches = max_eval_batches
        self.gradient_clip_norm = gradient_clip_norm
        self.resume_from = Path(resume_from) if resume_from else None
        self.seed = int(seed)

    def _write_run_metadata(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config_dict = resolved_config_dict(self.config) if self.config is not None else {}
        (self.output_dir / "resolved_config.json").write_text(
            json.dumps(config_dict, sort_keys=True, indent=2, default=str), encoding="utf-8"
        )
        try:
            import yaml

            (self.output_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(config_dict, sort_keys=True), encoding="utf-8"
            )
        except ImportError:  # pragma: no cover - Hydra installs PyYAML
            pass
        (self.output_dir / "environment.json").write_text(
            json.dumps(_environment_metadata(self.device), sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def run(self) -> list[dict[str, float]]:
        """Train through the configured epoch count and return JSONL records."""

        seed_everything(self.seed)
        self._write_run_metadata()
        start_epoch = 0
        global_step = 0
        history: list[dict[str, float]] = []
        if self.resume_from is not None:
            payload = load_checkpoint(
                self.resume_from,
                self.model,
                self.optimizer,
                device=self.device,
                restore_rng=True,
            )
            start_epoch = int(payload["epoch"]) + 1
            global_step = int(payload["global_step"])
        metrics_path = self.output_dir / "metrics.jsonl"
        if self.resume_from is None:
            metrics_path.unlink(missing_ok=True)

        writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        except (ImportError, ModuleNotFoundError):
            # Training remains usable in the minimal serving installation.
            writer = None

        best_ndcg = float("-inf")
        best_path = self.output_dir / "best.pt"
        if best_path.exists() and self.resume_from is not None:
            previous = load_checkpoint(best_path, self.model, device=self.device, restore_rng=False)
            best_ndcg = float(previous.get("metrics", {}).get("ndcg@100", float("-inf")))
            load_checkpoint(
                self.resume_from, self.model, self.optimizer, device=self.device, restore_rng=True
            )

        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            accum = {"loss": 0.0, "nll": 0.0, "kl": 0.0}
            batches = 0
            for batch_number, batch in enumerate(self.train_loader):
                if self.max_train_batches is not None and batch_number >= self.max_train_batches:
                    break
                data = batch["data"].to(self.device, dtype=torch.float32)
                output = self.model(data, sample=True)
                beta = beta_at_step(global_step, self.total_anneal_steps, self.beta_cap)
                terms = multivae_loss(output.logits, data, output.mu, output.logvar, beta)
                self.optimizer.zero_grad(set_to_none=True)
                terms["loss"].backward()  # type: ignore[no-untyped-call]
                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
                global_step += 1
                batches += 1
                for name in accum:
                    accum[name] += float(terms[name].detach().item())
            record: dict[str, float] = {
                "epoch": float(epoch),
                "global_step": float(global_step),
                "beta": beta_at_step(global_step, self.total_anneal_steps, self.beta_cap),
                **{f"train/{name}": value / max(1, batches) for name, value in accum.items()},
            }
            if epoch % max(1, self._validation_every()) == 0:
                validation = evaluate_model(
                    self.model,
                    self.validation_loader,
                    device=self.device,
                    ks=self.validation_ks,
                    mask_seen=True,
                    max_batches=self.max_eval_batches,
                )
                record.update({f"validation/{key}": value for key, value in validation.items()})
                ndcg = validation.get(
                    "ndcg@100", validation.get(f"ndcg@{max(self.validation_ks)}", 0.0)
                )
                if ndcg > best_ndcg or not best_path.exists():
                    best_ndcg = ndcg
                    save_checkpoint(
                        best_path,
                        self.model,
                        self.optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        beta=record["beta"],
                        metrics=validation,
                        config=resolved_config_dict(self.config) if self.config is not None else {},
                    )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            history.append(record)
            if writer is not None:
                for key, value in record.items():
                    writer.add_scalar(key, value, global_step)
            save_checkpoint(
                self.output_dir / "last.pt",
                self.model,
                self.optimizer,
                epoch=epoch,
                global_step=global_step,
                beta=record["beta"],
                metrics=record,
                config=resolved_config_dict(self.config) if self.config is not None else {},
            )
        if writer is not None:
            writer.close()
        return history

    def _validation_every(self) -> int:
        if isinstance(self.config, AppConfig):
            return self.config.trainer.validation_every
        if isinstance(self.config, dict):
            return int(self.config.get("trainer", {}).get("validation_every", 1))
        return 1


def train_model(
    model: BaseRecommender,
    train_loader: Iterable[Any],
    validation_loader: Iterable[Any],
    *,
    epochs: int = 200,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
    beta_cap: float = 0.2,
    anneal_steps: int = 200_000,
    checkpoint_path: str | Path | None = None,
    history_path: str | Path | None = None,
    likelihood: str = "multinomial",
) -> list[dict[str, float]]:
    """Compatibility wrapper for simple notebook-style training calls."""

    if likelihood not in {"multinomial", "multivae"}:
        raise ValueError("Multi-VAE uses the multinomial likelihood; Gaussian/BCE are unsupported")
    trainer = Trainer(
        model,
        train_loader,
        validation_loader,
        output_dir=Path(checkpoint_path).parent if checkpoint_path else Path("outputs/multvae"),
        device=device,
        epochs=epochs,
        beta_cap=beta_cap,
        total_anneal_steps=anneal_steps,
    )
    if optimizer is not None:
        trainer.optimizer = optimizer
    history = trainer.run()
    if checkpoint_path is not None and (trainer.output_dir / "best.pt").exists():
        (trainer.output_dir / "best.pt").replace(checkpoint_path)
    if history_path is not None:
        destination = Path(history_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for row in history for key in row})
        lines = [",".join(keys)]
        lines.extend(",".join(str(row.get(key, "")) for key in keys) for row in history)
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return history
