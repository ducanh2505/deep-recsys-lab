"""Fail-closed, resumable checkpoints for LightGCN experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from ..reproducibility import capture_rng_state, restore_rng_state
from .evaluation import BPRSampler
from .model import LightGCN

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: Path,
    model: LightGCN,
    optimizer: torch.optim.Optimizer,
    sampler: BPRSampler,
    *,
    step: int,
    dataset_hash: str,
    config_hash: str,
    config: dict[str, Any],
    training_state: dict[str, Any] | None = None,
) -> None:
    """Atomically store model, optimizer, RNG, sampler, and provenance state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "step": step,
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "sampler_state": sampler.state_dict(),
        "training_state": training_state or {},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: LightGCN,
    optimizer: torch.optim.Optimizer,
    sampler: BPRSampler,
    *,
    dataset_hash: str,
    config_hash: str,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint only when dataset and resolved config hashes match."""

    payload = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))
    if int(payload.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("unsupported LightGCN checkpoint version")
    if payload.get("dataset_hash") != dataset_hash:
        raise ValueError("checkpoint dataset hash mismatch")
    if payload.get("config_hash") != config_hash:
        raise ValueError("checkpoint config hash mismatch")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    sampler.load_state_dict(payload["sampler_state"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload
