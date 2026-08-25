"""Versioned checkpoint save/load with exact continuation state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from ..reproducibility import capture_rng_state, restore_rng_state


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    epoch: int = 0,
    global_step: int = 0,
    beta: float = 0.0,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Save model, optimizer, annealing position, and RNG state atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    architecture_fn = getattr(model, "architecture_config", None)
    architecture = architecture_fn() if callable(architecture_fn) else {}
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "model_state": model.state_dict(),
        "model_config": architecture,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "beta": float(beta),
        "metrics": metrics or {},
        "config": config or {},
        "rng_state": capture_rng_state(),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: torch.device | str = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a full training checkpoint and return its continuation metadata."""

    payload = cast(dict[str, Any], torch.load(path, map_location=device, weights_only=False))
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported checkpoint schema")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng and payload.get("rng_state"):
        restore_rng_state(payload["rng_state"])
    return payload
