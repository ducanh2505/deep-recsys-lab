"""Device selection shared by training and evaluation."""

from __future__ import annotations

from typing import Literal

import torch

DeviceName = Literal["auto", "cpu", "cuda", "mps"]


def select_device(requested: str = "auto") -> torch.device:
    """Return a usable torch device using the documented priority order."""

    name = requested.lower()
    if name not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but the PyTorch MPS backend is unavailable")
    return torch.device(name)
