"""Paths, fingerprints and atomic experiment artifacts."""

import hashlib
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
from typing import Iterator

import torch

from multvae.train import (_cpu_tree, environment as _environment, memory_snapshot as _memory_snapshot,
                          save_checkpoint, write_json)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/content_multvae_ml20m"
DATA = ROOT / "data/processed/ml20m_lightgcn"
CATALOG = ROOT / "data/processed/movie_content/catalog.parquet"
BASELINE = ROOT / "artifacts/multvae_ml20m/baseline_mps_seed42/best.pt"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


@contextmanager
def exclusive_lock(path: Path, wait: bool = False) -> Iterator[None]:
    """Prevent concurrent writers without stale lock-file recovery heuristics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is already writing {path.parent}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    """Hash canonical JSON metadata."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def code_hash() -> str:
    """Fingerprint code that affects training or evaluation."""
    names = ("common", "data", "encoder", "objective", "train", "evaluate")
    return fingerprint({name: sha256(Path(__file__).with_name(name + ".py")) for name in names})


def device_for(name: str) -> torch.device:
    """Select the requested accelerator without silent fallback."""
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA must be available; install CUDA-enabled PyTorch and check the GPU driver")
    if device.type == "mps" and (not torch.backends.mps.is_available()
                               or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"):
        raise RuntimeError("MPS must be available and CPU fallback disabled")
    return device


def environment() -> dict:
    """Record software and hardware, including the CUDA runtime and GPU models."""
    info = _environment()
    info.update({"cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda})
    if info["cuda_available"]:
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def memory_snapshot(device: torch.device) -> dict[str, float]:
    """Measure process/MPS memory and CUDA allocator usage in MiB."""
    value = _memory_snapshot(device)
    if device.type == "cuda":
        value.update({"cuda_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
                      "cuda_reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
                      "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                      "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20})
    return value


def emit(directory: Path, event: str, **values: object) -> None:
    """Append an English JSON progress event."""
    line = json.dumps({"event": event, "time_unix": time.time(), **values}, allow_nan=False)
    print(line, flush=True)
    with (directory / "run.log").open("a") as stream:
        stream.write(line + "\n")


def rng_state(device: torch.device) -> dict:
    """Capture torch RNG state for interruption recovery."""
    value = {"cpu": torch.get_rng_state()}
    if device.type == "mps":
        value["mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        value["cuda"] = torch.cuda.get_rng_state(device)
    return value


def restore_rng(value: dict, device: torch.device) -> None:
    """Restore RNG state after model and optimizer construction."""
    torch.set_rng_state(value["cpu"])
    if device.type == "mps":
        torch.mps.set_rng_state(value["mps"])
    elif device.type == "cuda":
        torch.cuda.set_rng_state(value["cuda"], device)
