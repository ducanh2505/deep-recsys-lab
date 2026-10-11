"""Small runtime snapshots for resumable LightGCN training."""

import platform
import resource

import torch


def torch_rng_state(device: torch.device) -> dict:
    """Capture CPU and active-accelerator RNG state without drawing randomness."""
    state = {"cpu": torch.get_rng_state()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_torch_rng(state: dict, device: torch.device) -> None:
    """Restore randomness after constructing model and optimizer instances."""
    torch.set_rng_state(state["cpu"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps"])
    elif device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def memory_snapshot(device: torch.device) -> dict[str, float]:
    """Record process peak RSS plus allocator memory on the active accelerator."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    values = {"process_peak_rss_mib": rss / (2**20 if platform.system() == "Darwin" else 1024)}
    if device.type == "mps":
        values.update({"mps_allocated_mib": torch.mps.current_allocated_memory() / 2**20,
                       "mps_driver_mib": torch.mps.driver_allocated_memory() / 2**20})
    elif device.type == "cuda":
        values.update({"cuda_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
                       "cuda_reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
                       "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20})
    return values
