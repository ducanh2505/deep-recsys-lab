"""KL annealing schedule."""

from __future__ import annotations


def beta_at_step(global_step: int, total_anneal_steps: int = 200_000, cap: float = 0.2) -> float:
    """Linearly ramp beta from zero to cap over the requested update count."""

    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if total_anneal_steps < 0:
        raise ValueError("total_anneal_steps must be non-negative")
    if total_anneal_steps <= 0:
        return float(cap)
    return float(min(cap, cap * global_step / total_anneal_steps))
