"""Identical candidate sampling and stable masked implicit losses."""

import numpy as np
import torch
from torch.nn import functional as F

from multvae.data import Dataset


def epoch_samples(data: Dataset, seed: int, epoch: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw a reproducible shuffled user traversal and one positive per user."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    users = rng.permutation(data.n_users)
    starts = data.train.offsets[users]
    lengths = data.train.offsets[users + 1] - starts
    if np.any(lengths < 2):
        raise ValueError("Leave-one-out profiles need at least two train positives per user")
    positions = starts + np.floor(rng.random(len(users)) * lengths).astype(np.int64)
    return users, data.train.items[positions].astype(np.int64)


def candidates(data: Dataset, users: np.ndarray, positives: np.ndarray,
               seed: int, epoch: int, step: int, negative_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a shared pool and mask train positives without consulting held-out labels."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, step, 2301]))
    shared = rng.choice(data.n_items, min(negative_count, data.n_items), replace=False)
    pool = np.unique(np.r_[positives, shared])
    keys = users[:, None].astype(np.int64) * data.n_items + pool
    positions = np.searchsorted(data.train.keys, keys)
    known = ((positions < len(data.train.keys)) &
             (data.train.keys[np.minimum(positions, len(data.train.keys) - 1)] == keys))
    negative = ~known
    if np.any(negative.sum(1) == 0):
        raise ValueError("A user has no valid negative in the sampled pool")
    return pool, np.searchsorted(pool, positives), negative


def implicit_loss(scores: torch.Tensor, targets: torch.Tensor, negative: torch.Tensor,
                  loss_name: str, temperature: float) -> torch.Tensor:
    """Compute user-averaged InfoNCE or mean-pair BPR on the same score matrix."""
    if temperature <= 0 or loss_name not in ("bpr", "infonce"):
        raise ValueError("Require positive temperature and bpr/infonce")
    positive = scores.gather(1, targets[:, None]) / temperature
    scaled = scores / temperature
    if loss_name == "bpr":
        pair = F.softplus(scaled - positive)
        return ((pair * negative).sum(1) / negative.sum(1)).mean()
    admitted = negative.clone()
    admitted.scatter_(1, targets[:, None], True)
    return (torch.logsumexp(scaled.masked_fill(~admitted, -torch.inf), dim=1) - positive[:, 0]).mean()
