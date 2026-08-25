"""The multinomial negative log-likelihood plus annealed KL objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multinomial_nll(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the mean negative multinomial log-likelihood."""

    if logits.shape != target.shape:
        raise ValueError("logits and target must have equal shapes")
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target.float() * log_probs).sum(dim=-1).mean()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Return mean KL(q(z|x)||N(0,I)) over a batch."""

    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have equal shapes")
    return (-0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1)).mean()


def multivae_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Return total, reconstruction, and KL terms for logging."""

    nll = multinomial_nll(logits, target)
    kl = kl_divergence(mu, logvar)
    return {"loss": nll + float(beta) * kl, "nll": nll, "kl": kl}
