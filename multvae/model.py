"""Mult-VAE PR architecture and user-averaged multinomial objective."""

import torch
from torch import nn
from torch.nn import functional as F


class MultVAE(nn.Module):
    """Encode binary histories into a diagonal Gaussian and decode item logits."""

    def __init__(
        self, n_items: int, hidden_dim: int = 600, latent_dim: int = 200,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if min(n_items, hidden_dim, latent_dim) < 1 or not 0 <= dropout < 1:
            raise ValueError("Dimensions must be positive and dropout must be in [0, 1)")
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.encoder = nn.Linear(n_items, hidden_dim)
        self.posterior = nn.Linear(hidden_dim, 2 * latent_dim)
        self.decoder = nn.Linear(latent_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_items)
        for layer in (self.encoder, self.posterior, self.decoder, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.trunc_normal_(layer.bias, std=0.001, a=-0.002, b=0.002)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits, posterior mean and log variance; sample only in training."""
        normalized = F.normalize(history, p=2, dim=1)
        noisy = F.dropout(normalized, p=self.dropout, training=self.training)
        mean, logvar = self.posterior(torch.tanh(self.encoder(noisy))).chunk(2, dim=1)
        latent = mean
        if self.training:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        logits = self.output(torch.tanh(self.decoder(latent)))
        return logits, mean, logvar


def objective(
    history: torch.Tensor, logits: torch.Tensor, mean: torch.Tensor,
    logvar: torch.Tensor, beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total loss, reconstruction NLL and unweighted KL, averaged by user."""
    nll = -(history * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).sum(dim=1).mean()
    return nll + beta * kl, nll, kl
