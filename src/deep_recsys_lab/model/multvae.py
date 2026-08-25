"""The Multi-VAE-PR architecture from Liang et al. (2018)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseRecommender, ModelOutput
from .registry import register_model


@register_model("multvae")
class MultiVAE(BaseRecommender):
    """Variational autoencoder with multinomial output for implicit feedback.

    The default dimensions are ``n_items -> 600 -> 200 -> 600 -> n_items``;
    the encoder uses tanh activations and a 0.5 input dropout. During training
    the posterior is reparameterized; evaluation uses the posterior mean.
    """

    def __init__(
        self,
        n_items: int,
        hidden_dims: Sequence[int] = (600, 200, 600),
        latent_dim: int = 200,
        dropout: float = 0.5,
        input_normalization: str = "l2",
        *,
        hidden: int | None = None,
        dimz: int | None = None,
        p: float | None = None,
    ) -> None:
        super().__init__()
        if n_items < 1:
            raise ValueError("n_items must be positive")
        if hidden is not None:
            hidden_dims = (hidden, dimz or latent_dim, hidden)
        if dimz is not None:
            latent_dim = dimz
        if p is not None:
            dropout = p
        if len(hidden_dims) != 3:
            raise ValueError("hidden_dims must be [encoder_hidden, latent_hidden, decoder_hidden]")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.n_items = int(n_items)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)
        self.input_normalization = input_normalization

        encoder_hidden, encoder_latent, decoder_hidden = self.hidden_dims
        # encoder_latent is kept in the config to make the paper's 600 -> 200
        # shape explicit; latent_dim is normally the same value.
        if encoder_latent != self.latent_dim:
            raise ValueError("hidden_dims[1] must equal latent_dim")
        self.encoder = nn.Sequential(
            nn.Linear(self.n_items, encoder_hidden),
            nn.Tanh(),
            nn.Linear(encoder_hidden, encoder_latent),
            nn.Tanh(),
        )
        self.mu_layer = nn.Linear(encoder_latent, self.latent_dim)
        self.logvar_layer = nn.Linear(encoder_latent, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, decoder_hidden),
            nn.Tanh(),
            nn.Linear(decoder_hidden, self.n_items),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _normalize_input(self, interactions: torch.Tensor) -> torch.Tensor:
        if self.input_normalization == "l2":
            return F.normalize(interactions, p=2, dim=1)
        if self.input_normalization == "row_sum":
            denom = interactions.sum(dim=1, keepdim=True).clamp_min(1.0)
            return interactions / denom
        if self.input_normalization in {"none", "identity"}:
            return interactions
        raise ValueError(f"unsupported input_normalization={self.input_normalization!r}")

    def forward(self, interactions: torch.Tensor, sample: bool | None = None) -> ModelOutput:
        if interactions.ndim != 2 or interactions.shape[1] != self.n_items:
            raise ValueError(f"expected [batch, {self.n_items}] interactions")
        x = interactions.float()
        x = self._normalize_input(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        hidden = self.encoder(x)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        should_sample = self.training if sample is None else sample
        if should_sample:
            std = torch.exp(0.5 * logvar)
            latent = mu + torch.randn_like(std) * std
        else:
            latent = mu
        return ModelOutput(logits=self.decoder(latent), mu=mu, logvar=logvar)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": "multvae",
            "n_items": self.n_items,
            "hidden_dims": list(self.hidden_dims),
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "input_normalization": self.input_normalization,
        }
