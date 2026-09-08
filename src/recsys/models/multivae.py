"""MultiVAE model plugin with a portable ONNX runtime artifact."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from recsys.artifacts import RuntimeSpec
from recsys.core.types import QueryMode

from .base import ModelPlugin, TrainingContext


class MultiVAE(nn.Module):
    def __init__(
        self,
        n_items: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if n_items < 1 or hidden_dim < 1 or latent_dim < 1:
            raise ValueError("MultiVAE dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.n_items = n_items
        self.dropout = dropout
        self.encoder = nn.Linear(n_items, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, n_items)
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, interactions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = functional.normalize(interactions.float(), p=2, dim=1)
        hidden = torch.tanh(
            self.encoder(functional.dropout(normalized, self.dropout, self.training))
        )
        return self.mu(hidden), self.logvar(hidden)

    def forward(
        self, interactions: torch.Tensor, *, sample: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(interactions)
        should_sample = self.training if sample is None else sample
        latent = mean
        if should_sample:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        return self.decoder(latent), mean, log_variance


class _InferenceModule(nn.Module):
    def __init__(self, model: MultiVAE) -> None:
        super().__init__()
        self.model = model

    def forward(self, interactions: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.model(interactions, sample=False)
        return cast(torch.Tensor, logits)


class MultiVAEPlugin(ModelPlugin):
    name = "multivae"

    def train(self, context: TrainingContext) -> RuntimeSpec:
        torch.manual_seed(context.training.seed)
        device = torch.device(context.training.device)
        parameters = context.parameters
        model = MultiVAE(
            context.data.shape[1],
            hidden_dim=int(parameters.get("hidden_dim", 64)),
            latent_dim=int(parameters.get("latent_dim", 32)),
            dropout=float(parameters.get("dropout", 0.2)),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=context.training.learning_rate)
        dense = context.data.train.toarray().astype(np.float32, copy=False)
        rng = np.random.default_rng(context.training.seed)
        batch_size = max(1, min(context.training.batch_size, len(dense)))
        model.train()
        for epoch in range(context.training.epochs):
            for start in range(0, len(dense), batch_size):
                indices = rng.permutation(len(dense))[start : start + batch_size]
                target = torch.from_numpy(dense[indices]).to(device)
                logits, mean, log_variance = model(target)
                negative_log_likelihood = (
                    -(functional.log_softmax(logits, dim=1) * target).sum(dim=1).mean()
                )
                divergence = (
                    -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(dim=1).mean()
                )
                beta = min(0.2, (epoch + 1) / max(context.training.epochs, 1) * 0.2)
                loss = negative_log_likelihood + beta * divergence
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval().cpu()
        context.scratch.mkdir(parents=True, exist_ok=True)
        onnx_path = context.scratch / "model.onnx"
        example = torch.zeros((1, context.data.shape[1]), dtype=torch.float32)
        torch.onnx.export(
            _InferenceModule(model),
            (example,),
            onnx_path,
            input_names=["interactions"],
            output_names=["scores"],
            dynamic_axes={"interactions": {0: "batch"}, "scores": {0: "batch"}},
            opset_version=20,
            dynamo=False,
        )
        return RuntimeSpec(
            plugin=self.name,
            runtime="onnx_dense",
            capabilities=(QueryMode.KNOWN_USER, QueryMode.HISTORY),
            arrays={},
            files={"model.onnx": onnx_path},
            metadata={"input": "interactions", "output": "scores"},
        )
