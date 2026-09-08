"""Pairwise matrix factorization and two-tower model plugins."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from recsys.artifacts import RuntimeSpec
from recsys.core.types import QueryMode

from .base import ModelPlugin, TrainingContext
from .sampling import PairwiseSampler


class BPRPlugin(ModelPlugin):
    name = "bpr"

    def train(self, context: TrainingContext) -> RuntimeSpec:
        torch.manual_seed(context.training.seed)
        device = torch.device(context.training.device)
        dimension = int(context.parameters.get("embedding_dim", 32))
        users = nn.Embedding(context.data.shape[0], dimension, device=device)
        items = nn.Embedding(context.data.shape[1], dimension, device=device)
        nn.init.normal_(users.weight, std=0.02)
        nn.init.normal_(items.weight, std=0.02)
        optimizer = torch.optim.Adam(
            [*users.parameters(), *items.parameters()], lr=context.training.learning_rate
        )
        sampler = PairwiseSampler(context.data.train, context.training.seed)
        batch_size = max(1, context.training.batch_size)
        steps = int(context.parameters.get("steps", max(1, context.data.train.nnz // batch_size)))
        for _ in range(max(1, context.training.epochs) * steps):
            user_ids, positive_ids, negative_ids = sampler.sample(batch_size)
            user = users(torch.from_numpy(user_ids).to(device))
            positive = items(torch.from_numpy(positive_ids).to(device))
            negative = items(torch.from_numpy(negative_ids).to(device))
            loss = -functional.logsigmoid(
                (user * positive).sum(1) - (user * negative).sum(1)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        return RuntimeSpec(
            plugin=self.name,
            runtime="embedding",
            capabilities=(QueryMode.KNOWN_USER, QueryMode.HISTORY),
            arrays={
                "user_embeddings": users.weight.detach().cpu().numpy().astype(np.float32),
                "item_embeddings": items.weight.detach().cpu().numpy().astype(np.float32),
            },
            metadata={"history_profile": "mean"},
        )


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, dimension: int) -> None:
        super().__init__()
        hidden = max(dimension * 2, 8)
        self.user_features = nn.Embedding(n_users, hidden)
        self.item_features = nn.Embedding(n_items, hidden)
        self.user_tower = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, dimension)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, dimension)
        )
        nn.init.normal_(self.user_features.weight, std=0.02)
        nn.init.normal_(self.item_features.weight, std=0.02)

    def encode_users(self, identifiers: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.user_tower(self.user_features(identifiers)), dim=1)

    def encode_items(self, identifiers: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.item_tower(self.item_features(identifiers)), dim=1)


class TwoTowerPlugin(ModelPlugin):
    name = "two_tower"

    def train(self, context: TrainingContext) -> RuntimeSpec:
        torch.manual_seed(context.training.seed)
        device = torch.device(context.training.device)
        dimension = int(context.parameters.get("embedding_dim", 32))
        model = TwoTower(*context.data.shape, dimension).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=context.training.learning_rate)
        sampler = PairwiseSampler(context.data.train, context.training.seed)
        batch_size = max(1, context.training.batch_size)
        steps = int(context.parameters.get("steps", max(1, context.data.train.nnz // batch_size)))
        model.train()
        for _ in range(max(1, context.training.epochs) * steps):
            user_ids, positive_ids, negative_ids = sampler.sample(batch_size)
            user = model.encode_users(torch.from_numpy(user_ids).to(device))
            positive = model.encode_items(torch.from_numpy(positive_ids).to(device))
            negative = model.encode_items(torch.from_numpy(negative_ids).to(device))
            loss = -functional.logsigmoid(
                (user * positive).sum(1) - (user * negative).sum(1)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            users = model.encode_users(torch.arange(context.data.shape[0], device=device))
            items = model.encode_items(torch.arange(context.data.shape[1], device=device))
        return RuntimeSpec(
            plugin=self.name,
            runtime="embedding",
            capabilities=(QueryMode.KNOWN_USER, QueryMode.HISTORY),
            arrays={
                "user_embeddings": users.cpu().numpy().astype(np.float32),
                "item_embeddings": items.cpu().numpy().astype(np.float32),
            },
            metadata={"history_profile": "mean", "architecture": "two_tower"},
        )
