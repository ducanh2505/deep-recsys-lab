"""LightGCN model and plugin with embedding-only served artifacts."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional

from recsys.artifacts import RuntimeSpec
from recsys.core.types import QueryMode

from .base import ModelPlugin, TrainingContext
from .sampling import PairwiseSampler


def normalized_adjacency(interactions: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    matrix = interactions.tocoo(copy=False)
    n_users, n_items = interactions.shape
    user_degree = np.asarray(interactions.getnnz(axis=1), dtype=np.float64)
    item_degree = np.asarray(interactions.getnnz(axis=0), dtype=np.float64)
    user_scale = np.divide(
        1.0, np.sqrt(user_degree), out=np.zeros_like(user_degree), where=user_degree > 0
    )
    item_scale = np.divide(
        1.0, np.sqrt(item_degree), out=np.zeros_like(item_degree), where=item_degree > 0
    )
    weights = (user_scale[matrix.row] * item_scale[matrix.col]).astype(np.float32)
    rows = np.concatenate((matrix.row, n_users + matrix.col)).astype(np.int64)
    columns = np.concatenate((n_users + matrix.col, matrix.row)).astype(np.int64)
    indices = torch.from_numpy(np.stack((rows, columns)))
    values = torch.from_numpy(np.concatenate((weights, weights)))
    return (
        torch.sparse_coo_tensor(
            indices, values, (n_users + n_items, n_users + n_items), check_invariants=False
        )
        .coalesce()
        .to(device)
    )


class LightGCN(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        layers: int,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("LightGCN layers must be positive")
        self.n_users = n_users
        self.layers = layers
        self.users = nn.Embedding(n_users, embedding_dim)
        self.items = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.users.weight)
        nn.init.xavier_uniform_(self.items.weight)
        self.register_buffer("adjacency", adjacency, persistent=False)

    def embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        current = torch.cat((self.users.weight, self.items.weight))
        values = [current]
        for _ in range(self.layers):
            current = torch.sparse.mm(self.adjacency, current)
            values.append(current)
        combined = torch.stack(values).mean(dim=0)
        return combined[: self.n_users], combined[self.n_users :]


class LightGCNPlugin(ModelPlugin):
    name = "lightgcn"

    def train(self, context: TrainingContext) -> RuntimeSpec:
        torch.manual_seed(context.training.seed)
        device = torch.device(context.training.device)
        parameters = context.parameters
        model = LightGCN(
            *context.data.shape,
            embedding_dim=int(parameters.get("embedding_dim", 32)),
            layers=int(parameters.get("layers", 2)),
            adjacency=normalized_adjacency(context.data.train, device),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=context.training.learning_rate)
        sampler = PairwiseSampler(context.data.train, context.training.seed)
        batch_size = max(1, context.training.batch_size)
        steps = int(parameters.get("steps", max(1, context.data.train.nnz // batch_size)))
        l2 = float(parameters.get("l2", 0.0001))
        model.train()
        for _ in range(max(1, context.training.epochs) * steps):
            users, positives, negatives = sampler.sample(batch_size)
            user_ids = torch.from_numpy(users).to(device)
            positive_ids = torch.from_numpy(positives).to(device)
            negative_ids = torch.from_numpy(negatives).to(device)
            all_users, all_items = model.embeddings()
            positive_score = (all_users[user_ids] * all_items[positive_ids]).sum(dim=1)
            negative_score = (all_users[user_ids] * all_items[negative_ids]).sum(dim=1)
            ranking = functional.softplus(negative_score - positive_score).mean()
            regularization = (
                model.users(user_ids).square().sum()
                + model.items(positive_ids).square().sum()
                + model.items(negative_ids).square().sum()
            ) / batch_size
            loss = ranking + l2 * regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            user_embeddings, item_embeddings = model.embeddings()
        return RuntimeSpec(
            plugin=self.name,
            runtime="embedding",
            # EmbeddingRuntime can build a history profile by averaging the
            # item embeddings, in addition to serving the learned user vector.
            capabilities=(QueryMode.KNOWN_USER, QueryMode.HISTORY),
            arrays={
                "user_embeddings": user_embeddings.cpu().numpy().astype(np.float32),
                "item_embeddings": item_embeddings.cpu().numpy().astype(np.float32),
            },
            metadata={"score": "inner_product", "history_profile": "mean"},
        )
