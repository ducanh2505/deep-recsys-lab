"""The LightGCN architecture in He et al. (SIGIR 2020), equations 3-8."""

import math

import torch
from torch import nn


class LightGCN(nn.Module):
    def __init__(
        self, n_users: int, n_items: int, embedding_dim: int, layers: int,
        layer_weights: tuple[float, ...] | None = None,
    ):
        super().__init__()
        if layers < 1 or embedding_dim < 1:
            raise ValueError("layers and embedding_dim must be positive")
        self.n_users = n_users
        self.n_items = n_items
        self.layers = layers
        if layer_weights is None:
            layer_weights = tuple(1 / (layers + 1) for _ in range(layers + 1))
        if (len(layer_weights) != layers + 1 or
                any(not math.isfinite(weight) or weight < 0 for weight in layer_weights)
                or abs(sum(layer_weights) - 1) > 1e-6):
            raise ValueError("layer_weights must be K+1 nonnegative values summing to 1")
        self.layer_weights = layer_weights
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def propagate(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine layers 0..K with fixed weights after symmetric aggregation."""
        embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight))
        combined = self.layer_weights[0] * embeddings
        for layer in range(self.layers):
            embeddings = torch.sparse.mm(adjacency, embeddings)
            combined = combined + self.layer_weights[layer + 1] * embeddings
        return combined.split((self.n_users, self.n_items))

    @staticmethod
    def scores(users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (users * items).sum(dim=-1)
