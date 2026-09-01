"""The deliberately simplified LightGCN architecture from the paper."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F


def build_normalized_adjacency(interactions: sp.csr_matrix) -> torch.Tensor:
    """Build the loop-free symmetric bipartite ``D^-1/2 A D^-1/2`` graph."""

    matrix = interactions.tocoo(copy=False)
    n_users, n_items = matrix.shape
    user_degree = np.asarray(interactions.getnnz(axis=1), dtype=np.float64)
    item_degree = np.asarray(interactions.getnnz(axis=0), dtype=np.float64)
    user_scale = np.zeros_like(user_degree)
    item_scale = np.zeros_like(item_degree)
    np.power(user_degree, -0.5, out=user_scale, where=user_degree > 0)
    np.power(item_degree, -0.5, out=item_scale, where=item_degree > 0)
    weights = (user_scale[matrix.row] * item_scale[matrix.col]).astype(np.float32)
    rows = np.concatenate((matrix.row, n_users + matrix.col)).astype(np.int64)
    cols = np.concatenate((n_users + matrix.col, matrix.row)).astype(np.int64)
    values = np.concatenate((weights, weights))
    indices = torch.from_numpy(np.stack((rows, cols), axis=0))
    return torch.sparse_coo_tensor(
        indices,
        torch.from_numpy(values),
        size=(n_users + n_items, n_users + n_items),
        dtype=torch.float32,
        check_invariants=False,
    ).coalesce()


class LightGCN(nn.Module):
    """LightGCN with only ego embeddings and linear neighborhood propagation."""

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
            raise ValueError("layers must be at least one")
        if adjacency.shape != (n_users + n_items, n_users + n_items):
            raise ValueError("adjacency shape does not match user/item counts")
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.register_buffer("adjacency", adjacency.coalesce(), persistent=False)

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniformly average layer-0 through layer-K embeddings."""

        current = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        layers = [current]
        for _ in range(self.layers):
            current = torch.sparse.mm(self.adjacency, current)
            layers.append(current)
        combined = torch.stack(layers, dim=0).mean(dim=0)
        return combined[: self.n_users], combined[self.n_users :]

    def bpr_loss(
        self,
        users: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
        l2: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total, ranking, and ego-embedding L2 losses."""

        all_users, all_items = self.propagated_embeddings()
        user_vectors = all_users[users]
        positive_vectors = all_items[positive_items]
        negative_vectors = all_items[negative_items]
        positive_scores = (user_vectors * positive_vectors).sum(dim=1)
        negative_scores = (user_vectors * negative_vectors).sum(dim=1)
        ranking = F.softplus(negative_scores - positive_scores).mean()
        ego_l2 = (
            0.5
            * (
                self.user_embedding(users).square().sum()
                + self.item_embedding(positive_items).square().sum()
                + self.item_embedding(negative_items).square().sum()
            )
            / users.shape[0]
        )
        return ranking + l2 * ego_l2, ranking, ego_l2

    def score(self, users: torch.Tensor) -> torch.Tensor:
        """Score every catalog item by inner product for the supplied users."""

        all_users, all_items = self.propagated_embeddings()
        return all_users[users] @ all_items.T
