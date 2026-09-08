"""Deterministic sampling shared by pairwise model trainers."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class PairwiseSampler:
    def __init__(self, train: sp.csr_matrix, seed: int) -> None:
        self.train = train.tocsr(copy=False)
        self.rng = np.random.default_rng(seed)
        self.users = np.flatnonzero(self.train.getnnz(axis=1) < self.train.shape[1])
        self.users = self.users[self.train.getnnz(axis=1)[self.users] > 0]
        if not len(self.users):
            raise ValueError("pairwise training requires users with positive and unseen items")

    def sample(self, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        users = self.rng.choice(self.users, size=size, replace=True).astype(np.int64)
        positives = np.empty(size, dtype=np.int64)
        negatives = np.empty(size, dtype=np.int64)
        for index, user in enumerate(users):
            seen = self.train.indices[self.train.indptr[user] : self.train.indptr[user + 1]]
            positives[index] = int(self.rng.choice(seen))
            candidate = int(self.rng.integers(self.train.shape[1]))
            while candidate in seen:
                candidate = int(self.rng.integers(self.train.shape[1]))
            negatives[index] = candidate
        return users, positives, negatives
