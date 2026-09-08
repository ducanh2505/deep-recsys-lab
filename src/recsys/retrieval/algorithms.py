"""Classical and representation-based retrieval algorithms."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np
import scipy.sparse as sp


def deterministic_topk(
    scores: np.ndarray, seen: np.ndarray | list[int] | set[int], k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact top-k with item-index tie breaking and seen-item masking."""

    values = np.asarray(scores, dtype=np.float64).copy()
    values[~np.isfinite(values)] = -np.inf
    if len(seen):
        values[np.asarray(list(seen), dtype=np.int64)] = -np.inf
    available = int(np.isfinite(values).sum())
    take = min(k, available)
    if take <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    provisional = np.flatnonzero(np.isfinite(values))
    order = np.lexsort((provisional, -values[provisional]))[:take]
    indices = provisional[order].astype(np.int64, copy=False)
    return indices, values[indices]


def item_knn(train: sp.csr_matrix, neighbours: int = 100, block_size: int = 512) -> sp.csr_matrix:
    norms = np.sqrt(np.asarray(train.power(2).sum(axis=0)).ravel())
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    normalized = train @ sp.diags(inverse)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for start in range(0, train.shape[1], block_size):
        stop = min(start + block_size, train.shape[1])
        scores = (normalized.T[start:stop] @ normalized).toarray().astype(np.float32)
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        for local in range(stop - start):
            picked, selected = deterministic_topk(scores[local], set(), neighbours)
            finite = np.isfinite(selected) & (selected > 0)
            if finite.any():
                rows.append(np.full(int(finite.sum()), start + local, dtype=np.int64))
                columns.append(picked[finite])
                values.append(selected[finite].astype(np.float32))
    if not rows:
        return sp.csr_matrix((train.shape[1], train.shape[1]), dtype=np.float32)
    return sp.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(train.shape[1], train.shape[1]),
    )


def transition_matrix(
    user_indices: np.ndarray, item_indices: np.ndarray, timestamps: np.ndarray, n_items: int
) -> sp.csr_matrix:
    order = np.lexsort((np.arange(len(user_indices)), timestamps, user_indices))
    users = user_indices[order]
    items = item_indices[order]
    adjacent = users[:-1] == users[1:]
    rows = items[:-1][adjacent]
    columns = items[1:][adjacent]
    matrix = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)), shape=(n_items, n_items)
    )
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
    return (sp.diags(inverse) @ matrix).tocsr()


def cooccurrence(train: sp.csr_matrix, min_count: int = 1) -> sp.csr_matrix:
    binary = train.copy().astype(np.float32)
    binary.data.fill(1.0)
    result = (binary.T @ binary).tocsr()
    result.setdiag(0)
    result.data[result.data < min_count] = 0
    result.eliminate_zeros()
    return result


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[\w-]+", value.lower(), flags=re.UNICODE)


def tfidf_matrix(documents: list[str]) -> sp.csr_matrix:
    tokens = [_tokenize(document) for document in documents]
    vocabulary = sorted({token for document in tokens for token in document})
    if not vocabulary:
        return sp.identity(len(documents), dtype=np.float32, format="csr")
    lookup = {token: index for index, token in enumerate(vocabulary)}
    document_frequency = Counter(token for document in tokens for token in set(document))
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, document in enumerate(tokens):
        counts = Counter(document)
        for token, count in counts.items():
            rows.append(row)
            columns.append(lookup[token])
            values.append(
                float(count) * (math.log((1 + len(tokens)) / (1 + document_frequency[token])) + 1)
            )
    matrix = sp.csr_matrix((values, (rows, columns)), shape=(len(documents), len(vocabulary)))
    norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=1)).ravel())
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    return (sp.diags(inverse) @ matrix).tocsr().astype(np.float32)


def hashed_text_embeddings(documents: list[str], dimension: int = 128) -> np.ndarray:
    if dimension < 2:
        raise ValueError("semantic embedding dimension must be at least two")
    embeddings = np.zeros((len(documents), dimension), dtype=np.float32)
    for row, document in enumerate(documents):
        for token in _tokenize(document):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            column = int.from_bytes(digest[:4], "big") % dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            embeddings[row, column] += sign
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0)


def similarity_from_embeddings(embeddings: np.ndarray, neighbours: int = 100) -> sp.csr_matrix:
    similarity = np.asarray(embeddings @ embeddings.T, dtype=np.float32)
    np.fill_diagonal(similarity, -np.inf)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for row in range(len(similarity)):
        picked, scores = deterministic_topk(similarity[row], set(), neighbours)
        finite = np.isfinite(scores) & (scores > 0)
        if finite.any():
            rows.append(np.full(int(finite.sum()), row, dtype=np.int64))
            columns.append(picked[finite])
            values.append(scores[finite].astype(np.float32))
    if not rows:
        return sp.csr_matrix(similarity.shape, dtype=np.float32)
    return sp.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=similarity.shape,
    )
