"""Full-catalog metrics using the existing repository's Recall/NDCG convention."""

import numpy as np
import torch

from multvae.data import Dataset
from multvae.model import MultVAE


CUTOFFS = (10, 20, 50, 100)


def ranking_values(matches: np.ndarray, truth_counts: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-user Recall/NDCG at each cutoff from a sorted hit matrix."""
    if np.any(truth_counts <= 0):
        raise ValueError("Ranking metrics require nonempty targets")
    discounts = 1.0 / np.log2(np.arange(2, matches.shape[1] + 2))
    ideal = np.r_[0.0, np.cumsum(discounts)]
    hits = np.cumsum(matches, axis=1)
    dcg = np.cumsum(matches * discounts, axis=1)
    values = {}
    for cutoff in CUTOFFS:
        depth = min(cutoff, matches.shape[1])
        values[f"recall@{cutoff}"] = hits[:, depth - 1] / truth_counts
        values[f"ndcg@{cutoff}"] = dcg[:, depth - 1] / ideal[np.minimum(truth_counts, depth)]
    return values


def evaluate(
    model: MultVAE, data: Dataset, split_name: str, batch_users: int = 500,
    test_input: str = "train-valid",
) -> dict:
    """Score every eligible user; use validation input only for final test."""
    if split_name not in ("valid", "test") or test_input not in ("train", "train-valid"):
        raise ValueError("Invalid split or test-input policy")
    if batch_users < 1:
        raise ValueError("Evaluation batch size must be positive")
    target = getattr(data, split_name)
    eligible = np.flatnonzero(np.diff(target.offsets) > 0)
    if len(eligible) == 0:
        raise ValueError(f"No eligible {split_name} users")
    device = next(model.parameters()).device
    sums = {f"{metric}@{k}": 0.0 for k in CUTOFFS for metric in ("recall", "ndcg")}
    squares = sums.copy()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(eligible), batch_users):
            users = eligible[start:start + batch_users]
            history = torch.from_numpy(data.batch(
                users, include_valid=split_name == "test" and test_input == "train-valid",
            )).to(device)
            scores, _, _ = model(history)
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("Non-finite ranking scores")
            scores.masked_fill_(history.bool(), -torch.inf)
            # Test always masks validation, including when using train-only input.
            if split_name == "test" and test_input == "train":
                rows, columns = data.valid.coordinates(users)
                scores[torch.from_numpy(rows).to(device), torch.from_numpy(columns).to(device)] = -torch.inf
            top_items = scores.topk(min(max(CUTOFFS), data.n_items), dim=1, sorted=True).indices.cpu().numpy()
            query_keys = users[:, None].astype(np.int64) * data.n_items + top_items
            positions = np.searchsorted(target.keys, query_keys)
            matches = (positions < len(target.keys)) & (
                target.keys[np.minimum(positions, len(target.keys) - 1)] == query_keys)
            counts = target.offsets[users + 1] - target.offsets[users]
            for name, values in ranking_values(matches, counts).items():
                sums[name] += float(values.sum())
                squares[name] += float(np.square(values).sum())
    means = {name: value / len(eligible) for name, value in sums.items()}
    standard_errors = {name: float(np.sqrt(max(0.0, squares[name] / len(eligible) - means[name] ** 2)
                                         / len(eligible))) for name in sums}
    return {"users": int(len(eligible)), **means, "standard_errors": standard_errors}
