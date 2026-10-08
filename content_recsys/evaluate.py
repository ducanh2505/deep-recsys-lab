"""Full-catalog scoring, validation-only fusion tuning and locked test evaluation."""

from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import torch

from content_recsys.common import code_hash, emit, sha256, write_json
from content_recsys.data import load_prepared, profiles
from multvae.evaluate import CUTOFFS, ranking_values
from multvae.train import model_from_config, verify_checkpoint


@lru_cache(maxsize=1)
def baseline_model(prepared: Path, device_name: str) -> torch.nn.Module:
    """Load and verify the immutable baseline for the prepared dataset."""
    data, _, _, manifest = load_prepared(prepared)
    path = Path(manifest["baseline"])
    if sha256(path) != manifest["sources"]["baseline"]:
        raise ValueError("Mult-VAE checkpoint changed after preparation")
    saved = torch.load(path, weights_only=True, map_location="cpu")
    verify_checkpoint(saved, data)
    model = model_from_config(saved["config"], torch.device(device_name))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model


def unseen_zscore(scores: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
    """Standardize only unseen scores; preserve constant-score rows safely."""
    count = (~seen).sum(1, keepdim=True)
    mean = scores.masked_fill(seen, 0).sum(1, keepdim=True) / count
    variance = (scores - mean).square().masked_fill(seen, 0).sum(1, keepdim=True) / count
    return (scores - mean) / variance.sqrt().clamp_min(1e-8)


def score_catalog(prepared: Path, embedding: torch.Tensor, split_name: str, profile: str,
                  alphas: list[float], device: torch.device, batch_users: int = 256,
                  depth: int = 100, user_limit: int | None = None) -> tuple[dict, dict]:
    """Return metrics and aligned per-user values without a global user-item score matrix."""
    data, weights, _, _ = load_prepared(prepared)
    if split_name not in ("valid", "test") or profile not in ("binary", "rating"):
        raise ValueError("Invalid split/profile")
    if not alphas or any(not 0 <= alpha <= 1 for alpha in alphas):
        raise ValueError("Fusion weights must be in [0, 1]")
    target = getattr(data, split_name)
    eligible = np.flatnonzero(np.diff(target.offsets) > 0)
    if user_limit is not None:
        eligible = eligible[:user_limit]
    if not len(eligible):
        raise ValueError("No eligible evaluation users")
    history_profiles = profiles(data, embedding, weights, profile, include_valid=split_name == "test")
    item_vectors = embedding.to(device)
    vae = baseline_model(prepared, str(device)) if any(alpha > 0 for alpha in alphas) else None
    names = [f"{metric}@{k}" for k in CUTOFFS if k <= depth for metric in ("recall", "ndcg")]
    per_user = {name: np.empty((len(alphas), len(eligible)), dtype=np.float64) for name in names}
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(eligible), batch_users):
            users = eligible[start:start + batch_users]
            history = torch.from_numpy(data.batch(users, include_valid=split_name == "test")).to(device)
            seen = history.bool()
            content = history_profiles[users].to(device) @ item_vectors.T
            cf = vae(history)[0] if vae is not None else None
            if not bool(torch.isfinite(content).all()) or (cf is not None and not bool(torch.isfinite(cf).all())):
                raise FloatingPointError("Non-finite ranking scores")
            content_z = unseen_zscore(content, seen)
            cf_z = unseen_zscore(cf, seen) if cf is not None else None
            for alpha_index, alpha in enumerate(alphas):
                # Raw endpoint scores avoid introducing new floating-point ties.
                scores = content if alpha == 0 else cf if alpha == 1 else alpha * cf_z + (1 - alpha) * content_z
                top = scores.masked_fill(seen, -torch.inf).topk(min(depth, data.n_items), sorted=True).indices.cpu().numpy()
                keys = users[:, None].astype(np.int64) * data.n_items + top
                positions = np.searchsorted(target.keys, keys)
                matches = ((positions < len(target.keys)) &
                           (target.keys[np.minimum(positions, len(target.keys) - 1)] == keys))
                counts = target.offsets[users + 1] - target.offsets[users]
                values = ranking_values(matches, counts)
                for name in names:
                    per_user[name][alpha_index, start:start + len(users)] = values[name]
    metrics = {str(alpha): {"users": len(eligible), **{name: float(values[k].mean()) for name, values in per_user.items()}}
               for k, alpha in enumerate(alphas)}
    metrics["elapsed_seconds"] = time.monotonic() - started
    per_user.update({"users": eligible, "user_ids": data.user_ids[eligible], "alphas": np.array(alphas),
                     "train_counts": np.diff(data.train.offsets)[eligible]})
    return metrics, per_user


def evaluate_run(prepared: Path, run_dir: Path, device: torch.device, split: str = "valid",
                 batch_users: int = 256, selection_lock: Path | None = None,
                 user_limit: int | None = None) -> dict:
    """Tune fusion on validation, or evaluate only prelocked choices on test."""
    config = json.loads((run_dir / "config.json").read_text())
    data, _, _, manifest = load_prepared(prepared)
    if config["prepared_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Run input fingerprint differs from prepared data")
    embedding_path = run_dir / "embeddings.npy"
    embedding_hash = sha256(embedding_path)
    if config["mode"] != "frozen":
        training_path = run_dir / "training.json"
        if not training_path.exists():
            raise ValueError("Training must finish before fusion tuning")
        if (json.loads(training_path.read_text())["embedding_sha256"] != embedding_hash
                or config["code_hash"] != code_hash()):
            raise ValueError("Training embeddings or implementation changed")
    else:
        expected = json.loads((prepared / "frozen" / "manifest.json").read_text())["artifacts"][config["layout"] + ".npy"]
        if embedding_hash != expected:
            raise ValueError("Frozen control differs from prepared features")
    result_path = run_dir / f"{split}_evaluation.json"
    locked = None
    lock_hash = None
    if split == "test":
        if selection_lock is None:
            raise ValueError("Test requires a selection lock created before any test evaluation")
        lock = json.loads(selection_lock.read_text())
        if lock["prepared_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Selection lock inputs differ")
        locked = lock["runs"][str(run_dir.resolve())]
        if locked["embedding_sha256"] != embedding_hash:
            raise ValueError("Embeddings changed after selection was locked")
        lock_hash = sha256(selection_lock)
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if (existing["embedding_sha256"] != embedding_hash or existing["user_limit"] != user_limit
                or existing["selection_lock_sha256"] != lock_hash):
            raise ValueError("Evaluation cache does not match embeddings, population or selection lock")
        return existing
    embedding = torch.from_numpy(np.load(embedding_path))
    if embedding.shape[0] != data.n_items or not bool(torch.isfinite(embedding).all()):
        raise ValueError("Embedding matrix is not a finite full-catalog matrix")
    result = {"split": split, "embedding_sha256": embedding_hash,
              "prepared_fingerprint": manifest["fingerprint"], "user_limit": user_limit,
              "selection_lock_sha256": lock_hash, "profiles": {}}
    for profile in ("binary", "rating"):
        alphas = [round(k / 20, 2) for k in range(21)] if split == "valid" else sorted({0.0, 1.0, locked["alphas"][profile]})
        metrics, values = score_catalog(prepared, embedding, split, profile, alphas, device, batch_users,
                                        depth=20 if split == "valid" else 100, user_limit=user_limit)
        best = (max(alphas, key=lambda a: (metrics[str(a)]["recall@20"], metrics[str(a)]["ndcg@20"], a))
                if split == "valid" else locked["alphas"][profile])
        np.savez_compressed(run_dir / f"{split}_{profile}_users.npz", **values)
        result["profiles"][profile] = {"alpha": best, "content": metrics[str(0.0)],
                                         "hybrid": metrics[str(best)], "multvae": metrics[str(1.0)], "grid": metrics}
        emit(run_dir, "fusion_evaluation", split=split, profile=profile, alpha=best,
             recall20=metrics[str(best)]["recall@20"], seconds=metrics["elapsed_seconds"])
    write_json(result_path, result)
    return result
