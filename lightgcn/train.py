"""Train and evaluate LightGCN on the prepared MovieLens 20M splits."""

import argparse
from dataclasses import dataclass
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lightgcn.data import Dataset, Split, load_dataset, normalized_adjacency, sample_negatives
from lightgcn.model import LightGCN


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainLossStopping:
    """Track the lowest loss separately from significant patience improvements."""

    patience: int
    min_delta: float
    best_loss: float = math.inf
    improvement_loss: float = math.inf
    stale_epochs: int = 0

    def update(self, loss: float) -> tuple[bool, bool]:
        """Return whether to save this epoch and whether to stop training."""
        if not math.isfinite(loss):
            raise ValueError("Non-finite training loss")
        save = loss < self.best_loss
        self.best_loss = min(self.best_loss, loss)
        if loss < self.improvement_loss and self.improvement_loss - loss >= self.min_delta:
            self.improvement_loss = loss
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return save, self.stale_epochs >= self.patience


def warmup_learning_rate(base_lr: float, step: int, warmup_steps: int) -> float:
    """Return the learning rate for a one-based optimizer step."""
    return base_lr * min(step / warmup_steps, 1.0) if warmup_steps else base_lr


def validate_checkpoint(checkpoint: dict, data: Dataset, model: LightGCN) -> None:
    """Reject checkpoint ID or architecture mismatches before loading weights."""
    for name in ("user_ids", "item_ids"):
        if not np.array_equal(checkpoint[name].cpu().numpy(), getattr(data, name)):
            raise ValueError(f"Checkpoint {name} mapping does not match the data")
    config = checkpoint["config"]
    weights = tuple(config.get("layer_weights", [1 / (config["layers"] + 1)] * (config["layers"] + 1)))
    if (config["layers"] != model.layers or
            config["embedding_dim"] != model.user_embedding.embedding_dim or
            not np.allclose(weights, model.layer_weights, rtol=0, atol=1e-8)):
        raise ValueError("Checkpoint architecture does not match the requested model")


def train_epoch(
    model: LightGCN, adjacency: torch.Tensor, data: Dataset,
    optimizer: torch.optim.Optimizer, rng: np.random.Generator,
    batch_size: int, l2: float, base_lr: float,
    step_offset: int = 0, warmup_steps: int = 0,
) -> dict[str, float | int]:
    """Train one shuffled epoch, retaining optimizer moments across warmup."""
    model.train()
    device = model.user_embedding.weight.device
    order = rng.permutation(len(data.train.users))
    epoch_loss = 0.0
    first_lr = last_lr = base_lr
    for batch, begin in enumerate(range(0, len(order), batch_size), start=1):
        lr = warmup_learning_rate(base_lr, step_offset + batch, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if batch == 1:
            first_lr = lr
        last_lr = lr
        indices = order[begin : begin + batch_size]
        users_np = data.train.users[indices]
        positives_np = data.train.items[indices]
        negatives_np = sample_negatives(users_np, data.train_bits, data.n_items, rng)
        users = torch.from_numpy(users_np).to(device=device, dtype=torch.long)
        positives = torch.from_numpy(positives_np).to(device=device, dtype=torch.long)
        negatives = torch.from_numpy(negatives_np).to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        user_embeddings, item_embeddings = model.propagate(adjacency)
        pos_scores = model.scores(user_embeddings[users], item_embeddings[positives])
        neg_scores = model.scores(user_embeddings[users], item_embeddings[negatives])
        bpr = F.softplus(neg_scores - pos_scores).mean()
        initial_users = model.user_embedding(users)
        initial_pos = model.item_embedding(positives)
        initial_neg = model.item_embedding(negatives)
        reg = (initial_users.square().sum() + initial_pos.square().sum() +
               initial_neg.square().sum()) / (2 * len(indices))
        loss = bpr + l2 * reg
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise ValueError(f"Non-finite training loss at optimizer step {step_offset + batch}")
        loss.backward()
        optimizer.step()
        epoch_loss += loss_value * len(indices)
    return {"train_loss": epoch_loss / len(order), "learning_rate_start": first_lr,
            "learning_rate_end": last_lr,
            "optimizer_steps": step_offset + math.ceil(len(order) / batch_size)}


def evaluate(
    model: LightGCN,
    adjacency: torch.Tensor,
    data: Dataset,
    target: Split,
    target_offsets: np.ndarray,
    mask_valid: bool,
    max_users: int | None = None,
    batch_users: int = 256,
) -> dict[str, float | int]:
    """Full-catalog Recall@20 and NDCG@20, excluding known interactions."""
    if data.train_split == "train-valid" and target is data.valid:
        raise ValueError("Validation interactions have already been used for training")
    model.eval()
    device = model.user_embedding.weight.device
    with torch.no_grad():
        user_embeddings, item_embeddings = model.propagate(adjacency)
        item_embeddings_t = item_embeddings.T
        eligible = np.flatnonzero(np.diff(target_offsets) > 0)
        if not len(eligible):
            return {"users": 0, "recall@20": 0.0, "ndcg@20": 0.0}
        if max_users is not None and len(eligible) > max_users:
            # A fixed uniform subset for inexpensive intermediate validation.
            eligible = eligible[np.linspace(0, len(eligible) - 1, max_users, dtype=np.int64)]
        recall_total = 0.0
        ndcg_total = 0.0
        k = min(20, data.n_items)
        discounts = 1 / np.log2(np.arange(2, k + 2))
        ideal = np.r_[0, np.cumsum(discounts)]
        target_keys = target.users.astype(np.int64) * data.n_items + target.items
        for start in range(0, len(eligible), batch_users):
            users = eligible[start : start + batch_users]
            scores = user_embeddings[torch.as_tensor(users, device=device).long()] @ item_embeddings_t
            for seen, offsets in ((data.train, data.train_offsets), (data.valid, data.valid_offsets)):
                if seen is data.valid and (not mask_valid or data.train_split == "train-valid"):
                    continue
                lengths = offsets[users + 1] - offsets[users]
                rows = np.repeat(np.arange(len(users), dtype=np.int64), lengths)
                positions = np.concatenate([np.arange(offsets[u], offsets[u + 1]) for u in users])
                columns = seen.items[positions].astype(np.int64)
                scores[
                    torch.as_tensor(rows, device=device),
                    torch.as_tensor(columns, device=device),
                ] = -torch.inf
            top_items = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
            query_keys = users[:, None].astype(np.int64) * data.n_items + top_items
            positions = np.searchsorted(target_keys, query_keys)
            matches = (positions < len(target_keys)) & (
                target_keys[np.minimum(positions, len(target_keys) - 1)] == query_keys
            )
            truth_counts = target_offsets[users + 1] - target_offsets[users]
            recall_total += float((matches.sum(axis=1) / truth_counts).sum())
            dcg = (matches * discounts).sum(axis=1)
            ndcg_total += float((dcg / ideal[np.minimum(truth_counts, k)]).sum())
    return {
        "users": int(len(eligible)),
        "recall@20": recall_total / len(eligible),
        "ndcg@20": ndcg_total / len(eligible),
    }


def save_checkpoint(
    path: Path, model: LightGCN, data: Dataset, config: dict, epoch: int,
    valid: dict | None = None, optimizer: torch.optim.Optimizer | None = None,
    training: dict | None = None, rng: np.random.Generator | None = None,
) -> None:
    """Atomically save selected weights, provenance, and available Adam state."""
    payload = {
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "user_ids": torch.from_numpy(data.user_ids.copy()),
        "item_ids": torch.from_numpy(data.item_ids.copy()),
        "config": config,
        "epoch": epoch,
        "validation": valid,
    }
    if optimizer is not None:
        state = optimizer.state_dict()
        state["state"] = {
            key: {name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                  for name, value in values.items()}
            for key, values in state["state"].items()
        }
        payload["optimizer_state_dict"] = state
    if training is not None:
        payload["training"] = training
    if rng is not None:
        payload["numpy_rng_state"] = rng.bit_generator.state
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/ml20m_lightgcn")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/lightgcn_ml20m")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--layer-weights", type=str, default=None,
                        help="Comma-separated nonnegative alpha_0..alpha_K summing to 1; default uniform")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--max-epochs", type=int, default=30, help="Main epochs, excluding warmup")
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5, help="Checks/epochs without sufficient improvement")
    parser.add_argument("--train-split", choices=["train", "train-valid"], default="train")
    parser.add_argument("--init-checkpoint", type=Path, help="Initialize weights only; use a fresh Adam optimizer")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Linear per-step learning-rate warmup")
    parser.add_argument("--early-stopping", choices=["validation", "train-loss", "none"], default="validation")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Absolute train-loss improvement for patience")
    parser.add_argument("--validation-users", type=int, default=20000, help="0 means all users")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--skip-test", action="store_true", help="Do not evaluate test; use for validation-only search trials")
    args = parser.parse_args()
    try:
        layer_weights = (tuple(float(x) for x in args.layer_weights.split(","))
                         if args.layer_weights is not None else None)
    except ValueError:
        parser.error("layer-weights must be comma-separated numbers")
    if args.batch_size < 1 or args.max_epochs < 1 or args.eval_every < 1 or args.patience < 1:
        parser.error("batch-size, max-epochs, eval-every and patience must be positive")
    if (not math.isfinite(args.l2) or not math.isfinite(args.learning_rate) or
            args.l2 < 0 or args.learning_rate <= 0 or args.validation_users < 0):
        parser.error("l2 and validation-users must be nonnegative; learning-rate must be positive")
    if args.warmup_epochs < 0 or not math.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error("warmup-epochs and min-delta must be nonnegative and finite")
    if args.train_split == "train-valid" and args.early_stopping == "validation":
        parser.error("train-valid cannot use validation early stopping; select train-loss or none")
    for name in ("data_dir", "output_dir", "init_checkpoint"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, (ROOT / value).resolve())
    if args.init_checkpoint == args.output_dir / "best.pt":
        parser.error("output-dir would overwrite the initialization checkpoint")
    if args.train_split == "train-valid" and args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("train-valid requires an empty output directory")
    run_started = time.monotonic()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = load_dataset(args.data_dir, args.train_split)
    adjacency = normalized_adjacency(data, device)
    model = LightGCN(data.n_users, data.n_items, args.embedding_dim, args.layers,
                     layer_weights=layer_weights).to(device)
    source_epoch = None
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        validate_checkpoint(initial, data, model)
        model.load_state_dict(initial["state_dict"])
        source_epoch = initial["epoch"]
        manifest_path = args.data_dir / "manifest.json"
        if manifest_path.exists() and initial["config"].get("data_canonical_sha256"):
            current_hashes = json.loads(manifest_path.read_text()).get("canonical_sha256")
            if initial["config"]["data_canonical_sha256"] != current_hashes:
                raise ValueError("Initialization checkpoint data manifest hashes do not match")
        del initial
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["layer_weights"] = list(model.layer_weights)
    config.update({"device_used": device_name, "n_users": data.n_users, "n_items": data.n_items,
                   "n_train": len(data.train.users), "n_valid": len(data.valid.users), "n_test": len(data.test.users)})
    config["source_split_counts"] = {
        "train": len(data.train.users) - (len(data.valid.users) if args.train_split == "train-valid" else 0),
        "valid": len(data.valid.users), "test": len(data.test.users),
    }
    batches_per_epoch = math.ceil(len(data.train.users) / args.batch_size)
    warmup_steps = args.warmup_epochs * batches_per_epoch
    config.update({"initialization": "checkpoint" if args.init_checkpoint else "xavier",
                   "source_epoch": source_epoch, "optimizer_initialization": "fresh_adam",
                   "warmup_steps": warmup_steps, "selection_criterion": args.early_stopping})
    manifest_path = args.data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        config["data_canonical_sha256"] = manifest.get("canonical_sha256")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"event": "start", "config": config}), flush=True)
    history = []
    best_recall = -math.inf
    best_epoch = 0
    stale_checks = 0
    loss_stopping = TrainLossStopping(args.patience, args.min_delta)
    checkpoint = args.output_dir / "best.pt"
    stop_reason = "max_epochs"
    for epoch in range(1, args.warmup_epochs + args.max_epochs + 1):
        started = time.monotonic()
        phase = "warmup" if epoch <= args.warmup_epochs else "main"
        main_epoch = max(0, epoch - args.warmup_epochs)
        record = {"epoch": epoch, "phase": phase, "main_epoch": main_epoch,
                  **train_epoch(model, adjacency, data, optimizer, rng, args.batch_size,
                                args.l2, args.learning_rate,
                                step_offset=(epoch - 1) * batches_per_epoch, warmup_steps=warmup_steps),
                  "seconds": time.monotonic() - started}
        save = stop = False
        valid = None
        if phase == "main" and args.early_stopping == "validation" and (
                main_epoch % args.eval_every == 0 or main_epoch == args.max_epochs):
            valid = evaluate(model, adjacency, data, data.valid, data.valid_offsets, False,
                             max_users=args.validation_users or None)
            record["validation"] = valid
            if valid["recall@20"] > best_recall:
                best_recall = valid["recall@20"]
                stale_checks = 0
                save = True
            else:
                stale_checks += 1
            stop = stale_checks >= args.patience
        elif phase == "main" and args.early_stopping == "train-loss":
            save, stop = loss_stopping.update(record["train_loss"])
            record["stale_epochs"] = loss_stopping.stale_epochs
        elif phase == "main" and args.early_stopping == "none":
            save = main_epoch == args.max_epochs
        if save:
            best_epoch = epoch
            save_checkpoint(checkpoint, model, data, config, epoch, valid,
                            optimizer=optimizer, training=record.copy())
        record["checkpoint_saved"] = save
        history.append(record)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if stop:
            stop_reason = "early_stopping"
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    metrics = {"selected_epoch": best_epoch, "selected_main_epoch": best_epoch - args.warmup_epochs,
               "selection_criterion": args.early_stopping, "selected_train_loss": saved["training"]["train_loss"],
               "epochs_completed": len(history), "warmup_epochs_completed": args.warmup_epochs,
               "main_epochs_completed": len(history) - args.warmup_epochs,
               "stop_reason": stop_reason, "train_split": args.train_split,
               "training_seconds": sum(record["seconds"] for record in history)}
    if args.train_split == "train":
        metrics["selection_validation"] = saved["validation"]
        metrics["full_validation"] = evaluate(model, adjacency, data, data.valid, data.valid_offsets, False)
    if not args.skip_test:
        evaluation_started = time.monotonic()
        metrics["test"] = evaluate(model, adjacency, data, data.test, data.test_offsets, True)
        metrics["test_seconds"] = time.monotonic() - evaluation_started
    metrics["total_seconds"] = time.monotonic() - run_started
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"event": "complete", "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
