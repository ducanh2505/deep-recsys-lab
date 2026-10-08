"""Resumable projection and LoRA training with detached full-history memory banks."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from content_recsys.common import (ROOT, _cpu_tree, code_hash, device_for, emit, environment, exclusive_lock, memory_snapshot,
                                   restore_rng, rng_state, save_checkpoint, sha256, write_json)
from content_recsys.data import load_prepared, profiles
from content_recsys.encoder import ContentModel, ensure_frozen
from content_recsys.evaluate import score_catalog
from content_recsys.objective import candidates, epoch_samples, implicit_loss


def run(config: dict) -> dict:
    """Lock a single training directory before starting or resuming it."""
    with exclusive_lock(Path(config["output"]) / ".training.lock"):
        return _run(config)


def _run(config: dict) -> dict:
    """Train one configuration and automatically resume its last atomic checkpoint."""
    config = dict(config)
    prepared, output = Path(config["prepared"]), Path(config["output"])
    device = device_for(config["device"])
    data, weights, _, manifest = load_prepared(prepared)
    config.update({"prepared_fingerprint": manifest["fingerprint"], "code_hash": code_hash(),
                   "dependency_lock_sha256": sha256(ROOT / "uv.lock"),
                   "selection_metric": "content_binary_validation_recall@20",
                   "memory_bank": "epoch-refreshed; detached full histories; candidate-only gradients"})
    frozen = ensure_frozen(prepared, device)
    config["frozen_sha256"] = frozen["artifacts"][config["layout"] + ".npy"]
    if config["loss"] not in ("bpr", "infonce") or config["temperature"] <= 0 or config["learning_rate"] <= 0:
        raise ValueError("Invalid loss/temperature/learning rate")
    if min(config[k] for k in ("max_epochs", "patience", "batch_size", "negative_count", "eval_batch_size", "checkpoint_batches")) < 1:
        raise ValueError("Epoch, patience and batch settings must be positive")
    if output.exists() and (output / "config.json").exists():
        old = json.loads((output / "config.json").read_text())
        if old != config:
            raise ValueError("Resume config or implementation fingerprint differs")
        if (output / "training.json").exists():
            completed = json.loads((output / "training.json").read_text())
            if sha256(output / "embeddings.npy") != completed["embedding_sha256"]:
                raise ValueError("Completed run embedding fingerprint mismatch")
            return completed
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "config.json").exists() and any(p.name != ".training.lock" for p in output.iterdir()):
        raise ValueError("Training output directory contains unrelated files")
    # The frozen cache key is a persistent part of resume compatibility.
    if (output / "config.json").exists():
        old = json.loads((output / "config.json").read_text())
        if old != config:
            raise ValueError("Frozen input cache changed")
    write_json(output / "config.json", config)
    if not (output / "environment.json").exists():
        write_json(output / "environment.json", environment())
    torch.manual_seed(config["seed"])
    model = ContentModel(prepared, config["layout"], config["mode"], device)
    projection = list(model.projection.parameters())
    groups = [{"params": projection, "lr": config["learning_rate"] if config["mode"] == "projection" else 1e-3}]
    if model.text is not None:
        groups.append({"params": [p for p in model.text.parameters() if p.requires_grad], "lr": config["learning_rate"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01, foreach=False)
    history, best, stale, start_epoch, start_batch, elapsed_before = [], None, 0, 1, 0, 0.0
    partial_loss, partial_users = 0.0, 0
    peaks = {key: 0.0 for key in memory_snapshot(device)}
    last_path = output / "last.pt"
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["config"] != config:
            raise ValueError("Checkpoint config mismatch")
        model.restore_learned(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"], device)
        history, best, stale = saved["history"], saved["best"], saved["stale"]
        start_epoch, start_batch = saved["next_epoch"], saved["next_batch"]
        elapsed_before, peaks = saved["elapsed_seconds"], saved["memory"]
        partial_loss, partial_users = saved["partial_loss"], saved["partial_users"]
        if start_batch and sha256(output / "epoch_bank.npy") != saved["epoch_bank_sha256"]:
            raise ValueError("Epoch-start memory bank differs from the resume checkpoint")
        if best is not None:
            save_checkpoint(output / "best.pt", best)
    started = time.monotonic()
    emit(output, "training_start", config=config, environment=environment(), resumed=last_path.exists(),
         trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))

    def save(next_epoch: int, next_batch: int, loss_sum: float, user_count: int,
             epoch_bank: torch.Tensor | None = None) -> None:
        if epoch_bank is not None:
            np.save(output / "epoch_bank.npy", epoch_bank.numpy())
        for key, value in memory_snapshot(device).items():
            peaks[key] = max(peaks[key], value)
        save_checkpoint(last_path, {"config": config, "state_dict": model.learned_state(),
                                   "optimizer": _cpu_tree(optimizer.state_dict()), "rng": rng_state(device),
                                   "history": history, "best": best, "stale": stale,
                                   "next_epoch": next_epoch, "next_batch": next_batch,
                                   "partial_loss": loss_sum, "partial_users": user_count,
                                   "elapsed_seconds": elapsed_before + time.monotonic() - started, "memory": peaks,
                                   "epoch_bank_sha256": sha256(output / "epoch_bank.npy") if next_batch else None})

    stop = "max_epochs"
    refreshed = None
    for epoch in range(start_epoch, config["max_epochs"] + 1):
        if stale >= config["patience"]:
            stop = "early_stopping"
            break
        epoch_started = time.monotonic()
        if epoch == start_epoch and start_batch > 0:
            bank = torch.from_numpy(np.load(output / "epoch_bank.npy"))
        else:
            bank = refreshed if refreshed is not None else model.catalog(
                batch_size=256 if config["mode"] == "projection" else 8, initial=not last_path.exists() and epoch == 1)
            # Persist the epoch-start bank; a resumed epoch must use this exact bank.
            temporary = output / "epoch_bank.tmp.npy"
            np.save(temporary, bank.numpy())
            temporary.replace(output / "epoch_bank.npy")
        history_sums = profiles(data, bank, weights, "binary", normalize=False)
        users, positives = epoch_samples(data, config["seed"], epoch)
        loss_sum = partial_loss if epoch == start_epoch else 0.0
        user_count = partial_users if epoch == start_epoch else 0
        first = start_batch if epoch == start_epoch else 0
        total_batches = (data.n_users + config["batch_size"] - 1) // config["batch_size"]
        model.train()
        for batch_index in range(first, total_batches):
            offset = batch_index * config["batch_size"]
            batch_users = users[offset:offset + config["batch_size"]]
            positive = positives[offset:offset + config["batch_size"]]
            pool, target, negative = candidates(data, batch_users, positive, config["seed"], epoch,
                                                 batch_index, config["negative_count"])
            query = F.normalize(history_sums[batch_users] - bank[positive], dim=1).to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(pool)
            scores = query @ embeddings.T
            loss = implicit_loss(scores, torch.from_numpy(target).to(device), torch.from_numpy(negative).to(device),
                                 config["loss"], config["temperature"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                                   error_if_nonfinite=True, foreach=False)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_users)
            user_count += len(batch_users)
            if batch_index == first or (batch_index + 1) % 100 == 0:
                for key, value in memory_snapshot(device).items():
                    peaks[key] = max(peaks[key], value)
                emit(output, "training_batch", epoch=epoch, batch=batch_index + 1, total_batches=total_batches,
                     loss=float(loss.detach()), gradient_norm=float(norm), seconds=time.monotonic() - epoch_started,
                     memory=peaks)
            if (batch_index + 1) % config.get("checkpoint_batches", 200) == 0:
                save(epoch, batch_index + 1, loss_sum, user_count)
        del history_sums
        refreshed = model.catalog(batch_size=256 if config["mode"] == "projection" else 8)
        metrics, _ = score_catalog(prepared, refreshed, "valid", "binary", [0.0], device, config["eval_batch_size"])
        valid = metrics["0.0"]
        key = (valid["recall@20"], valid["ndcg@20"])
        improved = best is None or key > (best["validation"]["recall@20"], best["validation"]["ndcg@20"])
        if improved:
            best = {"config": config, "state_dict": model.learned_state(), "epoch": epoch,
                    "validation": valid, "item_ids": torch.from_numpy(data.item_ids),
                    "user_ids": torch.from_numpy(data.user_ids), "embedding": refreshed.clone()}
            save_checkpoint(output / "best.pt", best)
            np.save(output / "embeddings.npy", refreshed.numpy())
            stale = 0
        else:
            stale += 1
        record = {"epoch": epoch, "loss": loss_sum / user_count, "validation": valid, "improved": improved,
                  "seconds": time.monotonic() - epoch_started}
        history.append(record)
        write_json(output / "history.json", history)
        emit(output, "epoch_complete", **record)
        partial_loss, partial_users = 0.0, 0
        save(epoch + 1, 0, 0.0, 0)
    if best is None:
        raise RuntimeError("Training finished without a selected checkpoint")
    np.save(output / "embeddings.npy", best["embedding"].numpy())
    result = {"completed": True, "selected_epoch": best["epoch"], "validation": best["validation"],
              "history": history, "elapsed_seconds": elapsed_before + time.monotonic() - started,
              "memory": peaks, "stop_reason": stop, "embedding_sha256": sha256(output / "embeddings.npy")}
    write_json(output / "training.json", result)
    emit(output, "training_complete", **{k: v for k, v in result.items() if k != "history"})
    return result
