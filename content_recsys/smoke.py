"""Operational smoke runs on real audited inputs; no unit-test runner."""

import gc
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from content_recsys.common import (_cpu_tree, code_hash, emit, memory_snapshot, restore_rng, rng_state,
                                   save_checkpoint, write_json)
from content_recsys.data import load_prepared, profiles
from content_recsys.encoder import ContentModel
from content_recsys.evaluate import baseline_model, score_catalog
from content_recsys.objective import candidates, epoch_samples, implicit_loss
from multvae.evaluate import ranking_values


def smoke(prepared: Path, output: Path, device: torch.device,
          modes: tuple[str, ...] = ("projection",)) -> dict:
    """Check actual forward/backward, optimizer recovery and fusion endpoints before training."""
    if not modes or not set(modes).issubset({"projection", "lora"}):
        raise ValueError("Smoke modes must contain projection and/or lora")
    output.mkdir(parents=True, exist_ok=True)
    data, weights, _, manifest = load_prepared(prepared)
    checks = []
    # A real one-negative objective must reduce to the same pairwise likelihood.
    scores = torch.tensor([[0.2, -0.1], [0.1, 0.3]], device=device, requires_grad=True)
    targets = torch.tensor([0, 0], device=device)
    negatives = torch.tensor([[False, True], [False, True]], device=device)
    left = implicit_loss(scores, targets, negatives, "bpr", 0.1)
    right = implicit_loss(scores, targets, negatives, "infonce", 0.1)
    grad_left = torch.autograd.grad(left, scores, retain_graph=True)[0]
    grad_right = torch.autograd.grad(right, scores)[0]
    if not torch.allclose(left, right, atol=1e-6) or not torch.allclose(grad_left, grad_right, atol=1e-5):
        raise RuntimeError("Single-negative likelihood or gradient differs")
    for layout in ("single", "multi"):
        for mode in modes:
            torch.manual_seed(42)
            model = ContentModel(prepared, layout, mode, device)
            initial = model.learned_state()
            bank = model.catalog(batch_size=256, initial=True)
            sums = profiles(data, bank, weights, "binary", normalize=False)
            users, positive = epoch_samples(data, 42, 1)
            count = 256 if mode == "projection" else 8
            batch_users, positive = users[:count], positive[:count]
            pool, target, negative = candidates(data, batch_users, positive, 42, 1, 0, 128 if mode == "projection" else 16)
            query = F.normalize(sums[batch_users] - bank[positive], dim=1).to(device)
            for loss_name in ("bpr", "infonce"):
                model.restore_learned(initial)
                model.train()
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3,
                                             weight_decay=0.01, foreach=False)
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                value = model(pool)
                loss = implicit_loss(query @ value.T, torch.from_numpy(target).to(device),
                                     torch.from_numpy(negative).to(device), loss_name, 0.1)
                loss.backward()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite smoke loss")
                if mode == "lora" and not any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                               for name, p in model.named_parameters() if "lora_B" in name):
                    raise RuntimeError("LoRA adapters received no gradient")
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                               error_if_nonfinite=True, foreach=False)
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError("Frozen backbone accumulated gradients")
                optimizer.step()
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_seconds = time.monotonic() - started
                changed = model.learned_state()
                if not any(not torch.equal(initial[k], changed[k]) for k in initial):
                    raise RuntimeError("Trainable parameters did not update")
                adapter_updated = any("lora_" in k and not torch.equal(initial[k], changed[k]) for k in initial)
                if mode == "lora" and not adapter_updated:
                    raise RuntimeError("LoRA adapters did not update")
                payload = {"state_dict": changed, "optimizer": _cpu_tree(optimizer.state_dict()), "rng": rng_state(device)}
                checkpoint_path = output / f"{layout}_{mode}_{loss_name}.pt"
                save_checkpoint(checkpoint_path, payload)
                loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                model.restore_learned(initial)
                model.restore_learned(loaded["state_dict"])
                optimizer.load_state_dict(loaded["optimizer"])
                restore_rng(loaded["rng"], device)
                restored = model.learned_state()
                if any(not torch.equal(restored[k], changed[k]) for k in changed):
                    raise RuntimeError("Checkpoint parameters failed to round-trip")
                record = {"layout": layout, "mode": mode, "loss": loss_name, "value": float(loss.detach()),
                          "batch_users": count, "candidate_items": len(pool),
                          "seconds": time.monotonic() - started, "memory": memory_snapshot(device),
                          "step_seconds": step_seconds, "checkpoint_roundtrip": True, "adapter_updated": adapter_updated}
                checks.append(record)
                emit(output, "smoke_batch", **record)
            if mode == "projection":
                # Check sparse profiles against direct dense aggregation for selected real users.
                dense = torch.from_numpy(data.batch(batch_users[:8])) @ bank
                if not torch.allclose(F.normalize(dense, dim=1), F.normalize(sums[batch_users[:8]], dim=1), atol=1e-5):
                    raise RuntimeError("Sparse and dense history profiles disagree")
                metrics, _ = score_catalog(prepared, bank, "valid", "binary", [0.0, 1.0], device, user_limit=64)
                eligible = np.flatnonzero(np.diff(data.valid.offsets) > 0)[:64]
                history = torch.from_numpy(data.batch(eligible)).to(device)
                with torch.inference_mode():
                    cf = baseline_model(prepared, str(device))(history)[0]
                    top = cf.masked_fill(history.bool(), -torch.inf).topk(100, sorted=True).indices.cpu().numpy()
                keys = eligible[:, None] * data.n_items + top
                positions = np.searchsorted(data.valid.keys, keys)
                matches = ((positions < len(data.valid.keys)) &
                           (data.valid.keys[np.minimum(positions, len(data.valid.keys) - 1)] == keys))
                original = ranking_values(matches, np.diff(data.valid.offsets)[eligible])
                if abs(metrics["1.0"]["recall@20"] - float(original["recall@20"].mean())) > 1e-10:
                    raise RuntimeError("Alpha=1 failed to reproduce Mult-VAE ranking")
                checks.append({"layout": layout, "fusion_endpoints": True, "dense_sparse_profiles": True,
                               "evaluation_users": 64, "metrics": metrics})
            del model, optimizer, bank, sums, initial, changed, payload, loaded, restored
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
    result = {"passed": True, "code_hash": code_hash(), "prepared_fingerprint": manifest["fingerprint"],
              "unit_tests_run": False, "modes": list(modes), "checks": checks}
    times = {mode: float(np.mean([c["step_seconds"] for c in checks if c.get("mode") == mode]))
             for mode in modes}
    estimate = {"caveat": "Single-batch extrapolation, excludes catalog refresh/evaluation; not an ETA."}
    for mode in modes:
        epochs, batch_size = (10, 256) if mode == "projection" else (3, 8)
        estimate[f"{mode}_24_runs_max_epochs"] = 24 * epochs * np.ceil(data.n_users / batch_size) * times[mode] / 3600
    result["rough_training_step_hours"] = estimate
    write_json(output / "smoke.json", result)
    lines = ["# Operational smoke verification", "", f"All {4 * len(modes)} real-data forward/backward checks passed on the requested device.", "",
             "| Layout | Mode | Loss | Step seconds | Adapter updated | Checkpoint restored |", "|---|---|---|---:|---|---|"]
    for record in checks:
        if "mode" in record:
            lines.append(f"| {record['layout']} | {record['mode']} | {record['loss']} | {record['step_seconds']:.3f} | {record['adapter_updated']} | True |")
    if "projection" in modes:
        lines.extend(["", "Binary sparse profiles matched direct aggregation. Alpha=1 matched the baseline evaluator for 64 validation users."])
    lines.extend(["", "Single-negative BPR/InfoNCE likelihoods and gradients agreed. No unit tests were run.", "",
                  "## Rough compute extrapolation", ""])
    for mode in modes:
        lines.extend([f"{mode.capitalize()} training steps at the epoch caps: {estimate[f'{mode}_24_runs_max_epochs']:.1f} hours across 24 runs.", ""])
    lines.extend(["This is a single-batch extrapolation, excludes catalog refresh/evaluation and thermal variability, and is not a completion-time estimate.", ""])
    (output / "report.md").write_text("\n".join(lines))
    return result
