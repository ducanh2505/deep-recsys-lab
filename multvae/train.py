"""Train the paper's Mult-VAE baseline on existing MovieLens splits."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from multvae.data import Dataset, load_dataset
from multvae.evaluate import CUTOFFS, evaluate
from multvae.model import MultVAE, objective


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    """Atomically replace a JSON artifact with serializable run state."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path: Path, payload: dict) -> None:
    """Atomically save a weights_only-compatible checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _cpu_tree(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def environment() -> dict:
    """Record software and hardware without collecting device identifiers."""
    info = {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "os": platform.platform(),
            "architecture": platform.machine(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}
    if platform.system() == "Darwin":
        for name, key in (("chip", "machdep.cpu.brand_string"), ("ram_bytes", "hw.memsize"),
                          ("cpu_cores", "hw.ncpu")):
            result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True)
            info[name] = result.stdout.strip() if name == "chip" else int(result.stdout.strip())
    return info


def verify_checkpoint(checkpoint: dict, data: Dataset) -> None:
    """Reject checkpoints built from different mappings or split content."""
    if not np.array_equal(checkpoint["user_ids"].numpy(), data.user_ids):
        raise ValueError("Checkpoint user mapping does not match data")
    if not np.array_equal(checkpoint["item_ids"].numpy(), data.item_ids):
        raise ValueError("Checkpoint item mapping does not match data")
    if checkpoint["config"]["data_canonical_sha256"] != data.audit["canonical_sha256"]:
        raise ValueError("Checkpoint data fingerprints do not match")


def model_from_config(config: dict, device: torch.device) -> MultVAE:
    """Construct the saved architecture on the requested device."""
    return MultVAE(config["n_items"], config["hidden_dim"], config["latent_dim"], config["dropout"]).to(device)


def synchronize(device: torch.device) -> None:
    """Wait for accelerator work before measuring elapsed time."""
    if device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot(device: torch.device) -> dict[str, float]:
    """Measure current MPS memory and process high-water RSS in MiB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
    return {"mps_allocated_mib": torch.mps.current_allocated_memory() / 2**20 if device.type == "mps" else 0.0,
            "mps_driver_mib": torch.mps.driver_allocated_memory() / 2**20 if device.type == "mps" else 0.0,
            "process_peak_rss_mib": rss_bytes / 2**20}


def main() -> None:
    """Run audited training, resumable checkpoints, final evaluation and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/ml20m_lightgcn")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/multvae_ml20m/baseline_mps_seed42")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--hidden-dim", type=int, default=600)
    parser.add_argument("--latent-dim", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=500)
    parser.add_argument("--anneal-steps", type=int, default=200000)
    parser.add_argument("--anneal-cap", type=float, default=0.2)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early-stop-start", choices=("anneal-cap", "immediate"), default="anneal-cap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-input", choices=("train", "train-valid"), default="train-valid")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-test", action="store_true", help="For smoke checks or validation-only runs")
    args = parser.parse_args()
    if min(args.hidden_dim, args.latent_dim, args.batch_size, args.eval_batch_size,
           args.anneal_steps, args.max_epochs, args.patience) < 1:
        parser.error("Dimensions, batch sizes, anneal-steps, max-epochs and patience must be positive")
    if not 0 <= args.dropout < 1 or not 0 <= args.anneal_cap <= 1 or args.learning_rate <= 0:
        parser.error("Require dropout in [0,1), anneal-cap in [0,1], and positive learning rate")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; training cannot fall back to CPU")
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("Disable PYTORCH_ENABLE_MPS_FALLBACK for this MPS run")
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.resume is None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new directory or pass --resume")
    if args.resume is not None and args.resume.resolve().parent != args.output_dir:
        parser.error("Resume checkpoint must belong to output-dir")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run_started = time.monotonic()
    print(json.dumps({"event": "preflight", "data_dir": str(args.data_dir),
                      "checks": "schema, counts, canonical SHA-256, mappings, disjoint splits"}), flush=True)
    data = load_dataset(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"device_used": args.device, "n_users": data.n_users, "n_items": data.n_items,
                   "data_canonical_sha256": data.audit["canonical_sha256"],
                   "split_counts": data.audit["split_counts"], "eligible_users": data.audit["eligible_users"],
                   "metric_cutoffs": list(CUTOFFS), "selection_metric": "recall@20",
                   "recall_denominator": "number of target interactions",
                   "validation_input": "train", "test_mask": "train+valid",
                   "environment": environment(),
                   "started_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
                   "paper": "https://arxiv.org/abs/1802.05814",
                   "reference_code": "https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb",
                   "adam": {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0}})
    model = model_from_config(config, device)
    config["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0, foreach=False)
    history = []
    global_step = 0
    best_score = -math.inf
    best_checkpoint = None
    stale_checks = 0
    stopping_active = False
    elapsed_before = 0.0
    peaks = {key: 0.0 for key in memory_snapshot(device)}
    stop_reason = "max_epochs"
    previous_stop = None
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        verify_checkpoint(saved, data)
        for key in ("hidden_dim", "latent_dim", "dropout", "batch_size", "learning_rate",
                    "anneal_steps", "anneal_cap", "patience", "early_stop_start", "seed", "test_input", "device"):
            if config[key] != saved["config"][key]:
                raise ValueError(f"Resume config mismatch for {key}")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        global_step = saved["global_step"]
        best_checkpoint = saved["best_checkpoint"]
        best_score = best_checkpoint["validation"]["recall@20"]
        stale_checks = saved["stale_checks"]
        stopping_active = saved["stopping_active"]
        elapsed_before = saved["elapsed_seconds"]
        peaks = saved["observed_memory_peaks"]
        previous_stop = saved.get("stop_reason")
        torch.set_rng_state(saved["torch_rng_state"])
        rng.bit_generator.state = saved["numpy_rng_state"]
        if device.type == "mps":
            torch.mps.set_rng_state(saved["mps_rng_state"])
        for key in ("started_at", "environment"):
            config[key] = saved["config"][key]
        # The embedded best checkpoint makes interrupted multi-file writes recoverable.
        save_checkpoint(args.output_dir / "best.pt", best_checkpoint)
    write_json(args.output_dir / "config.json", config)
    write_json(args.output_dir / "data_manifest.json", data.manifest)
    write_json(args.output_dir / "data_audit.json", data.audit)

    def emit(record: dict) -> None:
        line = json.dumps(record, allow_nan=False)
        print(line, flush=True)
        with (args.output_dir / "run.log").open("a") as stream:
            stream.write(line + "\n")

    def observe_memory() -> None:
        for key, value in memory_snapshot(device).items():
            peaks[key] = max(peaks[key], value)

    emit({"event": "resume" if args.resume else "start", "config": config,
          "global_step": global_step, "data_audit": data.audit})
    first_epoch = len(history) + 1
    epochs = range(first_epoch, args.max_epochs + 1) if previous_stop != "early_stopping" else ()
    if previous_stop == "early_stopping":
        stop_reason = previous_stop
    for epoch in epochs:
        synchronize(device)
        epoch_started = time.monotonic()
        model.train()
        order = rng.permutation(data.n_users)
        totals = torch.zeros(4, device=device)
        beta_sum = 0.0
        beta_first = min(args.anneal_cap, global_step / args.anneal_steps)
        batches = math.ceil(data.n_users / args.batch_size)
        for batch, start in enumerate(range(0, data.n_users, args.batch_size), start=1):
            users = order[start:start + args.batch_size]
            inputs = torch.from_numpy(data.batch(users)).to(device)
            beta = min(args.anneal_cap, global_step / args.anneal_steps)
            optimizer.zero_grad(set_to_none=True)
            logits, mean, logvar = model(inputs)
            loss, nll, kl = objective(inputs, logits, mean, logvar, beta)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch}")
            loss.backward()
            if not bool(torch.stack([torch.isfinite(parameter.grad).all()
                                     for parameter in model.parameters()]).all()):
                raise FloatingPointError(f"Non-finite gradient at epoch {epoch}, batch {batch}")
            optimizer.step()
            totals += torch.stack((loss.detach(), nll.detach(), kl.detach(), beta * kl.detach())) * len(users)
            beta_sum += beta * len(users)
            global_step += 1
            if batch == 1 or batch % 100 == 0 or batch == batches:
                observe_memory()
                emit({"event": "batch", "epoch": epoch, "batch": batch, "batches": batches,
                      "global_step": global_step, "beta": beta})
        synchronize(device)
        train_seconds = time.monotonic() - epoch_started
        losses = (totals / data.n_users).cpu().tolist()
        validation_started = time.monotonic()
        valid = evaluate(model, data, "valid", args.eval_batch_size, args.test_input)
        synchronize(device)
        validation_seconds = time.monotonic() - validation_started
        improved = valid["recall@20"] > best_score
        if improved:
            best_score = valid["recall@20"]
            best_checkpoint = {"state_dict": _cpu_tree(model.state_dict()),
                               "user_ids": torch.from_numpy(data.user_ids.copy()),
                               "item_ids": torch.from_numpy(data.item_ids.copy()),
                               "config": config, "epoch": epoch, "global_step": global_step,
                               "beta": beta, "validation": valid}
            save_checkpoint(args.output_dir / "best.pt", best_checkpoint)
        ready = args.early_stop_start == "immediate" or beta >= args.anneal_cap
        if ready and not stopping_active:
            stopping_active = True
            stale_checks = 0
        elif stopping_active:
            stale_checks = 0 if improved else stale_checks + 1
        record = {"epoch": epoch, "global_step": global_step,
                  "train_loss": losses[0], "train_nll": losses[1],
                  "train_kl": losses[2], "train_weighted_kl": losses[3],
                  "beta_first": beta_first, "beta_last": beta,
                  "beta_mean": beta_sum / data.n_users,
                  "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                  "validation": valid, "improved": improved,
                  "selected_epoch": best_checkpoint["epoch"],
                  "early_stopping_active": stopping_active, "stale_checks": stale_checks}
        history.append(record)
        observe_memory()
        should_stop = stopping_active and stale_checks >= args.patience
        if should_stop:
            stop_reason = "early_stopping"
        last = {"state_dict": _cpu_tree(model.state_dict()), "optimizer": _cpu_tree(optimizer.state_dict()),
                "user_ids": best_checkpoint["user_ids"], "item_ids": best_checkpoint["item_ids"],
                "config": config, "epoch": epoch, "global_step": global_step,
                "best_checkpoint": best_checkpoint, "history": history,
                "stale_checks": stale_checks, "stopping_active": stopping_active,
                "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": rng.bit_generator.state,
                "mps_rng_state": torch.mps.get_rng_state() if device.type == "mps" else None,
                "observed_memory_peaks": peaks, "elapsed_seconds": elapsed_before + time.monotonic() - run_started,
                "stop_reason": "early_stopping" if should_stop else None}
        save_checkpoint(args.output_dir / "last.pt", last)
        write_json(args.output_dir / "history.json", history)
        emit({"event": "epoch", **record, "observed_memory_peaks": peaks})
        if epoch == first_epoch:
            emit({"event": "runtime_estimate", "seconds_per_epoch": time.monotonic() - epoch_started,
                  "remaining_epochs_upper_bound": args.max_epochs - epoch,
                  "remaining_seconds_upper_bound": (time.monotonic() - epoch_started) * (args.max_epochs - epoch)})
        if should_stop:
            break
    if best_checkpoint is None:
        raise ValueError("No trained checkpoint is available")
    model.load_state_dict(best_checkpoint["state_dict"])
    emit({"event": "final_evaluation", "selected_epoch": best_checkpoint["epoch"], "stop_reason": stop_reason})
    final_started = time.monotonic()
    full_valid = evaluate(model, data, "valid", args.eval_batch_size, args.test_input)
    metrics = {"selected_epoch": best_checkpoint["epoch"], "selected_beta": best_checkpoint["beta"],
               "epochs_completed": len(history), "global_step": global_step, "stop_reason": stop_reason,
               "selection_metric": "recall@20", "selection_validation": best_checkpoint["validation"],
               "full_validation": full_valid, "test_input": args.test_input}
    if not args.skip_test:
        metrics["test"] = evaluate(model, data, "test", args.eval_batch_size, args.test_input)
    synchronize(device)
    observe_memory()
    metrics.update({"final_evaluation_seconds": time.monotonic() - final_started,
                    "total_seconds": elapsed_before + time.monotonic() - run_started,
                    "observed_memory_peaks": peaks,
                    "completed_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()})
    write_json(args.output_dir / "metrics.json", metrics)
    from multvae.report import generate_report
    generate_report(args.output_dir)
    emit({"event": "complete", "metrics": metrics, "report": str(args.output_dir / "report.md")})


if __name__ == "__main__":
    main()
