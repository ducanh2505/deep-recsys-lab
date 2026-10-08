"""Staged execution of the fixed 48-run design and validation-only selection."""

import gc
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from content_recsys.common import (MODEL_ID, REVISION, ROOT, code_hash, device_for, emit,
                                   exclusive_lock, sha256, write_json)
from content_recsys.data import load_prepared, prepare
from content_recsys.encoder import ensure_frozen
from content_recsys.evaluate import evaluate_run
from content_recsys.train import run


def configurations(prepared: Path, root: Path, device: str, eval_batch_size: int = 256) -> list[dict]:
    """Return all 32 prespecified seed-42 search configurations."""
    result = []
    for layout, mode, loss in itertools.product(("single", "multi"), ("projection", "lora"), ("bpr", "infonce")):
        for learning_rate, temperature in itertools.product(
                (3e-4, 1e-3) if mode == "projection" else (1e-4, 2e-4), (0.05, 0.10)):
            name = f"{layout}_{mode}_{loss}_lr{learning_rate:g}_tau{temperature:g}_seed42"
            result.append({"prepared": str(prepared.resolve()), "output": str((root / "runs" / name).resolve()),
                           "layout": layout, "mode": mode, "loss": loss, "seed": 42,
                           "learning_rate": learning_rate, "temperature": temperature, "device": device,
                           "batch_size": 256 if mode == "projection" else 8,
                           "negative_count": 128 if mode == "projection" else 16,
                           "max_epochs": 10 if mode == "projection" else 3,
                           "patience": 3 if mode == "projection" else 2,
                           "eval_batch_size": eval_batch_size, "checkpoint_batches": 200})
    # Finish inexpensive branches first; every branch retains its prespecified budget.
    return sorted(result, key=lambda config: (config["mode"] == "lora", config["layout"], config["loss"],
                                               config["learning_rate"], config["temperature"]))


def run_benchmark(root: Path, data_dir: Path, catalog: Path, baseline: Path,
                  device_name: str = "mps", eval_batch_size: int = 256,
                  stage: str = "projection") -> dict:
    """Execute projection locally, or explicitly opt into the entire design."""
    if stage not in ("projection", "all"):
        raise ValueError("Stage must be projection or all")
    with exclusive_lock(root / ".benchmark.lock"):
        return _run_benchmark(root, data_dir, catalog, baseline, device_name, eval_batch_size, stage)


def defer_lora(root: Path, prepared: Path, manifest: dict, eval_batch_size: int) -> Path:
    """Record the GPU experiment specification without launching any GPU jobs."""
    configurations_gpu = []
    for config in configurations(prepared, root, "cuda", eval_batch_size):
        if config["mode"] != "lora":
            continue
        # Paths and execution environment must be resolved on the future GPU host.
        specification = {k: v for k, v in config.items() if k not in ("prepared", "output")}
        specification["run_name"] = Path(config["output"]).name
        configurations_gpu.append(specification)
    path = root / "deferred_lora_gpu.json"
    value = {
        "status": "deferred_by_user", "execution_ready": False, "target_device": "cuda",
        "planned_training_runs": 24, "search_runs": configurations_gpu,
        "replications": {"seeds": [43, 44], "runs": 8,
                         "rule": "Replicate the content-binary validation winner of each of four LoRA branches."},
        "model_id": MODEL_ID, "model_revision": REVISION,
        "prepared_fingerprint": manifest["fingerprint"], "sources": manifest["sources"],
        "reference_code_hash": code_hash(), "dependency_lock_sha256": sha256(ROOT / "uv.lock"),
        "pending_requirements": [
            "Provide a GPU host and resolve source, baseline, prepared and output paths there.",
            "Verify CUDA RNG checkpoint/restore and accelerator memory reporting on the GPU host.",
            "Run real-data CUDA smoke, memory and resume checks before launching LoRA.",
            "Keep paired loss batches, float32, training budgets and validation selection unchanged."],
        "test_policy": "Keep learned-model test evaluation deferred until all 48 runs and validation choices are locked.",
        "partial_mac_runs": [str(p.resolve()) for p in sorted((root / "runs").glob("*lora*"))
                             if (p / "config.json").exists() and not (p / "training.json").exists()],
    }
    write_json(path, value)
    return path


def save_selection_lock(path: Path, lock: dict) -> None:
    """Retain immutable validation choices when a completed stage is resumed."""
    if path.exists():
        old = json.loads(path.read_text())
        if {k: v for k, v in old.items() if k != "created_unix"} != {k: v for k, v in lock.items() if k != "created_unix"}:
            raise ValueError("Pre-test selection lock is immutable; computed choices differ")
    else:
        write_json(path, lock)


def _run_benchmark(root: Path, data_dir: Path, catalog: Path, baseline: Path,
                   device_name: str = "mps", eval_batch_size: int = 256,
                   stage: str = "projection") -> dict:
    """Resume a stage and defer global test evaluation while GPU LoRA is pending."""
    from content_recsys.report import report
    root.mkdir(parents=True, exist_ok=True)
    prepared = root / "prepared"
    device = device_for(device_name)
    status_path = root / "benchmark.json"
    existing = json.loads(status_path.read_text()) if status_path.exists() else None
    if existing and existing.get("code_hash") != code_hash():
        raise ValueError("Benchmark implementation changed; choose a new output root")
    if existing and existing.get("dependency_lock_sha256") != sha256(ROOT / "uv.lock"):
        raise ValueError("Benchmark dependency lock changed")
    status = existing or {"phase": "prepare", "completed": False, "planned_training_runs": 48,
                           "finished_training_runs": 0, "runs": [], "code_hash": code_hash(),
                           "dependency_lock_sha256": sha256(ROOT / "uv.lock"), "started_unix": time.time()}
    if status.get("last_error"):
        status.setdefault("recovered_errors", []).append(status.pop("last_error"))
    status.update({"requested_stage": stage, "active_run": None})
    if stage == "projection":
        status.update({"lora_status": "deferred_to_gpu", "projection_planned_training_runs": 24,
                       "lora_planned_training_runs": 24, "test_status": "deferred_until_full_selection_lock"})
    write_json(status_path, status)
    report(root)
    try:
        manifest = prepare(data_dir, catalog, baseline, prepared)
        status["prepared_fingerprint"] = manifest["fingerprint"]
        if stage == "projection":
            status["deferred_lora_plan"] = str(defer_lora(root, prepared, manifest, eval_batch_size).resolve())
        write_json(status_path, status)
        ensure_frozen(prepared, device)
        # Require smoke verification for this implementation before expensive training.
        from content_recsys.smoke import smoke
        smoke_path = root / "smoke" / "smoke.json"
        modes = ("projection",) if stage == "projection" else ("projection", "lora")
        smoke_result = json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
        verified_modes = {c.get("mode") for c in smoke_result.get("checks", [])}
        if (not smoke_result.get("passed") or smoke_result.get("code_hash") != code_hash()
                or not set(modes).issubset(verified_modes)):
            smoke(prepared, root / "smoke", device, modes=modes)
        configs = [c for c in configurations(prepared, root, device_name, eval_batch_size) if c["mode"] in modes]
        results = {}
        status["phase"] = "hyperparameter_search"
        for config in configs:
            status["active_run"] = config["output"]
            write_json(status_path, status)
            report(root)
            results[config["output"]] = run(config)
            if config["output"] not in status["runs"]:
                status["runs"].append(config["output"])
            status["finished_training_runs"] = len(status["runs"])
            write_json(status_path, status)
            report(root)
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
        winners = []
        for layout, mode, loss in itertools.product(("single", "multi"), modes, ("bpr", "infonce")):
            branch = [c for c in configs if (c["layout"], c["mode"], c["loss"]) == (layout, mode, loss)]
            selected = max(branch, key=lambda c: (results[c["output"]]["validation"]["recall@20"],
                                                  results[c["output"]]["validation"]["ndcg@20"],
                                                  -results[c["output"]]["selected_epoch"], -c["learning_rate"], -c["temperature"]))
            winners.append(selected)
        status["phase"] = "seed_replication"
        selected_configs = list(winners)
        for config in winners:
            for seed in (43, 44):
                original = Path(config["output"])
                replica = {**config, "seed": seed,
                           "output": str(original.with_name(original.name.replace("seed42", f"seed{seed}")))}
                status["active_run"] = replica["output"]
                write_json(status_path, status)
                report(root)
                run(replica)
                selected_configs.append(replica)
                if replica["output"] not in status["runs"]:
                    status["runs"].append(replica["output"])
                status["finished_training_runs"] = len(status["runs"])
                write_json(status_path, status)
                report(root)
                gc.collect()
                if device.type == "mps":
                    torch.mps.empty_cache()
        _, _, _, prepared_manifest = load_prepared(prepared)
        controls = []
        for layout in ("single", "multi"):
            directory = root / "controls" / f"frozen_{layout}"
            directory.mkdir(parents=True, exist_ok=True)
            control_config = {"prepared_fingerprint": manifest["fingerprint"], "layout": layout,
                              "mode": "frozen", "loss": "none", "seed": None}
            write_json(directory / "config.json", control_config)
            if not (directory / "embeddings.npy").exists():
                np.save(directory / "embeddings.npy", np.load(prepared / "frozen" / f"{layout}.npy"))
            controls.append(directory)
        selected_dirs = [Path(c["output"]) for c in selected_configs]
        status["selected_runs"] = [str(p.resolve()) for p in selected_dirs]
        if stage == "projection":
            status["projection_selected_runs"] = list(status["selected_runs"])
        status["controls"] = [str(p.resolve()) for p in controls]
        status["phase"] = "validation_fusion"
        write_json(status_path, status)
        locked_runs = {}
        for directory in selected_dirs + controls:
            status["active_run"] = str(directory)
            write_json(status_path, status)
            values = evaluate_run(prepared, directory, device, batch_users=eval_batch_size)
            locked_runs[str(directory.resolve())] = {"embedding_sha256": values["embedding_sha256"],
                                                     "alphas": {p: v["alpha"] for p, v in values["profiles"].items()}}
        # The deployment choice uses validation only, averaging the three seeds per branch.
        comparisons = []
        for layout, mode, loss in itertools.product(("single", "multi"), modes, ("bpr", "infonce")):
            branch = [Path(c["output"]) for c in selected_configs if (c["layout"], c["mode"], c["loss"]) == (layout, mode, loss)]
            metrics = [json.loads((p / "valid_evaluation.json").read_text())["profiles"]["binary"]["hybrid"] for p in branch]
            comparisons.append({"layout": layout, "mode": mode, "loss": loss,
                                "recall@20": float(np.mean([m["recall@20"] for m in metrics])),
                                "ndcg@20": float(np.mean([m["ndcg@20"] for m in metrics]))})
        deployment = max(comparisons, key=lambda m: (m["recall@20"], m["ndcg@20"]))
        lock = {"prepared_fingerprint": prepared_manifest["fingerprint"], "code_hash": code_hash(),
                "created_unix": time.time(), "runs": locked_runs, "deployment_choice": deployment,
                "validation_comparisons": comparisons, "test_used_for_selection": False}
        if stage == "projection":
            lock["scope"] = "projection_validation_only; global test selection remains pending"
            save_selection_lock(root / "projection_selection_lock.json", lock)
            status.update({"phase": "waiting_for_lora_gpu", "completed": False,
                           "projection_completed": True, "projection_completed_unix": time.time(),
                           "active_run": None, "projection_validation_choice": deployment})
            status.pop("last_error", None)
            write_json(status_path, status)
            report(root)
            emit(root, "projection_stage_complete", training_runs=24,
                 validation_choice=deployment, deferred_lora_runs=24, test_deferred=True)
            return status
        lock_path = root / "selection_lock.json"
        save_selection_lock(lock_path, lock)
        status["phase"] = "locked_test_evaluation"
        write_json(status_path, status)
        for directory in selected_dirs + controls:
            status["active_run"] = str(directory)
            write_json(status_path, status)
            evaluate_run(prepared, directory, device, split="test", batch_users=eval_batch_size, selection_lock=lock_path)
        status.update({"phase": "completed", "completed": True, "active_run": None,
                       "completed_unix": time.time(), "deployment_choice": deployment})
        status.pop("last_error", None)
        write_json(status_path, status)
        report(root)
        emit(root, "benchmark_complete", training_runs=48, deployment_choice=deployment)
        return status
    except BaseException as error:
        status["last_error"] = {"type": type(error).__name__, "message": str(error), "time_unix": time.time()}
        write_json(status_path, status)
        report(root)
        raise
