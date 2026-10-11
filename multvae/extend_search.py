"""Promote four objective configurations to 50 epochs, then one to 200."""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import torch

from multvae.data import Dataset, load_dataset
from multvae.evaluate import evaluate
from multvae.report import generate_report
from multvae.train import ROOT, model_from_config, run_training, save_checkpoint, verify_checkpoint, write_json
from training_logging.wandb import TrainingTracker, TrackingOptions, add_arguments, options_from_args


def file_sha256(path: Path) -> str:
    """Hash a checkpoint or configuration without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote_checkpoint(source: Path, destination: Path, data: Dataset) -> dict:
    """Fork weights, optimizer, RNG, history and best checkpoint into a new run."""
    provenance_path = destination / "promotion.json"
    source_digest = file_sha256(source)
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if provenance["source_sha256"] != source_digest or provenance["source"] != str(source):
            raise ValueError("Promotion source changed; use a new experiment directory")
        return provenance
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Nonempty destination without promotion provenance: {destination}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    verify_checkpoint(checkpoint, data)
    destination.mkdir(parents=True, exist_ok=True)
    provenance = {"source": str(source), "source_sha256": source_digest,
                  "source_epoch": checkpoint["epoch"], "source_global_step": checkpoint["global_step"],
                  "source_best_epoch": checkpoint["best_checkpoint"]["epoch"],
                  "source_stop_reason": checkpoint.get("stop_reason"),
                  "weights_optimizer_rng_history_retained": True, "wandb_identity": "new_run"}
    # Keep the source files and their W&B runs immutable; the promoted run gets
    # a fresh logging identity while retaining every numerical training state.
    checkpoint["tracking"] = None
    checkpoint["best_checkpoint"]["tracking"] = None
    checkpoint["config"] = dict(checkpoint["config"], output_dir=str(destination))
    checkpoint["best_checkpoint"]["config"] = dict(
        checkpoint["best_checkpoint"]["config"], output_dir=str(destination))
    save_checkpoint(destination / "last.pt", checkpoint)
    save_checkpoint(destination / "best.pt", checkpoint["best_checkpoint"])
    write_json(destination / "history.json", checkpoint["history"])
    write_json(provenance_path, provenance)
    return provenance


def continue_run(
    source: Path, destination: Path, stage: str, max_epochs: int, patience: int,
    data: Dataset, options: TrackingOptions,
) -> dict:
    """Resume a promoted run, restarting KL and patience once per new stage."""
    provenance = promote_checkpoint(source, destination, data)
    marker = destination / "stage_complete.json"
    if not marker.exists():
        checkpoint = torch.load(destination / "last.pt", map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        keys = ("device", "hidden_dim", "latent_dim", "dropout", "learning_rate", "batch_size",
                "eval_batch_size", "anneal_steps", "anneal_cap", "negative_count", "early_stop_start",
                "seed", "test_input", "nll_weight", "bpr_weight")
        values = {key: config[key] for key in keys}
        values["early_stop_start"] = "anneal-cap" if config["anneal_cap"] > 0 else "immediate"
        fresh_stage = config.get("continuation_stage") != stage
        args = argparse.Namespace(
            **values, data_dir=Path(config["data_dir"]), output_dir=destination,
            max_epochs=max_epochs, patience=patience, skip_test=True, require_wandb_online=True,
            resume=destination / "last.pt", continuation_stage=stage,
            continuation_source=str(source), continuation_source_sha256=provenance["source_sha256"],
            restart_early_stopping=fresh_stage, restart_kl_annealing=fresh_stage,
        )
        print(json.dumps({"event": "promotion_start", "stage": stage, "run": destination.name,
                          "from_epoch": checkpoint["epoch"], "max_epochs": max_epochs,
                          "patience": patience, "restart_early_stopping": args.restart_early_stopping,
                          "restart_kl_annealing": args.restart_kl_annealing}), flush=True)
        del checkpoint
        with TrainingTracker(destination, f"multvae_{stage}",
                             replace(options, name=f"{destination.name}_{stage}")) as tracker:
            run_training(args, tracker, data)
            if not tracker.active:
                raise RuntimeError("Online logging is required for promoted training")
        write_json(marker, {"stage": stage, "max_epochs": max_epochs, "patience": patience})
        torch.mps.empty_cache()
    metrics = json.loads((destination / "metrics.json").read_text())
    config = json.loads((destination / "config.json").read_text())
    tracking = json.loads((destination / "wandb_state.json").read_text())
    result = {"run": destination.name, "stage": stage,
              **{key: config[key] for key in ("nll_weight", "bpr_weight", "anneal_cap")},
              "source_epoch": provenance["source_epoch"], "epochs_completed": metrics["epochs_completed"],
              "new_epochs": metrics["epochs_completed"] - provenance["source_epoch"],
              "selected_epoch": metrics["selected_epoch"], "stop_reason": metrics["stop_reason"],
              "recall@20": metrics["full_validation"]["recall@20"],
              "ndcg@20": metrics["full_validation"]["ndcg@20"],
              "wandb_url": f"https://wandb.ai/{tracking['entity']}/{tracking['project']}/runs/{tracking['run_id']}",
              "directory": str(destination)}
    print(json.dumps({"event": "promotion_complete", **result}), flush=True)
    return result


def render_report(output: Path, status: str, stage50: list[dict], stage200: dict | None = None) -> dict:
    """Write an English staged-selection report and measured validation curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = json.loads((output / "plan.json").read_text())
    ranked = sorted(stage50, key=lambda row: (-row["recall@20"], -row["ndcg@20"]))
    summary = {"status": status, "stage50": ranked, "stage200": stage200,
               "selection_metric": "full-validation Recall@20; NDCG@20 breaks inter-run ties"}
    write_json(output / "summary.json", summary)
    lines = ["# Mult-VAE staged objective training", "", f"Status: **{status}**.", "",
             "The top four configurations from the ten-epoch grid are selected using validation Recall@20, "
             "with NDCG@20 breaking exact inter-run ties. Each is continued from its last checkpoint toward "
             "50 total epochs, retaining weights, Adam state, RNG, sampled-pair schedule and all epoch history. "
             "Original ten-epoch artifacts are preserved. Promoted stages use separate W&B run identities.", "",
             f"Stage 50 uses patience **{plan['patience50']}**. The validation winner "
             f"then continues toward 200 total epochs with patience **{plan['patience200']}**. "
             "Both promoted stages restart KL beta at zero and start a new patience window once; the global "
             "best checkpoint, optimizer and RNG state remain intact. Early stopping for nonzero KL starts "
             "after annealing reaches its cap. Early stopping can finish a stage before its maximum budget. "
             "Continuing an interrupted stage does not reset KL or patience again.", "",
             "Architecture, full train input/reconstruction target, objective weights, learning rate, "
             "batch size 500, 16 negatives and seed 42 are unchanged. KL uses "
             "`min(cap, (global_step - stage_start_step)/6500)`, reaching cap 0.2 after five new epochs. "
             "Only validation selects configurations and checkpoints. Test is evaluated for the final stage-200 "
             "selected checkpoint after writing a selection lock.", "", "## Stage 50", "",
             "| Run | NLL | BPR | KL cap | Total epochs | New epochs | Best epoch | Recall@20 | NDCG@20 | Stop | W&B |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for row in ranked:
        lines.append(f"| {row['run']} | {row['nll_weight']:g} | {row['bpr_weight']:g} | {row['anneal_cap']:g} | "
                     f"{row['epochs_completed']} | {row['new_epochs']} | {row['selected_epoch']} | "
                     f"{row['recall@20']:.6f} | {row['ndcg@20']:.6f} | {row['stop_reason']} | "
                     f"[Run]({row['wandb_url']}) |")
    if stage200:
        lines.extend(["", "## Stage 200", "",
                      f"Configuration: **{stage200['run']}**. Trained through epoch **{stage200['epochs_completed']}** "
                      f"({stage200['new_epochs']} new epochs); stop: `{stage200['stop_reason']}`. "
                      f"Selected checkpoint: epoch **{stage200['selected_epoch']}**. Validation Recall@20 "
                      f"**{stage200['recall@20']:.6f}**, NDCG@20 **{stage200['ndcg@20']:.6f}**.", "",
                      f"[Stage 200 W&B run]({stage200['wandb_url']})."])
    final_path = output / "final_metrics.json"
    if final_path.exists():
        final = json.loads(final_path.read_text())
        lines.extend(["", "## Final selected-checkpoint metrics", "",
                      "| Split | Recall@20 | NDCG@20 |", "| --- | ---: | ---: |"])
        for key, label in (("full_validation", "Validation"), ("test", "Test")):
            lines.append(f"| {label} | {final[key]['recall@20']:.6f} | {final[key]['ndcg@20']:.6f} |")
        lines.extend(["", f"Checkpoint: `{final['checkpoint']}`."])
    lines.extend(["", "## Interpretation", "",
                  "Epoch limits count the full training trajectory, including the original ten epochs. "
                  "One seed and checkpoint selection do not establish statistical significance across seeds. "
                  "The original grid's test metric is historical context, and is not used for stage promotion. "
                  "Final test uses train+validation encoder input and masks both histories, without changing weights.", "",
                  "![Validation Recall@20 over the continued trajectories](validation_curves.png)", "",
                  "Artifacts: `plan.json`, `data_audit.json`, `selection_50.json`, `selection_200.json`, "
                  "`summary.json`, `final_metrics.json`, and each promoted run's checkpoint, history, report and provenance.", ""])
    (output / "report.md").write_text("\n".join(lines))
    figure, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    for row in ranked:
        history = json.loads((Path(row["directory"]) / "history.json").read_text())
        axis.plot([r["epoch"] for r in history], [r["validation"]["recall@20"] for r in history], label=row["run"])
    if stage200:
        history = json.loads((Path(stage200["directory"]) / "history.json").read_text())
        axis.plot([r["epoch"] for r in history], [r["validation"]["recall@20"] for r in history],
                  label=f"{stage200['run']} (stage 200)", linestyle="--", linewidth=2)
    axis.axvline(10, color="#64748b", linestyle=":", label="Promotion from 10 epochs")
    axis.set(xlabel="Total epoch", ylabel="Full-validation Recall@20", title="Promoted objective configurations")
    if ranked:
        axis.legend(fontsize=8)
    figure.savefig(output / "validation_curves.png", dpi=160)
    plt.close(figure)
    return summary


def main() -> None:
    """Run validation-only promotion followed by final selected-model evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "artifacts/multvae_objectives_mps_seed42")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/multvae_objectives_extended_mps_seed42")
    parser.add_argument("--patience200", type=int, default=10)
    add_arguments(parser)
    args = parser.parse_args()
    options = options_from_args(args, parser)
    if options.mode != "online" or args.patience200 < 1:
        parser.error("Require W&B online and positive patience200")
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("MPS without CPU fallback is required")
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    initial = json.loads((source / "summary.json").read_text())
    if initial["status"] != "completed" or len(initial["runs"]) < 4:
        parser.error("Source grid must be completed with at least four configurations")
    top_four = sorted(initial["runs"], key=lambda r: (-r["recall@20"], -r["ndcg@20"]))[:4]
    sources = {r["run"]: source / "runs" / r["run"] / "last.pt" for r in top_four}
    first = torch.load(next(iter(sources.values())), map_location="cpu", weights_only=True)
    data = load_dataset(Path(first["config"]["data_dir"]))
    del first
    plan = {"source": str(source), "source_summary_sha256": file_sha256(source / "summary.json"),
            "selected_four": [{"run": r["run"], "recall@20": r["recall@20"], "ndcg@20": r["ndcg@20"]} for r in top_four],
            "source_checkpoints_sha256": {name: file_sha256(path) for name, path in sources.items()},
            "stage50_max_epochs": 50, "stage200_max_epochs": 200, "patience50": 3,
            "restart_kl_per_stage": True, "restart_patience_per_stage": True,
            "early_stopping_start": "anneal-cap for nonzero KL; immediate for no-KL",
            "patience200": args.patience200, "data_canonical_sha256": data.audit["canonical_sha256"],
            "source_sha256": {path: file_sha256(ROOT / path) for path in
                              ("multvae/model.py", "multvae/data.py", "multvae/train.py", "multvae/evaluate.py",
                               "multvae/extend_search.py", "multvae/report.py", "training_logging/wandb.py")}}
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Continuation plan changed; use a new experiment directory")
    write_json(plan_path, plan)
    write_json(output / "data_audit.json", data.audit)
    stage50 = []
    render_report(output, "stage50_running", stage50)
    for row in top_four:
        result = continue_run(sources[row["run"]], output / "stage50" / row["run"], "stage50", 50, 3, data, options)
        stage50.append(result)
        render_report(output, "stage50_running", stage50)
    winner = sorted(stage50, key=lambda r: (-r["recall@20"], -r["ndcg@20"]))[0]
    write_json(output / "selection_50.json", {"selection_metric": "validation Recall@20", "winner": winner,
                                             "candidates": stage50, "test_used_for_selection": False})
    render_report(output, "stage200_running", stage50)
    stage200 = continue_run(Path(winner["directory"]) / "last.pt", output / "stage200" / winner["run"],
                            "stage200", 200, args.patience200, data, options)
    checkpoint_path = Path(stage200["directory"]) / "best.pt"
    lock = {"run": stage200["run"], "selected_epoch": stage200["selected_epoch"],
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
            "selection_metric": "validation Recall@20", "test_used_for_selection": False}
    lock_path = output / "selection_200.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Final selected checkpoint changed")
    write_json(lock_path, lock)
    final_path = output / "final_metrics.json"
    if not final_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        verify_checkpoint(checkpoint, data)
        model = model_from_config(checkpoint["config"], torch.device("mps"))
        model.load_state_dict(checkpoint["state_dict"])
        final = {**lock, "full_validation": checkpoint["validation"],
                 "test": evaluate(model, data, "test", 500, "train-valid")}
        write_json(final_path, final)
    else:
        final = json.loads(final_path.read_text())
        if final["checkpoint_sha256"] != lock["checkpoint_sha256"]:
            raise ValueError("Saved final metrics refer to a different checkpoint")
    directory = Path(stage200["directory"])
    metrics = json.loads((directory / "metrics.json").read_text())
    metrics["test"] = final["test"]
    write_json(directory / "metrics.json", metrics)
    with TrainingTracker(directory, "multvae_stage200", replace(options, name=f"{stage200['run']}_stage200")) as tracker:
        tracker.start(json.loads((directory / "config.json").read_text()),
                      json.loads((directory / "wandb_state.json").read_text()), resumed=True)
        if not tracker.active:
            raise RuntimeError("Online logging is required for final metrics")
        tracker.complete(metrics)
        if not tracker.active:
            raise RuntimeError("Final metric upload failed")
    generate_report(directory)
    render_report(output, "completed", stage50, stage200)
    print(json.dumps({"event": "extension_complete", "stage50_winner": winner, "stage200": stage200,
                      "test": final["test"], "report": str(output / "report.md")}), flush=True)


if __name__ == "__main__":
    main()
