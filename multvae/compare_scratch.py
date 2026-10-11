"""Retrain the selected hybrid and paper Mult-VAE from scratch on MPS."""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import torch

from multvae.data import load_dataset
from multvae.evaluate import CUTOFFS
from multvae.train import ROOT, run_training, verify_checkpoint, write_json
from training_logging.wandb import TrainingTracker, add_arguments, options_from_args


SELECTED = ROOT / "artifacts/multvae_objectives_extended_mps_seed42/stage200/nll1_bpr1_kl0.2/config.json"
PAPER = ROOT / "artifacts/multvae_ml20m/baseline_mps_seed42/config.json"
PARAMETERS = ("hidden_dim", "latent_dim", "dropout", "learning_rate", "batch_size",
              "eval_batch_size", "anneal_steps", "anneal_cap", "negative_count", "seed",
              "test_input", "nll_weight", "bpr_weight")


def digest(path: Path) -> str:
    """Fingerprint a small source configuration or implementation file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_comparison(output: Path, results: dict) -> None:
    """Save measured differences and validation/annealing curves for both runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hybrid, baseline = results["hybrid"], results["paper_multvae"]
    differences = {}
    for split in ("full_validation", "test"):
        differences[split] = {}
        for cutoff in CUTOFFS:
            for metric in ("recall", "ndcg"):
                key = f"{metric}@{cutoff}"
                value, reference = hybrid["metrics"][split][key], baseline["metrics"][split][key]
                differences[split][key] = {
                    "hybrid_minus_baseline": value - reference,
                    "relative_percent": 100 * (value - reference) / reference if reference else None,
                }
    write_json(output / "comparison.json", {"status": "complete", "runs": results,
                                           "differences": differences})
    lines = ["# Fresh Mult-VAE objective comparison", "",
             "Both runs start at epoch 1 with newly initialized weights, Adam and RNG (seed 42). "
             "No weights or optimizer state from earlier grid/promotion/baseline runs are loaded. "
             "Each has a maximum of 200 epochs and patience 10 after KL reaches its cap. "
             "KL starts at zero once; no stage-boundary resets occur.", "",
             "The shared architecture is I→600→200→600→I, dropout 0.5, Adam lr 0.001, batch size 500, "
             "float32 on MPS. Full train history is the encoder input and multinomial reconstruction target. "
             "Validation Recall@20 selects each checkpoint independently. Test is evaluated only after "
             "that selection; its input is train+validation, with both histories masked in ranking.", "",
             "Hybrid: NLL + BPR + beta*KL; one train positive and 16 uniform train-unseen negatives per user "
             "per epoch, directly on decoder logits. beta=min(0.2, step/6500). "
             "Paper baseline: NLL + beta*KL, beta=min(0.2, step/200000). "
             "The shared trainer logs a diagnostic BPR value for baseline, with zero weight and no gradient contribution.", "",
             "The baseline architecture, objective and KL schedule follow the "
             "[author notebook](https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb). "
             "Both runs use this repository's existing per-user splits and Recall@20 selection, rather than "
             "the paper's separate held-out user protocol and NDCG@100 selection. "
             "These are comparisons on the repository dataset, not a reproduction of the paper's reported numbers.", "",
             "| Model | Epochs completed | Selected epoch | Selected beta | Runtime (min) | W&B |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name, result in results.items():
        metrics = result["metrics"]
        lines.append(f"| {name} | {metrics['epochs_completed']} | {metrics['selected_epoch']} | "
                     f"{metrics['selected_beta']:.6f} | {metrics['total_seconds'] / 60:.2f} | "
                     f"[Run]({result['wandb_url']}) |")
    for split, label in (("full_validation", "Validation"), ("test", "Test")):
        lines.extend(["", f"## {label}", "",
                      "| Metric | Hybrid | Paper baseline | Hybrid − baseline | Relative change |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for cutoff in CUTOFFS:
            for metric in ("recall", "ndcg"):
                key, delta = f"{metric}@{cutoff}", differences[split][f"{metric}@{cutoff}"]
                lines.append(f"| {key} | {hybrid['metrics'][split][key]:.6f} | "
                             f"{baseline['metrics'][split][key]:.6f} | "
                             f"{delta['hybrid_minus_baseline']:+.6f} | {delta['relative_percent']:+.2f}% |")
    lines.extend(["", "## Interpretation", "",
                  "The two configurations differ in both BPR and KL annealing speed. "
                  "Their difference cannot be attributed to BPR alone; an additional matched-annealing "
                  "ablation would be required for that claim. One seed does not quantify variation across "
                  "training seeds. Hyperparameters were previously selected on the same validation split.", "",
                  "![Validation metrics and KL schedules](comparison_curves.png)", "",
                  "See `plan.json`, `data_audit.json`, `verification.json`, `comparison.json`, "
                  "and each run's `config.json`, `history.json`, `best.pt`, `last.pt` and `report.md`.", ""])
    (output / "report.md").write_text("\n".join(lines))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for name, result in results.items():
        history = json.loads((Path(result["directory"]) / "history.json").read_text())
        epochs = [record["epoch"] for record in history]
        for axis, metric in zip(axes[0], ("recall@20", "ndcg@20")):
            axis.plot(epochs, [row["validation"][metric] for row in history], label=name)
            selected = history[result["metrics"]["selected_epoch"] - 1]
            axis.scatter([selected["epoch"]], [selected["validation"][metric]], s=30)
            axis.set(ylabel=metric, title=f"Full validation {metric}")
        axes[1, 0].plot(epochs, [row["beta_last"] for row in history], label=name)
        axes[1, 1].plot(epochs, [row["train_weighted_kl"] for row in history], label=name)
    axes[1, 0].set(ylabel="KL beta", title="Annealing from zero")
    axes[1, 1].set(ylabel="Mean weighted KL", title="KL contribution to training loss")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(output / "comparison_curves.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Train two independent fresh runs, then compare their selected checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/multvae_scratch_comparison_mps_seed42")
    add_arguments(parser)
    args = parser.parse_args()
    options = options_from_args(args, parser)
    if options.mode != "online":
        parser.error("W&B online is required")
    if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("MPS without CPU fallback is required")
    output = args.output_dir.resolve()
    configurations = {}
    for name, source in (("hybrid", SELECTED), ("paper_multvae", PAPER)):
        config = json.loads(source.read_text())
        if name == "paper_multvae":
            config.update(nll_weight=1.0, bpr_weight=0.0, negative_count=16)
        configurations[name] = {key: config[key] for key in PARAMETERS}
    baseline = configurations["paper_multvae"]
    baseline.update(nll_weight=1.0, bpr_weight=0.0, negative_count=16)
    common = set(PARAMETERS) - {"anneal_steps", "bpr_weight"}
    if any(configurations["hybrid"][key] != baseline[key] for key in common):
        raise ValueError("Shared architecture/data/optimizer parameters must match")
    source_hashes = {str(path): digest(path) for path in (SELECTED, PAPER)}
    code_hashes = {str(path.relative_to(ROOT)): digest(path)
                   for path in (ROOT / "multvae").glob("*.py")}
    plan = {"max_epochs": 200, "patience": 10, "early_stop_start": "anneal-cap", "device": "mps",
            "fresh_initialization": True, "kl_starts_at_step": 0, "selection": "validation Recall@20",
            "source_config_sha256": source_hashes, "code_sha256": code_hashes,
            "configurations": configurations}
    plan_path = output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Plan or source code changed; use a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use an empty output directory for a new comparison")
        output.mkdir(parents=True, exist_ok=True)
        write_json(plan_path, plan)
    print(json.dumps({"event": "comparison_preflight", "plan": plan}), flush=True)
    data = load_dataset(ROOT / "data/processed/ml20m_lightgcn")
    write_json(output / "data_audit.json", data.audit)
    results, verification = {}, {}
    for name, configuration in configurations.items():
        directory = output / name
        marker = directory / "complete.json"
        if not marker.exists():
            resume = directory / "last.pt" if (directory / "last.pt").exists() else None
            if resume is None and directory.exists() and any(directory.iterdir()):
                raise ValueError(f"Partial run without resumable checkpoint: {directory}")
            training_args = argparse.Namespace(
                **configuration, device="mps", data_dir=ROOT / "data/processed/ml20m_lightgcn",
                output_dir=directory, max_epochs=200, patience=10, early_stop_start="anneal-cap",
                resume=resume, restart_early_stopping=False, restart_kl_annealing=False,
                skip_test=False, require_wandb_online=True, initialization="fresh_seed42",
                comparison_run=name,
            )
            print(json.dumps({"event": "scratch_run_start", "run": name,
                              "resume_own_interrupted_run": str(resume) if resume else None}), flush=True)
            with TrainingTracker(directory, "multvae_scratch_comparison",
                                 replace(options, name=f"{name}_scratch_seed42_max200")) as tracker:
                run_training(training_args, tracker, data)
                if not tracker.active:
                    raise RuntimeError("Online logging failed during the comparison")
            write_json(marker, {"fresh_origin": True})
            torch.mps.empty_cache()
        metrics = json.loads((directory / "metrics.json").read_text())
        tracking = json.loads((directory / "wandb_state.json").read_text())
        history = json.loads((directory / "history.json").read_text())
        last = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        verify_checkpoint(last, data)
        checks = {
            "history_begins_at_epoch1": history[0]["epoch"] == 1,
            "initial_beta_zero": history[0]["beta_first"] == 0,
            "kl_offset_zero": last["kl_anneal_start_step"] == 0,
            "no_stage_restart": "kl_annealing_restart" not in last["config"],
            "epochs_within_budget": len(history) == metrics["epochs_completed"] <= 200,
            "selected_by_validation": metrics["selected_epoch"] == max(
                history, key=lambda row: row["validation"]["recall@20"])["epoch"],
            "test_after_selection": "test" in metrics,
            "wandb_online": tracking["mode"] == "online" and tracking["online_started"],
            "patience10": last["config"]["patience"] == 10,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Comparison verification failed: {checks}")
        del last
        verification[name] = checks
        results[name] = {"directory": str(directory), "metrics": metrics,
                         "wandb_url": f"https://wandb.ai/{tracking['entity']}/{tracking['project']}/runs/{tracking['run_id']}"}
        write_json(output / "progress.json", {"completed_runs": results})
        print(json.dumps({"event": "scratch_run_complete", "run": name,
                          "epochs": metrics["epochs_completed"], "selected_epoch": metrics["selected_epoch"],
                          "test_recall20": metrics["test"]["recall@20"],
                          "test_ndcg20": metrics["test"]["ndcg@20"]}), flush=True)
    verification["source_configs_unchanged"] = all(digest(Path(path)) == value for path, value in source_hashes.items())
    write_json(output / "verification.json", verification)
    render_comparison(output, results)
    print(json.dumps({"event": "comparison_complete", "report": str(output / "report.md")}), flush=True)


if __name__ == "__main__":
    main()
