"""Create an English Markdown report, CSV history and PNG training figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multvae.evaluate import CUTOFFS


ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#2563eb", "#059669", "#d97706", "#9333ea")


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _metric_table(result: dict) -> str:
    lines = ["| K | Recall@K | Standard error | NDCG@K | Standard error |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    errors = result.get("standard_errors", {})
    for cutoff in CUTOFFS:
        recall, ndcg = f"recall@{cutoff}", f"ndcg@{cutoff}"
        lines.append(f"| {cutoff} | {result[recall]:.6f} | {errors.get(recall, 0):.6f} | "
                     f"{result[ndcg]:.6f} | {errors.get(ndcg, 0):.6f} |")
    return "\n".join(lines)


def generate_report(directory: Path) -> Path:
    """Generate final or clearly labeled in-progress artifacts from measured logs."""
    config = json.loads((directory / "config.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    if not history:
        raise ValueError("A report requires at least one completed epoch")
    metrics_path = directory / "metrics.json"
    complete = metrics_path.exists()
    if complete:
        metrics = json.loads(metrics_path.read_text())
    else:
        selected = max(history, key=lambda record: record["validation"]["recall@20"])
        metrics = {"selected_epoch": selected["epoch"], "selected_beta": selected["beta_last"],
                   "epochs_completed": len(history), "selection_validation": selected["validation"],
                   "full_validation": selected["validation"], "stop_reason": "in_progress"}
    selected_epoch = metrics["selected_epoch"]
    selected_record = next(record for record in history if record["epoch"] == selected_epoch)
    last = history[-1]
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    flat_rows = []
    for record in history:
        row = {key: value for key, value in record.items() if key != "validation"}
        row["validation_users"] = record["validation"]["users"]
        for cutoff in CUTOFFS:
            for metric in ("recall", "ndcg"):
                name = f"{metric}@{cutoff}"
                row[f"validation_{name}"] = record["validation"][name]
                row[f"validation_se_{name}"] = record["validation"]["standard_errors"][name]
        flat_rows.append(row)
    csv_tmp = directory / "history.csv.tmp"
    with csv_tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    csv_tmp.replace(directory / "history.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.formatter.useoffset": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18})
    epochs = [record["epoch"] for record in history]

    def annotate(axes: object) -> None:
        for axis in np.asarray(axes).ravel():
            axis.axvline(selected_epoch, color="#64748b", linestyle="--", linewidth=1,
                         label=f"Selected epoch {selected_epoch}")
            axis.set_xlabel("Epoch")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    for key, label, color in (("train_loss", "Total objective", COLORS[0]),
                              ("train_nll", "Reconstruction NLL", COLORS[1])):
        axes[0].plot(epochs, [record[key] for record in history], label=label, color=color)
    axes[0].set_title("Training reconstruction and objective")
    axes[0].set_ylabel("Mean loss per user")
    for key, label, color in (("train_kl", "Unweighted KL", COLORS[0]),
                              ("train_weighted_kl", "Beta-weighted KL", COLORS[2])):
        axes[1].plot(epochs, [record[key] for record in history], label=label, color=color)
    axes[1].set_title("Posterior regularization")
    axes[1].set_ylabel("Mean KL per user")
    axes[2].plot(epochs, [record["beta_last"] for record in history], color=COLORS[3], label="End-of-epoch beta")
    axes[2].axhline(config["anneal_cap"], color="#475569", linestyle=":", label="Annealing cap")
    axes[2].set_title("KL annealing schedule")
    axes[2].set_ylabel("Beta")
    annotate(axes)
    for axis in axes:
        axis.legend(fontsize=8)
    _save_figure(figure, figures / "training_objective.png")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, metric in zip(axes, ("recall", "ndcg")):
        for cutoff, color in zip(CUTOFFS, COLORS):
            name = f"{metric}@{cutoff}"
            axis.plot(epochs, [record["validation"][name] for record in history], color=color,
                      label=f"{metric.upper()}@{cutoff}")
        axis.set_title(f"Full-validation {metric.upper()}")
        axis.set_ylabel("Macro mean across eligible users")
    annotate(axes)
    for axis in axes:
        axis.legend(fontsize=8)
    _save_figure(figure, figures / "ranking_metrics.png")

    figure, axis = plt.subplots(figsize=(10, 3.8), layout="constrained")
    axis.plot(epochs, [record["train_seconds"] for record in history], color=COLORS[0], label="Training")
    axis.plot(epochs, [record["validation_seconds"] for record in history], color=COLORS[1], label="Full validation")
    axis.set_title("Measured compute time by epoch")
    axis.set_ylabel("Seconds")
    annotate([axis])
    axis.legend(fontsize=8)
    _save_figure(figure, figures / "epoch_timing.png")

    if "test" in metrics:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
        positions = np.arange(len(CUTOFFS))
        for axis, metric in zip(axes, ("recall", "ndcg")):
            for shift, split, label, color in ((-0.18, "full_validation", "Validation (train input)", COLORS[0]),
                                               (0.18, "test", f"Test ({config['test_input']} input)", COLORS[1])):
                result = metrics[split]
                values = [result[f"{metric}@{cutoff}"] for cutoff in CUTOFFS]
                errors = [result["standard_errors"][f"{metric}@{cutoff}"] for cutoff in CUTOFFS]
                axis.bar(positions + shift, values, width=0.36, yerr=errors, capsize=3, label=label, color=color)
            axis.set_xticks(positions, [str(cutoff) for cutoff in CUTOFFS])
            axis.set_xlabel("K")
            axis.set_ylabel("Macro mean")
            axis.set_title(f"Selected checkpoint: {metric.upper()}@K")
            axis.legend(fontsize=8)
        _save_figure(figure, figures / "final_metrics.png")

    environment = config["environment"]
    lines = ["# Mult-VAE MovieLens 20M Training Report", "",
             f"Status: **{'Completed' if complete else 'Training in progress; test has not been evaluated'}**.", "",
             "## Run summary", "",
             f"The selected checkpoint is epoch **{selected_epoch}**, chosen by the highest full-validation "
             f"Recall@20 (**{metrics['full_validation']['recall@20']:.6f}**). "
             f"Training completed **{len(history)} of at most {config['max_epochs']} epochs**; "
             f"termination reason: `{metrics['stop_reason']}`. Selected beta: **{metrics['selected_beta']:.6f}**.", "",
             "This run uses the paper's published Mult-VAE PR architecture and the author's notebook baseline, "
             "adapted to the existing repository splits, seed, metric definitions, checkpoint selection and early stopping. "
             "It is not a reproduction of the paper's original strong-generalization experiment.", "",
             "## Configuration and execution environment", "",
             "| Setting | Value |", "| --- | --- |",
             f"| Encoder | {config['n_items']} -> {config['hidden_dim']} -> mean/log variance, {config['latent_dim']} each |",
             f"| Decoder | {config['latent_dim']} -> {config['hidden_dim']} -> {config['n_items']} logits |",
             f"| Parameters | {config['parameter_count']:,} |",
             "| Nonlinearity; input preprocessing | tanh; L2 normalization before dropout |",
             "| Initialization | Xavier uniform weights; truncated normal biases, std 0.001, bounds +/-0.002 |",
             f"| Dropout | {config['dropout']} (training only) |",
             f"| Adam | lr {config['learning_rate']}; betas 0.9/0.999; eps 1e-8; weight decay 0 |",
             f"| Train / evaluation batch size | {config['batch_size']} / {config['eval_batch_size']} users |",
             f"| KL schedule | min({config['anneal_cap']}, optimizer updates / {config['anneal_steps']}) |",
             f"| Early stopping | {config['selection_metric']}; patience {config['patience']}; start {config['early_stop_start']} |",
             f"| Seed; device; dtype | {config['seed']}; {config['device_used']}; float32 |",
             f"| Hardware | {environment.get('chip', environment['architecture'])}; RAM {environment.get('ram_bytes', 0) / 2**30:.0f} GiB |",
             f"| Python / PyTorch / NumPy | {environment['python']} / {environment['torch']} / {environment['numpy']} |",
             f"| OS | {environment['os']} |",
             f"| Started | {config['started_at']} |", "",
             "## Data and evaluation protocol", "",
             f"The prepared dataset has **{config['n_users']:,} users** and **{config['n_items']:,} items**. "
             f"Train / validation / test interactions: **{config['split_counts']['train']:,} / "
             f"{config['split_counts']['valid']:,} / {config['split_counts']['test']:,}**. "
             "Schema, canonical SHA-256 hashes, ID mappings, duplicates and split disjointness were checked before training. "
             "The original Parquet files and manifest were not regenerated.", "",
             "Only train histories enter the training objective. Validation uses train histories as encoder input and "
             "masks train items. Test uses " + ("train plus validation histories" if config["test_input"] == "train-valid" else "train histories")
             + " as encoder input, keeps weights fixed and masks train plus validation items. "
             "The entire catalog is ranked; there is no negative sampling or sampled-candidate evaluation. "
             "Dropout and latent sampling are disabled for evaluation, which uses the posterior mean. "
             "Test metrics are computed only after checkpoint selection.", "",
             "Recall@K is hits in the first K ranks divided by the user's total target interaction count, "
             "following the existing repository. NDCG@K uses binary relevance, discount 1/log2(rank+1), "
             "and ideal DCG for min(K, target count) relevant items. Users with no targets are omitted. "
             "Metrics are macro means over all eligible users. Standard errors are the population standard "
             "deviation of per-user values divided by sqrt(user count); they are descriptive and do not measure seed variability.", "",
             "## Training trajectory", "",
             "The objective is user-averaged multinomial reconstruction NLL plus beta-weighted KL. "
             "NLL and KL are summed over their respective item/latent dimensions before averaging over users. "
             "Epoch means weight the final partial batch by its user count. A changing beta means total loss "
             "alone should not be interpreted as a fixed-objective convergence curve.", "",
             "| Quantity | Selected epoch | Last epoch |", "| --- | ---: | ---: |"]
    for key, label in (("train_loss", "Total training objective"), ("train_nll", "Reconstruction NLL"),
                       ("train_kl", "Unweighted KL"), ("train_weighted_kl", "Weighted KL"),
                       ("beta_last", "End-of-epoch beta")):
        lines.append(f"| {label} | {selected_record[key]:.6f} | {last[key]:.6f} |")
    lines.extend(["", f"Last-epoch validation Recall@20: **{last['validation']['recall@20']:.6f}**; "
                  f"selected-epoch value: **{selected_record['validation']['recall@20']:.6f}**. "
                  "The saved model maximizes validation Recall@20 across every completed epoch, including epochs before the "
                  "annealing cap. For cap-gated stopping, the patience counter starts at zero at the first epoch that uses "
                  "the cap; subsequent checks compare with the global best Recall@20. Equal scores do not count as improvements.", "",
                  "![Training objective, KL and beta](figures/training_objective.png)", "",
                  "![All validation ranking metrics by epoch](figures/ranking_metrics.png)", "",
                  "## Selected checkpoint metrics", "",
                  f"### Full validation ({metrics['full_validation']['users']:,} eligible users)", "",
                  _metric_table(metrics["full_validation"]), ""])
    if "test" in metrics:
        lines.extend([f"### Final test ({metrics['test']['users']:,} eligible users)", "",
                      _metric_table(metrics["test"]), "",
                      "![Validation and test metrics at all cutoffs](figures/final_metrics.png)", ""])
    else:
        lines.extend(["Test results are unavailable for this report; they were skipped or training is still running.", ""])
    lightgcn_path = ROOT / "artifacts/lightgcn_ml20m/metrics.json"
    lightgcn_config_path = lightgcn_path.with_name("config.json")
    matching_lightgcn_data = (lightgcn_config_path.exists() and
                             json.loads(lightgcn_config_path.read_text()).get("data_canonical_sha256")
                             == config["data_canonical_sha256"])
    if lightgcn_path.exists() and matching_lightgcn_data:
        baseline = json.loads(lightgcn_path.read_text())
        lines.extend(["## Existing LightGCN result for context", "",
                      "| Split | Model | Recall@20 | NDCG@20 |", "| --- | --- | ---: | ---: |"])
        for split, label in (("full_validation", "Validation"), ("test", "Test")):
            if split in metrics and split in baseline:
                for name, result in (("Mult-VAE", metrics[split]), ("LightGCN (previous run)", baseline[split])):
                    lines.append(f"| {label} | {name} | {result['recall@20']:.6f} | {result['ndcg@20']:.6f} |")
        lines.extend(["", "These values use the same splits and Recall/NDCG formulas. "
                      "The existing LightGCN test embeddings use the train-only graph; "
                      "this Mult-VAE run uses " + ("additional validation history at test inference" if config["test_input"] == "train-valid" else "train-only test input")
                      + ". The models also use different training and selection procedures. "
                      "This table is contextual, not a controlled comparison with identical information and tuning budgets. "
                      "The existing LightGCN artifacts contain only K=20 metrics; other cutoffs are not inferred or fabricated.", ""])
    train_seconds = sum(record["train_seconds"] for record in history)
    validation_seconds = sum(record["validation_seconds"] for record in history)
    lines.extend(["## Runtime and memory", "",
                  f"Training compute: **{train_seconds / 60:.2f} min**; per-epoch full validation: "
                  f"**{validation_seconds / 60:.2f} min**. "
                  f"Mean train / validation time per epoch: **{train_seconds / len(history):.2f} / "
                  f"{validation_seconds / len(history):.2f} seconds**.", "",
                  "![Training and validation duration by epoch](figures/epoch_timing.png)", ""])
    if complete:
        peaks = metrics["observed_memory_peaks"]
        lines.extend([f"Measured total run time before report rendering: **{metrics['total_seconds'] / 60:.2f} min**, "
                      f"including preflight, training, checkpoint I/O and final evaluation. Final evaluation took "
                      f"**{metrics['final_evaluation_seconds']:.2f} seconds**. Completed: {metrics['completed_at']}.", "",
                      f"Observed maximum MPS tensor allocation: **{peaks['mps_allocated_mib']:.1f} MiB**; "
                      f"observed maximum MPS driver allocation: **{peaks['mps_driver_mib']:.1f} MiB**; "
                      f"process peak RSS: **{peaks['process_peak_rss_mib']:.1f} MiB**. "
                      "MPS readings are sampled at batch milestones and phase boundaries, not continuous memory profiling.", ""])
    lines.extend(["## Reproducibility and artifacts", "",
                  "- `config.json`: exact configuration, environment and data fingerprints.",
                  "- `data_manifest.json` and `data_audit.json`: input provenance and verified counts/hashes.",
                  "- `best.pt`: selected weights, architecture, ID mappings, epoch and selection metrics.",
                  "- `last.pt`: resumable model/optimizer/RNG state, global step, patience state and embedded best checkpoint.",
                  "- `history.json` / `history.csv`: every completed epoch, all eight validation metrics and their standard errors.",
                  "- `metrics.json`: selected-checkpoint full validation, final test and runtime summary.",
                  "- `run.log`: progress events; `figures/`: exported PNG figures.", "",
                  "### Canonical split SHA-256", "",
                  "| Split | SHA-256 |", "| --- | --- |"])
    for name, digest in config["data_canonical_sha256"].items():
        lines.append(f"| {name} | `{digest}` |")
    lines.extend(["", "### Sources", "",
                  f"- [Liang et al., Variational Autoencoders for Collaborative Filtering (WWW 2018)]({config['paper']}).",
                  f"- [Author's MovieLens 20M notebook]({config['reference_code']}).",
                  "- Repository documentation: `docs/data_preparation.md`, `docs/lightgcn.md`, `docs/multvae.md`.", "",
                  "One random seed was run. Fixed seeds do not guarantee bit-identical MPS results across PyTorch or macOS versions. "
                  "No unit-test suite was added, as requested; validation uses syntax checks, small CPU/MPS smoke runs, "
                  "manifest verification and checkpoint/evaluation checks.", ""])
    temporary = directory / "report.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(directory / "report.md")
    return directory / "report.md"


def main() -> None:
    """Regenerate an English report from an existing run without retraining."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(generate_report(args.run_dir.resolve()))


if __name__ == "__main__":
    main()
