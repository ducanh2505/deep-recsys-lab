"""Sequential, validation-selected objective ablations on the audited splits."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from multvae.data import load_dataset
from multvae.evaluate import evaluate
from multvae.report import generate_report
from multvae.train import ROOT, model_from_config, run_training, verify_checkpoint, write_json
from training_logging.wandb import TrainingTracker, add_arguments, options_from_args


def render_summary(output: Path, records: list[dict], status: str) -> dict:
    """Export measured rankings, matched ablations and an English plot/report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = json.loads((output / "plan.json").read_text())
    ranked = sorted(records, key=lambda row: (-row["recall@20"], -row["ndcg@20"]))
    hybrids = [row for row in ranked if row["nll_weight"] > 0 and row["bpr_weight"] > 0 and row["anneal_cap"] > 0]
    summary = {"status": status, "selection_metric": "full-validation recall@20", "runs": ranked,
               "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "best": ranked[0] if ranked else None, "best_three_objectives": hybrids[0] if hybrids else None}
    write_json(output / "summary.json", summary)
    if not ranked:
        return summary
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
    lines = ["# Mult-VAE NLL/BPR/KL objective grid", "", f"Status: **{status}**; completed runs: {len(records)}.", "",
             "Full train history enters both encoder input and multinomial reconstruction target. "
             "BPR uses raw decoder logits for one preselected train positive per user and 16 uniform "
             "train-unseen negatives with replacement per epoch. Training dropout is retained. "
             f"Negatives never consult validation or test labels. All runs start from scratch at seed {plan['seed']}.", "",
             "Objective: `nll_weight * NLL + bpr_weight * mean(softplus(s_neg - s_pos)) + beta(step) * KL`. "
             "NLL sums over observed items, KL over latent dimensions, BPR averages over pairs; all average over users. "
             "A fixed BPR coefficient therefore does not imply equal gradient influence or equal loss magnitude.", "",
             f"KL is annealed with `beta = min(cap, step / {plan['anneal_steps']})`. "
             f"Cap 0.2 is reached after {math.ceil(0.2 * plan['anneal_steps'])} completed updates. "
             f"Cap zero is the no-KL ablation. Every run has at most {plan['max_epochs']} epochs; "
             f"early stopping starts immediately with patience {plan['patience']}, using full-validation Recall@20. "
             "Test is evaluated only for the global selected run.", "",
             "## Validation ranking", "",
             "| Run | NLL weight | BPR weight | KL cap | Epochs | Best epoch | Recall@20 | NDCG@20 | W&B |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in ranked:
        lines.append(f"| {row['run']} | {row['nll_weight']:g} | {row['bpr_weight']:g} | {row['anneal_cap']:g} | "
                     f"{row['epochs_completed']} | {row['selected_epoch']} | {row['recall@20']:.6f} | "
                     f"{row['ndcg@20']:.6f} | [Run]({row['wandb_url']}) |")
    lines.extend(["", "## Matched ablation effects", "",
                  "Differences below use selected validation checkpoints within the same training budget. "
                  "Positive deltas favor the added objective. One seed and validation selection do not establish statistical significance.", "",
                  "| Added objective | Fixed settings | Delta Recall@20 | Delta NDCG@20 |",
                  "| --- | --- | ---: | ---: |"])
    index = {(r["nll_weight"], r["bpr_weight"], r["anneal_cap"]): r for r in records}
    for row in records:
        nll, bpr, cap = row["nll_weight"], row["bpr_weight"], row["anneal_cap"]
        for label, control_key, fixed in (
            ("BPR", (nll, 0.0, cap), f"NLL={nll:g}, KL={cap:g}, BPR={bpr:g}"),
            ("KL", (nll, bpr, 0.0), f"NLL={nll:g}, BPR={bpr:g}"),
            ("NLL", (0.0, bpr, cap), f"BPR={bpr:g}, KL={cap:g}"),
        ):
            enabled = {"BPR": bpr > 0, "KL": cap > 0, "NLL": nll > 0}[label]
            control = index.get(control_key)
            if enabled and control:
                lines.append(f"| {label} | {fixed} | {row['recall@20'] - control['recall@20']:+.6f} | "
                             f"{row['ndcg@20'] - control['ndcg@20']:+.6f} |")
    lines.extend(["", "## Selected models", "", f"Global best: **{ranked[0]['run']}**."])
    if hybrids:
        lines.append(f"Best with all three objectives: **{hybrids[0]['run']}**, "
                     f"Recall@20={hybrids[0]['recall@20']:.6f}, NDCG@20={hybrids[0]['ndcg@20']:.6f}.")
    final_path = output / "selected_metrics.json"
    if final_path.exists():
        final = json.loads(final_path.read_text())
        lines.extend(["", f"Selected-run final test: Recall@20 **{final['test']['recall@20']:.6f}**, "
                      f"NDCG@20 **{final['test']['ndcg@20']:.6f}**. Test input uses train+validation; "
                      "both histories are masked, and weights stay fixed."])
    lines.extend(["", "## Interpretation and limits", "",
                  "Decoder logits support pairwise ranking directly; their difference also equals the difference "
                  "of log-softmax scores because the per-user normalizer cancels. Adding BPR creates an auxiliary "
                  "ranking objective, rather than the original Mult-VAE likelihood objective. Full-history reconstruction "
                  "does not leak held-out labels, but training positives also occur in the input; unseen-item quality "
                  "must be established through held-out ranking. NLL already makes catalog scores compete, so BPR may "
                  "be redundant or may change optimization. Unobserved items can contain undiscovered positives.", "",
                  "This is a short-budget, single-seed study. KL and BPR effects can depend on longer training, "
                  "weight scale and negative count. The historical 177-epoch baseline uses a slower KL schedule "
                  "and a different budget; it is not a controlled comparator for these short-budget runs.", "",
                  "![Validation Recall@20 across objective weights](objective_grid.png)", "",
                  "Artifacts: `plan.json`, `data_audit.json`, `summary.json`, `results.csv`, "
                  "`selected_metrics.json`, and each run's checkpoints, history, figures and W&B metadata.", "",
                  "Sources: [BPR](https://arxiv.org/abs/1205.2618), "
                  "[Mult-VAE](https://arxiv.org/abs/1802.05814).", ""])
    (output / "report.md").write_text("\n".join(lines))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, cap in zip(axes, (0.0, 0.2)):
        matrix = np.full((2, 3), np.nan)
        for row in records:
            if row["anneal_cap"] == cap:
                matrix[int(row["nll_weight"]), (0.0, 1.0, 10.0).index(row["bpr_weight"])] = row["recall@20"]
        axis.imshow(matrix, cmap="viridis", vmin=0, vmax=max(r["recall@20"] for r in records))
        axis.set(xticks=range(3), xticklabels=["0", "1", "10"], yticks=range(2), yticklabels=["0", "1"],
                 xlabel="BPR weight", ylabel="NLL weight", title=f"KL cap {cap:g}: validation Recall@20")
        for (y, x), value in np.ndenumerate(matrix):
            color = "black" if not np.isfinite(value) or value > 0.6 * max(r["recall@20"] for r in records) else "white"
            axis.text(x, y, f"{value:.4f}" if np.isfinite(value) else "—", ha="center", va="center", color=color)
    figure.savefig(output / "objective_grid.png", dpi=160)
    plt.close(figure)
    return summary


def main() -> None:
    """Run the ten objective ablations sequentially on MPS with online logging."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/ml20m_lightgcn")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/multvae_objectives_mps_seed42")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--anneal-epochs", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    add_arguments(parser)
    args = parser.parse_args()
    options = options_from_args(args, parser)
    if options.mode != "online":
        parser.error("This search requires W&B online logging")
    if min(args.max_epochs, args.patience, args.anneal_epochs) <= 0 or not math.isfinite(args.anneal_epochs):
        parser.error("Epoch/patience/annealing budgets must be positive and finite")
    if not torch.backends.mps.is_available():
        parser.error("MPS is required")
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.data_dir)
    write_json(output / "data_audit.json", data.audit)
    anneal_steps = math.ceil(math.ceil(data.n_users / 500) * args.anneal_epochs / 0.2)
    trials = [{"nll_weight": nll, "bpr_weight": bpr, "anneal_cap": cap}
              for nll, bpr, cap in itertools.product((1.0, 0.0), (0.0, 1.0, 10.0), (0.2, 0.0)) if nll or bpr]
    source_paths = ["multvae/model.py", "multvae/data.py", "multvae/train.py", "multvae/evaluate.py",
                    "multvae/search.py", "multvae/report.py", "training_logging/wandb.py"]
    plan = {"trials": trials, "seed": args.seed, "max_epochs": args.max_epochs, "patience": args.patience,
            "anneal_steps": anneal_steps, "data_canonical_sha256": data.audit["canonical_sha256"],
            "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths},
            "wandb_entity": options.entity, "wandb_project": options.project}
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing search plan differs; use a new output directory")
    write_json(plan_path, plan)
    records = []
    started = time.monotonic()
    for index, trial in enumerate(trials, 1):
        name = f"nll{trial['nll_weight']:g}_bpr{trial['bpr_weight']:g}_kl{trial['anneal_cap']:g}"
        directory = output / "runs" / name
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            last = directory / "last.pt"
            if directory.exists() and any(directory.iterdir()) and not last.exists():
                raise ValueError(f"Incomplete run without checkpoint: {directory}; inspect before retrying")
            config = argparse.Namespace(
                data_dir=args.data_dir, output_dir=directory, device="mps", hidden_dim=600, latent_dim=200,
                dropout=0.5, learning_rate=0.001, batch_size=500, eval_batch_size=500,
                anneal_steps=anneal_steps, negative_count=16, max_epochs=args.max_epochs, patience=args.patience,
                early_stop_start="immediate", seed=args.seed, test_input="train-valid", skip_test=True,
                require_wandb_online=True, resume=last if last.exists() else None, **trial)
            print(json.dumps({"event": "grid_trial", "trial": index, "total": len(trials), "run": name}), flush=True)
            with TrainingTracker(directory, "multvae_objectives", options) as tracker:
                run_training(config, tracker, data)
                if not tracker.active:
                    raise RuntimeError(f"W&B logging became unavailable during {name}; inspect online state")
            torch.mps.empty_cache()
        metrics = json.loads(metrics_path.read_text())
        tracking = json.loads((directory / "wandb_state.json").read_text())
        selected = next(row for row in json.loads((directory / "history.json").read_text())
                        if row["epoch"] == metrics["selected_epoch"])
        row = {"run": name, **trial, "epochs_completed": metrics["epochs_completed"],
               "selected_epoch": metrics["selected_epoch"], "recall@20": metrics["full_validation"]["recall@20"],
               "ndcg@20": metrics["full_validation"]["ndcg@20"], "selected_beta": metrics["selected_beta"],
               "nll": selected["train_nll"], "bpr": selected["train_bpr"], "kl": selected["train_kl"],
               "weighted_nll": selected["train_weighted_nll"], "weighted_bpr": selected["train_weighted_bpr"],
               "weighted_kl": selected["train_weighted_kl"], "total_seconds": metrics["total_seconds"],
               "wandb_url": f"https://wandb.ai/{tracking['entity']}/{tracking['project']}/runs/{tracking['run_id']}"}
        records.append(row)
        summary = render_summary(output, records, "running")
        print(json.dumps({"event": "grid_progress", "completed": len(records), "total": len(trials),
                          "best": summary["best"], "elapsed_seconds": time.monotonic() - started}), flush=True)
    summary = render_summary(output, records, "selection_locked")
    write_json(output / "selection_lock.json", {"best": summary["best"], "plan": plan})
    winner = output / "runs" / summary["best"]["run"]
    final_path = output / "selected_metrics.json"
    if not final_path.exists():
        checkpoint = torch.load(winner / "best.pt", map_location="cpu", weights_only=True)
        verify_checkpoint(checkpoint, data)
        model = model_from_config(checkpoint["config"], torch.device("mps"))
        model.load_state_dict(checkpoint["state_dict"])
        final = {"run": summary["best"]["run"], "selected_epoch": checkpoint["epoch"],
                 "full_validation": checkpoint["validation"],
                 "test": evaluate(model, data, "test", 500, "train-valid")}
        write_json(final_path, final)
    else:
        final = json.loads(final_path.read_text())
        if final["run"] != summary["best"]["run"]:
            raise ValueError("Saved final test belongs to a different winner")
    metrics = json.loads((winner / "metrics.json").read_text())
    metrics["test"] = final["test"]
    write_json(winner / "metrics.json", metrics)
    tracking = json.loads((winner / "wandb_state.json").read_text())
    with TrainingTracker(winner, "multvae_objectives", options) as tracker:
        tracker.start(json.loads((winner / "config.json").read_text()), tracking, resumed=True)
        if not tracker.active:
            raise RuntimeError("Could not reconnect selected W&B run to publish final test metrics")
        tracker.complete(metrics)
        if not tracker.active:
            raise RuntimeError("W&B final metric publication failed")
    generate_report(winner)
    summary = render_summary(output, records, "completed")
    print(json.dumps({"event": "grid_complete", "best": summary["best"], "test": final["test"],
                      "report": str(output / "report.md")}), flush=True)


if __name__ == "__main__":
    main()
