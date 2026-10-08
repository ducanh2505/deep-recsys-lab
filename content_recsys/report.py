"""English experiment reports, paired bootstrap intervals and publication-ready figures."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from content_recsys.common import exclusive_lock, write_json


def bootstrap_interval(delta: np.ndarray, seed: int = 2026) -> tuple[float, float]:
    """Bootstrap paired user deltas 2,000 times with bounded working memory."""
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(125):
        indices = rng.integers(0, len(delta), size=(16, len(delta)), dtype=np.int32)
        means.extend(delta[indices].mean(1).tolist())
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def paired_delta(bpr: list[Path], nce: list[Path], profile: str, scoring: str) -> dict:
    """Average paired differences across matched seeds, then resample users."""
    differences = []
    counts = user_ids = None
    for left, right in zip(sorted(bpr), sorted(nce)):
        a = np.load(left / f"test_{profile}_users.npz")
        b = np.load(right / f"test_{profile}_users.npz")
        if not np.array_equal(a["user_ids"], b["user_ids"]):
            raise ValueError("Paired bootstrap user IDs differ")
        evaluation_a = json.loads((left / "test_evaluation.json").read_text())
        evaluation_b = json.loads((right / "test_evaluation.json").read_text())
        alpha_a = 0.0 if scoring == "content" else evaluation_a["profiles"][profile]["alpha"]
        alpha_b = 0.0 if scoring == "content" else evaluation_b["profiles"][profile]["alpha"]
        ia = int(np.flatnonzero(np.isclose(a["alphas"], alpha_a))[0])
        ib = int(np.flatnonzero(np.isclose(b["alphas"], alpha_b))[0])
        differences.append(a["recall@20"][ia] - b["recall@20"][ib])
        counts, user_ids = a["train_counts"], a["user_ids"]
    delta = np.mean(differences, axis=0)
    interval = bootstrap_interval(delta)
    return {"mean_bpr_minus_infonce": float(delta.mean()), "ci95": interval,
            "users": len(user_ids), "seed_deltas": [float(d.mean()) for d in differences],
            "positive_interval": interval[0] > 0, "negative_interval": interval[1] < 0,
            "counts": counts, "delta": delta}


def report(root: Path) -> Path:
    """Serialize report writers while allowing independent benchmark progress."""
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / ".report.lock", wait=True):
        return _report(root)


def projection_report(root: Path, status: dict) -> Path:
    """Report three-seed projection validation results while GPU LoRA is deferred."""
    selected = [Path(p) for p in status["projection_selected_runs"]]
    rows, pairs = [], []
    lines = ["# Projection stage: BPR vs InfoNCE", "",
             "**Validation results only.** All 24 projection training runs are complete. "
             "The 24 LoRA runs are deferred to a future GPU stage. Learned-model test evaluation and "
             "the final loss recommendation remain pending.", "",
             "## Experimental setup", "",
             "The full catalog contains 11,508 movies. All eligible validation users are evaluated. "
             "The frozen Qwen3-Embedding-0.6B features, original splits, binary training histories and "
             "fixed Mult-VAE checkpoint are unchanged. Four projection branches each search four "
             "learning-rate/temperature combinations at seed 42; their validation winners are replicated "
             "at seeds 43 and 44. The user profile is the normalized full-history weighted mean.", "",
             "Checkpoint and hyperparameter selection use binary content-only validation Recall@20, "
             "then NDCG@20. Fusion alpha is tuned separately for each seed and profile. "
             "Training uses binary weights; rating-minus-three weights are an inference-only control. "
             "Profiles are detached and built from an epoch-start memory bank; only candidate embeddings receive gradients.", "",
             "## Results", "", "Values are validation means ± sample standard deviation across three seeds.", "",
             "| Layout | Loss | Profile | Scoring | Recall@20 | NDCG@20 |",
             "|---|---|---|---|---:|---:|"]
    if status.get("synthetic_smoke"):
        lines[2:2] = ["**Synthetic rendering smoke artifact. These numbers are not benchmark results.**", ""]
    for layout in ("single", "multi"):
        for loss in ("bpr", "infonce"):
            branch = sorted(p for p in selected if p.name.startswith(f"{layout}_projection_{loss}_"))
            if len(branch) != 3:
                raise ValueError("Projection report requires three completed seeds per branch")
            training = [json.loads((p / "training.json").read_text()) for p in branch]
            for profile in ("binary", "rating"):
                evaluations = [json.loads((p / "valid_evaluation.json").read_text())["profiles"][profile] for p in branch]
                for scoring in ("content", "hybrid"):
                    row = {"layout": layout, "loss": loss, "profile": profile, "scoring": scoring,
                           "alphas": [e["alpha"] for e in evaluations],
                           "hours": float(np.mean([t["elapsed_seconds"] / 3600 for t in training])),
                           "peak_rss_mib": max(t["memory"]["process_peak_rss_mib"] for t in training),
                           "observed_mps_driver_mib": max(t["memory"]["mps_driver_mib"] for t in training)}
                    for metric in ("recall@20", "ndcg@20"):
                        values = [e[scoring][metric] for e in evaluations]
                        row[metric] = float(np.mean(values))
                        row[metric + "_std"] = float(np.std(values, ddof=1))
                        row[metric + "_seeds"] = values
                    rows.append(row)
                    lines.append(f"| {layout} | {loss} | {profile} | {scoring} | "
                                 f"{row['recall@20']:.6f} ± {row['recall@20_std']:.6f} | "
                                 f"{row['ndcg@20']:.6f} ± {row['ndcg@20_std']:.6f} |")
    lines.extend(["", "### Frozen controls and Mult-VAE", "",
                  "| Control | Profile | Content Recall@20 | Hybrid Recall@20 | Alpha |",
                  "|---|---|---:|---:|---:|"])
    baseline = None
    for name in status["controls"]:
        control = Path(name)
        for profile, value in json.loads((control / "valid_evaluation.json").read_text())["profiles"].items():
            baseline = value["multvae"]
            lines.append(f"| {control.name} | {profile} | {value['content']['recall@20']:.6f} | "
                         f"{value['hybrid']['recall@20']:.6f} | {value['alpha']:.2f} |")
    lines.extend(["", f"Mult-VAE validation Recall@20: **{baseline['recall@20']:.6f}**; "
                  f"NDCG@20: **{baseline['ndcg@20']:.6f}**; eligible users: **{baseline['users']:,}**.", "",
                  "## Paired loss comparison", "",
                  "Differences below pair seeds 42, 43 and 44 within each layout. They are descriptive "
                  "validation comparisons after hyperparameter and alpha selection. They are not held-out "
                  "test evidence. The planned user-bootstrap confirmation remains deferred until the GPU stage finishes.", "",
                  "| Layout | Binary scoring | BPR − InfoNCE Recall@20 (mean ± SD) | Seed deltas |",
                  "|---|---|---:|---|"])
    for layout in ("single", "multi"):
        for scoring in ("content", "hybrid"):
            bpr = next(r for r in rows if (r["layout"], r["loss"], r["profile"], r["scoring"]) ==
                       (layout, "bpr", "binary", scoring))
            nce = next(r for r in rows if (r["layout"], r["loss"], r["profile"], r["scoring"]) ==
                       (layout, "infonce", "binary", scoring))
            delta = np.array(bpr["recall@20_seeds"]) - np.array(nce["recall@20_seeds"])
            pair = {"layout": layout, "scoring": scoring, "mean": float(delta.mean()),
                    "std": float(delta.std(ddof=1)), "seed_deltas": delta.tolist()}
            pairs.append(pair)
            lines.append(f"| {layout} | {scoring} | {pair['mean']:+.6f} ± {pair['std']:.6f} | "
                         f"{', '.join(f'{x:+.6f}' for x in delta)} |")
    binary = [r for r in rows if r["profile"] == "binary" and r["scoring"] == "hybrid"]
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    prefix = "[Synthetic smoke] " if status.get("synthetic_smoke") else ""
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, scoring, color in ((-0.18, "content", "#2563eb"), (0.18, "hybrid", "#16a085")):
        values = [next(r for r in rows if (r["layout"], r["loss"], r["profile"], r["scoring"]) ==
                       (b["layout"], b["loss"], "binary", scoring)) for b in binary]
        ax.bar(np.arange(4) + offset, [v["recall@20"] for v in values], width=0.36,
               yerr=[v["recall@20_std"] for v in values], capsize=4, label=scoring, color=color)
    ax.axhline(baseline["recall@20"], color="red", linestyle="--", label="Mult-VAE")
    ax.set_xticks(np.arange(4), [f"{r['layout']} / {r['loss']}" for r in binary])
    ax.set_ylabel("Validation Recall@20")
    ax.set_title(prefix + "Projection: binary profiles, three-seed mean ± SD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "projection_validation_recall20.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(np.arange(4), [p["mean"] for p in pairs], yerr=[p["std"] for p in pairs], fmt="o", capsize=4)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xticks(np.arange(4), [f"{p['layout']} / {p['scoring']}" for p in pairs])
    ax.set_ylabel("BPR − InfoNCE validation Recall@20")
    ax.set_title(prefix + "Matched-seed deltas: mean ± SD (not a confidence interval)")
    fig.tight_layout()
    fig.savefig(figures / "projection_validation_loss_comparison.png", dpi=180)
    plt.close(fig)
    lines.extend(["", "![Projection validation scores](figures/projection_validation_recall20.png)", "",
                  "![Projection paired loss comparison](figures/projection_validation_loss_comparison.png)", "",
                  "## Runtime and memory", "",
                  "| Layout / Loss | Mean training hours | Peak process RSS (MiB) | Observed MPS driver (MiB) | Alphas (42 / 43 / 44) |",
                  "|---|---:|---:|---:|---|"])
    for row in binary:
        lines.append(f"| {row['layout']} / {row['loss']} | {row['hours']:.3f} | {row['peak_rss_mib']:.0f} | "
                     f"{row['observed_mps_driver_mib']:.0f} | {row['alphas']} |")
    lines.extend(["", "Measured training time includes catalog refresh and validation. Process peak RSS "
                  "includes earlier work in the same process; accelerator measurements are observations, not guaranteed peaks.", "",
                  "### Selected hyperparameters", "",
                  "| Run | Learning rate | Temperature | Selected epoch |", "|---|---:|---:|---:|"])
    for directory in selected:
        config = json.loads((directory / "config.json").read_text())
        training = json.loads((directory / "training.json").read_text())
        lines.append(f"| {directory.name} | {config['learning_rate']:g} | {config['temperature']:g} | {training['selected_epoch']} |")
    chosen = status["projection_validation_choice"]
    lines.extend(["", "## Provisional recommendation", "",
                  f"Within the projection stage, validation selects **{chosen['layout']} / {chosen['loss']}**, "
                  f"with three-seed mean hybrid binary Recall@20 **{chosen['recall@20']:.6f}**. "
                  "This is a provisional projection choice; the overall loss decision awaits LoRA and locked test confirmation.", ""])
    if any(all(a == 1 for a in row["alphas"]) for row in binary):
        lines.extend(["Branches with alpha=1 in every seed do not improve the selected hybrid over Mult-VAE on validation.", ""])
    lines.extend(["## Deferred GPU stage and limitations", "",
                  "[The GPU specification](deferred_lora_gpu.json) preserves the 16 seed-42 LoRA configurations "
                  "and the rule for eight replications. No GPU jobs have been launched. CUDA execution support, "
                  "host paths, memory checks and resume verification must be completed on the future GPU host. "
                  "Partial Mac LoRA files are retained and are not counted as a completed run.", "",
                  "Projection choices and alphas are frozen in `projection_selection_lock.json`. "
                  "The global test lock will be created after both stages. This stage uses warm-start random "
                  "splits and rating-derived positives; it does not establish temporal, cold-start or click-log performance. "
                  "Three seeds and the detached memory-bank approximation limit the conclusions.", "",
                  "No unit tests were added or executed. Operational verification is recorded in `smoke/smoke.json`.", ""])
    write_json(root / "projection_summary.json", {"split": "valid", "rows": rows, "paired_seed_deltas": pairs,
                                               "baseline": baseline, "projection_validation_choice": chosen,
                                               "test_evaluated": False, "lora_deferred": True,
                                               "synthetic_smoke": status.get("synthetic_smoke", False)})
    path = root / "projection_report.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)
    return path


def _report(root: Path) -> Path:
    """Write an honest partial report or the complete English benchmark report."""
    status_path = root / "benchmark.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"phase": "not_started", "completed": False}
    lines = ["# Content Encoder + Mult-VAE: BPR vs InfoNCE", "",
             f"Status: **{status['phase']}**. Completed training runs: **{status.get('finished_training_runs', 0)}/48**.", "",
             "## Experimental setup", "",
             "The benchmark compares two movie layouts (serialized text and weighted multiview content), two training modes "
             "(projection and LoRA + projection), and two implicit objectives (BPR and InfoNCE). "
             "It uses Qwen3-Embedding-0.6B at revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` and an immutable Mult-VAE checkpoint.", "",
             "The original MovieLens splits are retained. Ratings >= 4 define binary positives; these are not native impression/click logs. "
             "Training uses binary weights. Rating-minus-three weighting is an inference-only profile ablation. "
             "Validation uses train histories; test uses train + validation histories and masks both. Ranking covers all 11,508 movies.", "",
             "Profiles use complete histories with leave-one-positive-out training. Epoch-refreshed detached memory banks provide "
             "candidate-only gradients. Both losses share the sampling schedule and use cosine scores divided by temperature. "
             "BPR averages pairwise softplus terms; InfoNCE uses a joint log-sum-exp denominator.", "",
             "The search contains 32 seed-42 runs and 16 replications at seeds 43 and 44. "
             "Checkpoint/hyperparameter selection uses content-only binary validation Recall@20, then NDCG@20. "
             "Fusion weights are tuned on validation in steps of 0.05. The deployment choice uses mean hybrid binary validation Recall@20 across three seeds. "
             "All choices are locked before test evaluation.", ""]
    if status.get("synthetic_smoke"):
        lines[2:2] = ["**Synthetic reporting smoke artifact. These numbers are not benchmark results.**", ""]
    if status.get("last_error"):
        lines.extend(["## Execution issue", "", f"`{status['last_error']['type']}`: {status['last_error']['message']}", ""])
    if not status.get("completed"):
        lines.extend(["## Results", "", "The full benchmark is not complete. No final loss recommendation is available.", ""])
        if status.get("lora_status") == "deferred_to_gpu":
            lines.extend(["### Staged execution", "",
                          "Projection runs execute on the Mac: 16 seed-42 search runs and eight seed replications. "
                          "All 24 LoRA runs are deferred to a future GPU stage at the user's request. "
                          "Partial Mac LoRA artifacts are retained; no GPU jobs have been launched. "
                          "Learned-model test evaluation remains deferred until all 48 runs and validation choices are locked.", "",
                          "See [the deferred GPU specification](deferred_lora_gpu.json). "
                          "CUDA support and real-device smoke/resume checks remain part of that future stage.", ""])
        if status.get("projection_completed"):
            projection_report(root, status)
            lines.extend(["**The projection stage is complete (24/24 training runs).** "
                          "[Read the English projection validation report](projection_report.md) "
                          "for three-seed tables, figures, paired loss deltas, controls and the provisional choice.", ""])
        progress = []
        for path in sorted((root / "prepared" / "frozen").glob("*_progress.json")):
            value = json.loads(path.read_text())
            progress.append(f"| {path.stem.removesuffix('_progress')} | {value['completed']} | {value['total']} |")
        if progress:
            lines.extend(["### Frozen feature preparation", "", "| View | Encoded movies | Available movies |", "|---|---:|---:|", *progress, ""])
        smoke_path = root / "smoke" / "smoke.json"
        if smoke_path.exists():
            smoke = json.loads(smoke_path.read_text())
            lines.extend(["### Operational verification", "", f"Real-data smoke verification passed: **{smoke['passed']}**. No unit tests were run.", ""])
            if smoke.get("rough_training_step_hours"):
                estimate = smoke["rough_training_step_hours"]
                costs = "; ".join(f"{mode} steps {estimate[f'{mode}_24_runs_max_epochs']:.1f} hours"
                                  for mode in ("projection", "lora") if f"{mode}_24_runs_max_epochs" in estimate)
                lines.extend([f"Single-batch extrapolation at the epoch caps: {costs} across 24 runs each. "
                              "Catalog refresh, evaluation and thermal variability are excluded; this is not an ETA.", ""])
        if status.get("active_run"):
            lines.extend([f"Active run: `{Path(status['active_run']).name}`.", ""])
        completed = []
        for name in status.get("runs", []):
            directory = Path(name)
            value = json.loads((directory / "training.json").read_text())
            completed.append(f"| {directory.name} | {value['selected_epoch']} | {value['validation']['recall@20']:.6f} | {value['elapsed_seconds'] / 3600:.2f} |")
        if completed:
            lines.extend(["| Run | Selected epoch | Content validation Recall@20 | Hours |", "|---|---:|---:|---:|", *completed, ""])
    else:
        selected = [Path(p) for p in status["selected_runs"]]
        summary, paired = [], []
        lines.extend(["## Results", "", "Values below are test means ± sample standard deviation across three training seeds.", "",
                      "| Layout | Mode | Loss | Profile | Scoring | Recall@20 | NDCG@20 |", "|---|---|---|---|---|---:|---:|"])
        for layout in ("single", "multi"):
            for mode in ("projection", "lora"):
                for loss in ("bpr", "infonce"):
                    branch = [p for p in selected if p.name.startswith(f"{layout}_{mode}_{loss}_")]
                    for profile in ("binary", "rating"):
                        evaluations = [json.loads((p / "test_evaluation.json").read_text())["profiles"][profile] for p in branch]
                        training = [json.loads((p / "training.json").read_text()) for p in branch]
                        for scoring in ("content", "hybrid"):
                            row = {"layout": layout, "mode": mode, "loss": loss, "profile": profile, "scoring": scoring}
                            for metric in ("recall@20", "ndcg@20"):
                                values = [e[scoring][metric] for e in evaluations]
                                row[metric] = float(np.mean(values))
                                row[metric + "_std"] = float(np.std(values, ddof=1))
                            row["hours"] = float(np.mean([t["elapsed_seconds"] / 3600 for t in training]))
                            row["peak_rss_mib"] = max(t["memory"]["process_peak_rss_mib"] for t in training)
                            row["observed_mps_driver_mib"] = max(t["memory"]["mps_driver_mib"] for t in training)
                            row["alphas"] = [e["alpha"] for e in evaluations]
                            summary.append(row)
                            lines.append(f"| {layout} | {mode} | {loss} | {profile} | {scoring} | {row['recall@20']:.6f} ± {row['recall@20_std']:.6f} | {row['ndcg@20']:.6f} ± {row['ndcg@20_std']:.6f} |")
        controls = [Path(p) for p in status["controls"]]
        lines.extend(["", "### Frozen controls and Mult-VAE baseline", "", "| Control | Profile | Content Recall@20 | Hybrid Recall@20 | Alpha |", "|---|---|---:|---:|---:|"])
        baseline_metrics = None
        for control in controls:
            for profile, value in json.loads((control / "test_evaluation.json").read_text())["profiles"].items():
                baseline_metrics = value["multvae"]
                lines.append(f"| {control.name} | {profile} | {value['content']['recall@20']:.6f} | {value['hybrid']['recall@20']:.6f} | {value['alpha']:.2f} |")
        lines.extend(["", f"Mult-VAE test Recall@20: **{baseline_metrics['recall@20']:.6f}**; NDCG@20: **{baseline_metrics['ndcg@20']:.6f}**.", "",
                      "## Paired loss comparison", "", "Intervals resample users 2,000 times after averaging matched-seed user differences. "
                      "They describe user-sampling uncertainty, not training-seed uncertainty. Seed differences are reported separately.", "",
                      "| Layout / Mode | Profile / Scoring | BPR − InfoNCE Recall@20 | 95% CI |", "|---|---|---:|---|"])
        data_manifest = json.loads((root / "prepared" / "manifest.json").read_text())
        bundle = torch.load(root / "prepared" / "data.pt", weights_only=True, map_location="cpu")
        thresholds = np.quantile(np.diff(bundle["train"]["offsets"].numpy()), [0.25, 0.75])
        for layout in ("single", "multi"):
            for mode in ("projection", "lora"):
                bpr = [p for p in selected if p.name.startswith(f"{layout}_{mode}_bpr_")]
                nce = [p for p in selected if p.name.startswith(f"{layout}_{mode}_infonce_")]
                for profile in ("binary", "rating"):
                    for scoring in ("content", "hybrid"):
                        value = paired_delta(bpr, nce, profile, scoring)
                        groups = {"short_history": value["counts"] <= thresholds[0], "long_history": value["counts"] >= thresholds[1]}
                        value["history_groups"] = {name: {"users": int(mask.sum()), "delta": float(value["delta"][mask].mean())}
                                                   for name, mask in groups.items()}
                        value.pop("counts")
                        value.pop("delta")
                        value.update({"layout": layout, "mode": mode, "profile": profile, "scoring": scoring})
                        paired.append(value)
                        lines.append(f"| {layout} / {mode} | {profile} / {scoring} | {value['mean_bpr_minus_infonce']:+.6f} | [{value['ci95'][0]:+.6f}, {value['ci95'][1]:+.6f}] |")
        lines.extend(["", "### History-length groups", "",
                      f"Short histories contain at most {thresholds[0]:g} train positives; long histories contain at least {thresholds[1]:g}. "
                      "Thresholds are computed from all train users.", "",
                      "| Layout / Mode | Binary hybrid short-history delta | Long-history delta |", "|---|---:|---:|"])
        for value in paired:
            if value["profile"] == "binary" and value["scoring"] == "hybrid":
                groups = value["history_groups"]
                lines.append(f"| {value['layout']} / {value['mode']} | {groups['short_history']['delta']:+.6f} | {groups['long_history']['delta']:+.6f} |")
        lines.extend(["", "### Selected hyperparameters", "", "| Run | Learning rate | Temperature | Selected epoch |", "|---|---:|---:|---:|"])
        for directory in selected:
            config = json.loads((directory / "config.json").read_text())
            training = json.loads((directory / "training.json").read_text())
            lines.append(f"| {directory.name} | {config['learning_rate']:g} | {config['temperature']:g} | {training['selected_epoch']} |")
        figures = root / "figures"
        figures.mkdir(exist_ok=True)
        binary = [r for r in summary if r["profile"] == "binary" and r["scoring"] == "hybrid"]
        prefix = "[Synthetic smoke] " if status.get("synthetic_smoke") else ""
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(np.arange(len(binary)), [r["recall@20"] for r in binary], yerr=[r["recall@20_std"] for r in binary],
               color=["#2563eb" if r["loss"] == "bpr" else "#16a085" for r in binary], capsize=4)
        ax.axhline(baseline_metrics["recall@20"], color="red", linestyle="--", label="Mult-VAE baseline")
        ax.set_xticks(np.arange(len(binary)), [f"{r['layout']}\n{r['mode']}\n{r['loss']}" for r in binary])
        ax.set_ylabel("Test Recall@20")
        ax.set_title(prefix + "Hybrid binary profiles: three-seed mean ± standard deviation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "hybrid_recall20.png", dpi=180)
        plt.close(fig)
        comparisons = [p for p in paired if p["profile"] == "binary" and p["scoring"] == "hybrid"]
        fig, ax = plt.subplots(figsize=(9, 4))
        means = np.array([p["mean_bpr_minus_infonce"] for p in comparisons])
        intervals = np.array([p["ci95"] for p in comparisons])
        ax.errorbar(np.arange(4), means, yerr=np.maximum(0, np.stack([means - intervals[:, 0], intervals[:, 1] - means])), fmt="o", capsize=5)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_xticks(np.arange(4), [f"{p['layout']} / {p['mode']}" for p in comparisons])
        ax.set_ylabel("BPR − InfoNCE test Recall@20")
        ax.set_title(prefix + "Paired user bootstrap: 95% confidence intervals")
        fig.tight_layout()
        fig.savefig(figures / "loss_comparison.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(np.arange(len(binary)), [r["hours"] for r in binary])
        ax.set_xticks(np.arange(len(binary)), [f"{r['layout']}\n{r['mode']}\n{r['loss']}" for r in binary])
        ax.set_ylabel("Mean training hours per seed")
        ax.set_title(prefix + "Measured training cost, including catalog refresh and validation")
        fig.tight_layout()
        fig.savefig(figures / "runtime.png", dpi=180)
        plt.close(fig)
        lines.extend(["", "![Hybrid Recall@20](figures/hybrid_recall20.png)", "", "![Paired loss comparison](figures/loss_comparison.png)", "",
                      "## Runtime and memory", "", "| Layout / Mode / Loss | Mean training hours | Peak process RSS (MiB) | Observed MPS driver (MiB) | Hybrid alphas |", "|---|---:|---:|---:|---|"])
        for row in binary:
            lines.append(f"| {row['layout']} / {row['mode']} / {row['loss']} | {row['hours']:.2f} | {row['peak_rss_mib']:.0f} | {row['observed_mps_driver_mib']:.0f} | {row['alphas']} |")
        lines.extend(["", "![Runtime](figures/runtime.png)", "", "## Recommendation", ""])
        chosen = status["deployment_choice"]
        lines.append(f"The validation-selected deployment configuration is **{chosen['layout']} / {chosen['mode']} / {chosen['loss']}**. "
                     f"Its three-seed mean hybrid binary validation Recall@20 is {chosen['recall@20']:.6f}. Test results did not change this choice.")
        for p in comparisons:
            conclusion = "BPR has a positive user-bootstrap interval" if p["positive_interval"] else "InfoNCE has a positive user-bootstrap interval" if p["negative_interval"] else "the interval includes zero; no clear loss advantage is established"
            lines.append(f"\nFor {p['layout']} / {p['mode']}, {conclusion}. Matched-seed deltas: {p['seed_deltas']}.")
        if any(all(alpha == 1 for alpha in row["alphas"]) for row in binary):
            lines.append("\nFor branches with alpha=1 in every seed, content did not improve the validation-selected hybrid beyond Mult-VAE.")
        write_json(root / "summary.json", {"rows": summary, "paired_comparisons": paired,
                                          "history_thresholds": thresholds.tolist(), "baseline": baseline_metrics,
                                          "deployment_choice": chosen, "data_audit": data_manifest["data_audit"]})
    lines.extend(["", "## Limitations", "", "These are warm-start random per-user splits, not temporal or cold-start evaluation. "
                  "Unobserved items are sampled optimization negatives, not verified dislikes. "
                  "Memory-bank profiles omit gradients through histories and become stale within an epoch. "
                  "LoRA has a three-epoch cap and projection a ten-epoch cap; conclusions apply to these budgets. "
                  "Three seeds provide limited evidence about optimizer variability. Process peak RSS includes earlier work in the same process.", "",
                  "## References", "", "- [BPR](https://arxiv.org/abs/1205.2618)",
                  "- [InfoNCE](https://arxiv.org/abs/1807.03748)",
                  "- [Content-based recommendation objectives, SIGIR 2023](https://arxiv.org/abs/2304.03112)",
                  "- [Qwen3 Embedding](https://arxiv.org/abs/2506.05176)", ""])
    path = root / "report.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)
    return path
