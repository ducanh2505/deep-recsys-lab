"""Continue train+validation fine-tuning with saved Adam and train-loss stopping."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from lightgcn.data import Dataset, load_dataset, normalized_adjacency, sample_negatives
from lightgcn.model import LightGCN
from lightgcn.retrain import canonical_hash, file_hash
from lightgcn.train import ROOT, TrainLossStopping, evaluate, save_checkpoint, train_epoch, validate_checkpoint


def restore_sampling_rng(checkpoint: dict, data: Dataset) -> tuple[np.random.Generator, str]:
    """Restore sampling state, replaying only random draws for older checkpoints."""
    config = checkpoint["config"]
    rng = np.random.default_rng(config["seed"])
    if "numpy_rng_state" in checkpoint:
        rng.bit_generator.state = checkpoint["numpy_rng_state"]
        return rng, "saved_state"
    if config.get("resume_checkpoint"):
        raise ValueError("Continuation checkpoint is missing its saved sampling RNG state")
    for _ in range(checkpoint["epoch"]):
        order = rng.permutation(len(data.train.users))
        for begin in range(0, len(order), config["batch_size"]):
            users = data.train.users[order[begin : begin + config["batch_size"]]]
            sample_negatives(users, data.train_bits, data.n_items, rng)
    return rng, "replayed_sampling_only"


def restore_stopping(checkpoint: dict, source: Path, patience: int, min_delta: float) -> TrainLossStopping:
    """Carry stopping counters at the selected source epoch into continuation."""
    state = checkpoint["training"].get("early_stopping_state")
    if state is not None:
        if state["patience"] != patience or state["min_delta"] != min_delta:
            raise ValueError("Continuation must retain the source early-stopping settings")
        return TrainLossStopping(**state)
    stopping = TrainLossStopping(patience, min_delta)
    history_path = source.parent / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        for record in history:
            if record["epoch"] <= checkpoint["epoch"] and record["phase"] == "main":
                stopping.update(record["train_loss"])
        if stopping.best_loss != checkpoint["training"]["train_loss"]:
            raise ValueError("Source history does not match the selected checkpoint loss")
    else:
        if checkpoint["training"].get("stale_epochs", 0) != 0:
            raise ValueError("Source stopping history is required to recover stale counters")
        stopping.update(checkpoint["training"]["train_loss"])
    return stopping


def write_report(output: Path, metrics: dict, source_metrics: dict | None, baseline: dict | None) -> None:
    """Report observed test metrics and deltas without selecting on test."""
    config = json.loads((output / "config.json").read_text())
    rows = []
    if baseline is not None:
        rows.append(("Baseline train-only", baseline["test"]))
    if source_metrics is not None:
        rows.append(("Fine-tune trước khi tiếp tục", source_metrics["test"]))
    rows.append(("Fine-tune sau khi tiếp tục", metrics["test"]))
    lines = ["# Tiếp tục fine-tune LightGCN", "",
             f"Thực hiện lúc {datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat()}.", "",
             f"Khôi phục trọng số và Adam state từ epoch {metrics['source_epoch']}; "
             f"giữ K={config['layers']}, embedding={config['embedding_dim']}, α={config['layer_weights']}, "
             f"learning rate={config['learning_rate']}, L2={config['l2']}, "
             f"batch size={config['batch_size']:,} và seed={config['seed']}. Không warmup lại.", "",
             f"Đã chạy thêm {metrics['additional_epochs_completed']} epoch (giới hạn 10); "
             f"lý do dừng `{metrics['stop_reason']}`. Checkpoint được giữ ở epoch tổng "
             f"{metrics['selected_epoch']} / epoch chính {metrics['selected_main_epoch']}, "
             f"loss {metrics['selected_train_loss']:.6f}.", "",
             "Early stopping theo train loss, patience=3 và min-delta=0.0001, "
             "tiếp tục counters tại checkpoint nguồn. Train loss đo hội tụ objective, "
             "không trực tiếp chọn chất lượng ranking.", "",
             f"Test được đánh giá một lần sau chọn checkpoint, trên {metrics['test']['users']:,} users; "
             f"catalog {config['n_items']:,} items, mask train + validation. Test không tham gia early stopping.", "",
             "| Hướng | Recall@20 | NDCG@20 |", "| --- | ---: | ---: |"]
    for label, result in rows:
        lines.append(f"| {label} | {result['recall@20']:.6f} | {result['ndcg@20']:.6f} |")
    comparisons = {}
    for name, reference in (("previous_finetune", source_metrics), ("baseline", baseline)):
        if reference is None:
            continue
        comparisons[name] = {}
        for metric in ("recall@20", "ndcg@20"):
            delta = metrics["test"][metric] - reference["test"][metric]
            comparisons[name][metric] = {"absolute": delta,
                                        "relative_percent": 100 * delta / reference["test"][metric]}
        lines += ["", f"So với {'fine-tune trước đó' if name == 'previous_finetune' else 'baseline train-only'}:", ""]
        for metric, delta in comparisons[name].items():
            lines.append(f"- {metric}: {delta['absolute']:+.6f} ({delta['relative_percent']:+.2f}%).")
    lines += ["", f"Thời gian học thêm: {metrics['training_seconds']:.2f}s; "
              f"test: {metrics['test_seconds']:.2f}s; tổng run: {metrics['total_seconds']:.2f}s.", "",
              "Sampling RNG được khôi phục bằng phát lại random draws cho checkpoint cũ, không huấn luyện lại; "
              "checkpoint mới lưu RNG state để tiếp tục trực tiếp. Adam step và moments được giữ nguyên. "
              "Graph và positive samples chỉ dùng train + validation. Checkpoint và báo cáo nguồn được giữ nguyên.", "",
              "Artifacts: `best.pt`, `config.json`, `history.json`, `train.log`, `metrics.json`, "
              "`data_audit.json` và `comparison.json`. `epoch`/`main_epoch` là số tích lũy; "
              "`continued_epoch` là số epoch chạy thêm trong lần này.", ""]
    (output / "report.md").write_text("\n".join(lines))
    (output / "comparison.json").write_text(json.dumps({"metrics": metrics, "previous_finetune": source_metrics,
                                                       "baseline": baseline, "deltas": comparisons}, indent=2) + "\n")


def main() -> None:
    """Resume selected weights and Adam for a bounded number of additional epochs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/lightgcn_retrain/finetune/best.pt")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/ml20m_lightgcn")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/lightgcn_retrain/finetune_continue_10")
    parser.add_argument("--max-epochs", type=int, default=10, help="Maximum additional epochs")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if args.max_epochs < 1 or args.patience < 1 or not math.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error("max-epochs and patience must be positive; min-delta must be nonnegative and finite")
    for name in ("checkpoint", "data_dir", "output_dir"):
        setattr(args, name, (ROOT / getattr(args, name)).resolve())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty to preserve prior runs")
    run_started = time.monotonic()
    source_digest = file_hash(args.checkpoint)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    source_config = source["config"]
    if source_config.get("train_split") != "train-valid" or source_config.get("early_stopping") != "train-loss":
        raise ValueError("Expected a train+validation checkpoint selected by training loss")
    if not source.get("optimizer_state_dict", {}).get("state"):
        raise ValueError("Cannot continue without saved Adam state")
    if args.patience != source_config["patience"] or args.min_delta != source_config["min_delta"]:
        raise ValueError("Continuation must use the source patience and min-delta")
    data = load_dataset(args.data_dir, "train-valid")
    audit = {"source_checkpoint_sha256": source_digest, "n_train": len(data.train.users),
             "n_users": data.n_users, "n_items": data.n_items, "n_test": len(data.test.users),
             "split_pairs_disjoint": True, "source_epoch": source["epoch"]}
    manifest_path = args.data_dir / "manifest.json"
    manifest_digest = file_hash(manifest_path) if manifest_path.exists() else None
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text()).get("canonical_sha256")
        if source_config.get("data_canonical_sha256") != expected:
            raise ValueError("Source checkpoint split hashes do not match the data manifest")
        if expected:
            actual = {name: canonical_hash(args.data_dir / f"{name}.parquet") for name in ("train", "valid", "test")}
            if actual != expected:
                raise ValueError("Actual split hashes do not match the data manifest")
            audit["verified_canonical_sha256"] = actual
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(source_config["seed"])
    model = LightGCN(data.n_users, data.n_items, source_config["embedding_dim"], source_config["layers"],
                     tuple(source_config["layer_weights"])).to(device)
    validate_checkpoint(source, data, model)
    model.load_state_dict(source["state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=source_config["learning_rate"])
    optimizer.load_state_dict(source["optimizer_state_dict"])
    source_step = source["training"]["optimizer_steps"]
    if any(int(state["step"].item()) != source_step for state in optimizer.state.values()):
        raise ValueError("Saved Adam steps do not match checkpoint training metadata")
    print(json.dumps({"event": "restore_sampling_start", "source_epoch": source["epoch"],
                      "adam_steps": source_step}), flush=True)
    rng, rng_method = restore_sampling_rng(source, data)
    stopping = restore_stopping(source, args.checkpoint, args.patience, args.min_delta)
    adjacency = normalized_adjacency(data, device)
    config = {**source_config, "output_dir": str(args.output_dir), "data_dir": str(args.data_dir),
              "resume_checkpoint": str(args.checkpoint), "source_epoch": source["epoch"],
              "source_main_epoch": source["training"]["main_epoch"], "max_epochs": args.max_epochs,
              "warmup_epochs": 0, "warmup_steps": 0,
              "cumulative_warmup_epochs": source_config.get("cumulative_warmup_epochs", source_config["warmup_epochs"]),
              "optimizer_initialization": "restored_adam", "initialization": "resume_checkpoint",
              "sampling_rng_restore": rng_method, "source_checkpoint_sha256": source_digest,
              "device": args.device, "device_used": device_name}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    checkpoint_path = args.output_dir / "best.pt"
    parent_training = {**source["training"], "continued_epoch": 0, "early_stopping_state": asdict(stopping)}
    save_checkpoint(checkpoint_path, model, data, config, source["epoch"], optimizer=optimizer,
                    training=parent_training, rng=rng)
    source_epoch = source["epoch"]
    source_main_epoch = source["training"]["main_epoch"]
    del source
    history = []
    batches = math.ceil(len(data.train.users) / config["batch_size"])
    stop_reason = "max_epochs"
    with (args.output_dir / "train.log").open("w") as log:
        def emit(record: dict) -> None:
            line = json.dumps(record)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        emit({"event": "start", "config": config, "early_stopping_state": asdict(stopping)})
        for added_epoch in range(1, args.max_epochs + 1):
            started = time.monotonic()
            record = {"epoch": source_epoch + added_epoch, "main_epoch": source_main_epoch + added_epoch,
                      "continued_epoch": added_epoch, "phase": "main",
                      **train_epoch(model, adjacency, data, optimizer, rng, config["batch_size"],
                                    config["l2"], config["learning_rate"],
                                    step_offset=source_step + (added_epoch - 1) * batches),
                      "seconds": time.monotonic() - started}
            save, stop = stopping.update(record["train_loss"])
            record.update({"stale_epochs": stopping.stale_epochs, "checkpoint_saved": save,
                           "early_stopping_state": asdict(stopping)})
            if save:
                save_checkpoint(checkpoint_path, model, data, config, record["epoch"],
                                optimizer=optimizer, training=record.copy(), rng=rng)
            history.append(record)
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            emit(record)
            if stop:
                stop_reason = "early_stopping"
                break
        selected = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(selected["state_dict"])
        evaluation_started = time.monotonic()
        test = evaluate(model, adjacency, data, data.test, data.test_offsets, True)
        test_seconds = time.monotonic() - evaluation_started
        metrics = {"source_epoch": source_epoch, "additional_epochs_completed": len(history),
                   "selected_epoch": selected["epoch"], "selected_main_epoch": selected["training"]["main_epoch"],
                   "selected_additional_epoch": selected["training"]["continued_epoch"],
                   "selected_train_loss": selected["training"]["train_loss"], "stop_reason": stop_reason,
                   "train_split": "train-valid", "selection_criterion": "train-loss", "warmup_epochs_completed": 0,
                   "total_finetune_main_epochs_completed": source_main_epoch + len(history),
                   "total_epochs_completed": source_epoch + len(history),
                   "training_seconds": sum(row["seconds"] for row in history), "test_seconds": test_seconds,
                   "total_seconds": time.monotonic() - run_started, "test": test}
        if file_hash(args.checkpoint) != source_digest or (manifest_digest is not None and file_hash(manifest_path) != manifest_digest):
            raise ValueError("Source checkpoint or data manifest changed during continuation")
        audit["source_checkpoint_and_manifest_unchanged"] = True
        (args.output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        source_metrics_path = args.checkpoint.parent / "metrics.json"
        baseline_path = ROOT / "artifacts/lightgcn_ml20m/metrics.json"
        source_metrics = json.loads(source_metrics_path.read_text()) if source_metrics_path.exists() else None
        baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
        write_report(args.output_dir, metrics, source_metrics, baseline)
        emit({"event": "complete", "metrics": metrics})


if __name__ == "__main__":
    main()
