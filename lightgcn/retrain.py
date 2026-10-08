"""Audit data, run both selected LightGCN retraining protocols, and report test metrics."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import torch

from lightgcn.data import load_dataset
from lightgcn.model import LightGCN
from lightgcn.train import ROOT, validate_checkpoint


def canonical_hash(path: Path) -> str:
    """Hash the original four-column rows using the preparation manifest format."""
    digest = hashlib.sha256()
    frame = pl.read_parquet(path, columns=["userId", "movieId", "rating", "timestamp"])
    for user, item, rating, timestamp in frame.sort(["userId", "movieId"]).iter_rows():
        digest.update(f"{user},{item},{rating:.1f},{timestamp}\n".encode())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit_data(data_dir: Path, source: Path) -> dict:
    """Verify counts, mappings, disjointness, and actual canonical split hashes."""
    manifest = json.loads((data_dir / "manifest.json").read_text())
    data = load_dataset(data_dir, "train-valid")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    model = LightGCN(data.n_users, data.n_items, 32, 2, (0.975, 0.0125, 0.0125))
    validate_checkpoint(checkpoint, data, model)
    if checkpoint["epoch"] != 6 or checkpoint["config"].get("train_split", "train") != "train":
        raise ValueError("Expected the selected epoch-6 train-only checkpoint")
    for field, value in (("users", data.n_users), ("items", data.n_items)):
        if value != manifest["ten_core"][field]:
            raise ValueError(f"{field} count does not match manifest")
    counts = {"train": len(data.train.users) - len(data.valid.users),
              "valid": len(data.valid.users), "test": len(data.test.users)}
    audit = {"source_split_counts": counts, "combined_training_interactions": len(data.train.users),
             "users": data.n_users, "items": data.n_items,
             "eligible_test_users": int(np.count_nonzero(np.diff(data.test_offsets))),
             "split_pairs_disjoint": True, "unique_interactions_within_splits": True,
             "id_mapping_matches_checkpoint": True, "source_checkpoint_epoch": checkpoint["epoch"],
             "source_checkpoint_sha256": file_hash(source)}
    del data, checkpoint, model
    hashes = {name: canonical_hash(data_dir / f"{name}.parquet") for name in counts}
    if hashes != manifest["canonical_sha256"]:
        raise ValueError("Actual split canonical hashes do not match manifest")
    audit["verified_canonical_sha256"] = hashes
    return audit


def run_training(command: list[str], output_dir: Path) -> None:
    """Stream a training subprocess while keeping its complete log."""
    temporary_log = output_dir.parent / f"{output_dir.name}.log"
    with temporary_log.open("w") as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
    if output_dir.exists():
        temporary_log.replace(output_dir / "train.log")
    if return_code:
        raise RuntimeError(f"Training failed with exit status {return_code}; inspect {output_dir}")


def write_report(output_dir: Path, baseline: dict, commands: dict, audit: dict) -> None:
    """Write a comparison using measured metrics, without selecting models on test."""
    runs = {name: json.loads((output_dir / name / "metrics.json").read_text())
            for name in ("scratch", "finetune")}
    comparison = {}
    for name, metrics in runs.items():
        comparison[name] = {}
        for metric in ("recall@20", "ndcg@20"):
            difference = metrics["test"][metric] - baseline["test"][metric]
            comparison[name][metric] = {"absolute": difference,
                                        "relative_percent": 100 * difference / baseline["test"][metric]}
    summary = {"created_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
               "baseline": baseline, "runs": runs, "deltas_from_baseline": comparison,
               "commands": commands, "data_audit": audit,
               "versions": {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__, "polars": pl.__version__}}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Kết quả huấn luyện lại LightGCN trên train + validation", "",
             f"Thực hiện lúc {summary['created_at']} trên {json.loads((output_dir / 'scratch/config.json').read_text())['device_used']}.", "",
             "Giữ K=2, embedding=32, α=(0.975, 0.0125, 0.0125), Adam learning rate=0.001, "
             "L2=0.001, batch size=524.288 và seed=42.", "",
             f"Dữ liệu học gồm {audit['combined_training_interactions']:,} tương tác, "
             f"{audit['users']:,} users và {audit['items']:,} items. Counts, ID mapping, "
             "disjointness và SHA-256 canonical thực tế đã được kiểm tra với manifest.", "",
             f"Mỗi hướng đánh giá test một lần trên toàn bộ {audit['eligible_test_users']:,} users, "
             f"ranking trên catalog {audit['items']:,} items và loại item đã thấy trong train + validation.", "",
             "| Hướng | Warmup đã chạy | Epoch chính đã chạy | Epoch chính được giữ | Recall@20 | NDCG@20 | Thời gian học (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
             f"| Baseline train-only | 0 | — | {baseline['selected_epoch']} | "
             f"{baseline['test']['recall@20']:.6f} | {baseline['test']['ndcg@20']:.6f} | — |"]
    labels = {"scratch": "Từ đầu", "finetune": "Fine-tune"}
    for name, metrics in runs.items():
        lines.append(f"| {labels[name]} | {metrics['warmup_epochs_completed']} | {metrics['main_epochs_completed']} | "
                     f"{metrics['selected_main_epoch']} | {metrics['test']['recall@20']:.6f} | "
                     f"{metrics['test']['ndcg@20']:.6f} | {metrics['training_seconds']:.2f} |")
    lines += ["", "| Hướng so với baseline | Δ Recall@20 | Δ Recall tương đối | Δ NDCG@20 | Δ NDCG tương đối |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, delta in comparison.items():
        lines.append(f"| {labels[name]} | {delta['recall@20']['absolute']:+.6f} | "
                     f"{delta['recall@20']['relative_percent']:+.2f}% | {delta['ndcg@20']['absolute']:+.6f} | "
                     f"{delta['ndcg@20']['relative_percent']:+.2f}% |")
    lines += ["", "## Quy trình và cách diễn giải", "",
              "Hướng từ đầu khởi tạo Xavier mới và chạy đúng 6 epoch; giữ trọng số cuối.", "",
              "Hướng fine-tune nạp trọng số baseline ở epoch 6, khởi tạo Adam mới, rồi cập nhật cả trọng số "
              "và moments trong 3 epoch warmup. Learning rate tăng tuyến tính theo từng step từ 0.001/W "
              "đến 0.001; Adam được giữ nguyên khi chuyển sang tối đa 10 epoch chính. "
              "Không early-stop hoặc chọn checkpoint trong warmup. Giai đoạn chính dùng train loss, "
              "patience=3 và min-delta=0.0001; lưu checkpoint có loss thấp nhất.", ""]
    for name, metrics in runs.items():
        lines.append(f"- {labels[name]}: lý do dừng `{metrics['stop_reason']}`, loss tại checkpoint "
                     f"{metrics['selected_train_loss']:.6f}, thời gian test {metrics['test_seconds']:.2f}s, "
                     f"tổng thời gian run {metrics['total_seconds']:.2f}s.")
    lines += ["", "Early stopping theo train loss đo sự hội tụ của loss, không trực tiếp chọn chất lượng ranking. "
              "Test không được dùng để chọn epoch hay điều chỉnh cấu hình. Các chênh lệch là kết quả quan sát "
              "của một seed; hai hướng có ngân sách học khác nhau và fine-tune kế thừa 6 epoch train-only. "
              "Baseline là số liệu đã lưu từ architecture search, không được chạy lại trong thí nghiệm này.", "",
              "## Tái lập và artifacts", "",
              "Chạy `uv run python -m lightgcn.retrain --device mps` từ repository root, với output directory "
              "mới qua `--output-dir` nếu kết quả đã tồn tại. MPS có thể dao động nhẹ giữa các phiên bản thiết bị/PyTorch.", "",
              "Kiểm chứng: `uv run python -m unittest discover -s tests -v`; "
              "`uv run python -m compileall lightgcn scripts tests main.py`.", "",
              "Mỗi thư mục run có `best.pt` (bao gồm Adam state), `config.json`, `history.json`, "
              "`train.log` và `metrics.json`. `epoch` là epoch tổng tính từ đầu run; "
              "`main_epoch` loại warmup. `summary.json` chứa số liệu đầy đủ, delta, phiên bản và lệnh thực tế; "
              "`data_audit.json` lưu bằng chứng kiểm tra dữ liệu.", ""]
    for name, command in commands.items():
        lines += [f"Lệnh {labels[name]}:", "", "```bash", shlex.join(command), "```", ""]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    """Run the agreed scratch and warmup/fine-tune experiments sequentially."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/ml20m_lightgcn")
    parser.add_argument("--source-checkpoint", type=Path, default=ROOT / "artifacts/lightgcn_ml20m/best.pt")
    parser.add_argument("--baseline-metrics", type=Path, default=ROOT / "artifacts/lightgcn_ml20m/metrics.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/lightgcn_retrain")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    for name in ("data_dir", "source_checkpoint", "baseline_metrics", "output_dir"):
        setattr(args, name, (ROOT / getattr(args, name)).resolve())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty; use a new directory to preserve existing results")
    baseline = json.loads(args.baseline_metrics.read_text())
    print(json.dumps({"event": "audit_start"}), flush=True)
    audit = audit_data(args.data_dir, args.source_checkpoint)
    protected = {str(path): file_hash(path) for path in
                 (args.source_checkpoint, args.baseline_metrics, args.data_dir / "manifest.json")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"event": "audit_complete", "audit": audit}), flush=True)
    common = [sys.executable, "-m", "lightgcn.train", "--data-dir", str(args.data_dir),
              "--train-split", "train-valid", "--layers", "2", "--embedding-dim", "32",
              "--layer-weights", "0.975,0.0125,0.0125", "--learning-rate", "0.001", "--l2", "0.001",
              "--batch-size", "524288", "--seed", "42", "--device", args.device]
    commands = {
        "scratch": common + ["--output-dir", str(args.output_dir / "scratch"), "--max-epochs", "6",
                             "--early-stopping", "none"],
        "finetune": common + ["--output-dir", str(args.output_dir / "finetune"),
                              "--init-checkpoint", str(args.source_checkpoint), "--warmup-epochs", "3",
                              "--max-epochs", "10", "--early-stopping", "train-loss",
                              "--patience", "3", "--min-delta", "0.0001"],
    }
    for name, command in commands.items():
        run_training(command, args.output_dir / name)
    if any(file_hash(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError("Source checkpoint, baseline metrics or data manifest changed during retraining")
    audit["source_artifacts_unchanged_after_runs"] = True
    (args.output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    write_report(args.output_dir, baseline, commands, audit)
    print(json.dumps({"event": "report_complete", "report": str(args.output_dir / "report.md")}), flush=True)


if __name__ == "__main__":
    main()
