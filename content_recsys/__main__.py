"""Command-line entry points for the reproducible content/Mult-VAE benchmark."""

import argparse
import json
from pathlib import Path

import torch

from content_recsys.common import BASELINE, CATALOG, DATA, OUTPUT, device_for


def main() -> None:
    """Expose preparation, single-run training, complete search, evaluation and reporting."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "search", "evaluate", "report", "smoke"):
        sub = commands.add_parser(name)
        sub.add_argument("--output-dir", type=Path, default=OUTPUT)
        if name != "report":
            sub.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
            sub.add_argument("--threads", type=int, default=8)
        if name in ("prepare", "search"):
            sub.add_argument("--data-dir", type=Path, default=DATA)
            sub.add_argument("--catalog", type=Path, default=CATALOG)
            sub.add_argument("--baseline", type=Path, default=BASELINE)
        if name in ("train", "evaluate", "smoke"):
            sub.add_argument("--prepared", type=Path, default=OUTPUT / "prepared")
        if name in ("train", "evaluate", "search"):
            sub.add_argument("--eval-batch-size", type=int, default=256)
        if name == "prepare":
            sub.add_argument("--encode-batch-size", type=int, default=8)
        if name == "search":
            sub.add_argument("--stage", choices=("projection", "all"), default="projection",
                             help="Run projection only by default; defer LoRA and global test evaluation.")
        if name == "smoke":
            sub.add_argument("--mode", choices=("projection", "lora", "all"), default="projection")
        if name == "train":
            sub.add_argument("--layout", choices=("single", "multi"), required=True)
            sub.add_argument("--mode", choices=("projection", "lora"), required=True)
            sub.add_argument("--loss", choices=("bpr", "infonce"), required=True)
            sub.add_argument("--temperature", type=float, default=0.1)
            sub.add_argument("--learning-rate", type=float)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--max-epochs", type=int)
            sub.add_argument("--patience", type=int)
            sub.add_argument("--checkpoint-batches", type=int, default=200)
        if name == "evaluate":
            sub.add_argument("--run-dir", type=Path, required=True)
            sub.add_argument("--split", choices=("valid", "test"), default="valid")
            sub.add_argument("--selection-lock", type=Path)
    args = parser.parse_args()
    if args.command != "report":
        if args.threads < 1:
            parser.error("threads must be positive")
        torch.set_num_threads(args.threads)
    root = args.output_dir.resolve()
    if args.command == "report":
        from content_recsys.report import report
        print(report(root))
        return
    device = device_for(args.device)
    if args.command == "prepare":
        from content_recsys.data import prepare
        from content_recsys.encoder import ensure_frozen
        value = prepare(args.data_dir.resolve(), args.catalog.resolve(), args.baseline.resolve(), root / "prepared")
        ensure_frozen(root / "prepared", device, args.encode_batch_size)
    elif args.command == "search":
        from content_recsys.search import run_benchmark
        value = run_benchmark(root, args.data_dir.resolve(), args.catalog.resolve(), args.baseline.resolve(),
                              args.device, args.eval_batch_size, args.stage)
    elif args.command == "train":
        from content_recsys.train import run
        value = run({"prepared": str(args.prepared.resolve()), "output": str(root), "layout": args.layout,
                     "mode": args.mode, "loss": args.loss, "temperature": args.temperature,
                     "learning_rate": args.learning_rate if args.learning_rate is not None else (1e-3 if args.mode == "projection" else 1e-4),
                     "seed": args.seed, "device": args.device, "eval_batch_size": args.eval_batch_size,
                     "batch_size": 256 if args.mode == "projection" else 8,
                     "negative_count": 128 if args.mode == "projection" else 16,
                     "max_epochs": args.max_epochs if args.max_epochs is not None else (10 if args.mode == "projection" else 3),
                     "patience": args.patience if args.patience is not None else (3 if args.mode == "projection" else 2),
                     "checkpoint_batches": args.checkpoint_batches})
    elif args.command == "evaluate":
        from content_recsys.evaluate import evaluate_run
        value = evaluate_run(args.prepared.resolve(), args.run_dir.resolve(), device, args.split,
                             args.eval_batch_size, args.selection_lock)
    else:
        from content_recsys.smoke import smoke
        modes = ("projection", "lora") if args.mode == "all" else (args.mode,)
        value = smoke(args.prepared.resolve(), root / "smoke", device, modes=modes)
    print(json.dumps(value, allow_nan=False, default=str), flush=True)


if __name__ == "__main__":
    main()
