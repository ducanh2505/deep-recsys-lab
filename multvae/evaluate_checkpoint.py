"""Re-evaluate a saved Mult-VAE checkpoint using existing split conventions."""

import argparse
from pathlib import Path

import torch

from multvae.data import load_dataset
from multvae.evaluate import evaluate
from multvae.train import ROOT, model_from_config, verify_checkpoint, write_json


def main() -> None:
    """Verify dataset fingerprints and evaluate full validation and optional test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/ml20m_lightgcn")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--eval-batch-size", type=int, default=500)
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    data = load_dataset(args.data_dir.resolve())
    verify_checkpoint(checkpoint, data)
    config = checkpoint["config"]
    model = model_from_config(config, torch.device(args.device))
    model.load_state_dict(checkpoint["state_dict"])
    results = {"selected_epoch": checkpoint["epoch"], "test_input": config["test_input"],
               "full_validation": evaluate(model, data, "valid", args.eval_batch_size, config["test_input"])}
    if not args.skip_test:
        results["test"] = evaluate(model, data, "test", args.eval_batch_size, config["test_input"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output.resolve(), results)
    print(results)


if __name__ == "__main__":
    main()
