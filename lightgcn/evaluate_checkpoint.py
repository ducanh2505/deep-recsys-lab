"""Evaluate a selected LightGCN checkpoint on full validation and test splits."""

import argparse
import json
from pathlib import Path

import torch

from lightgcn.data import load_dataset, normalized_adjacency
from lightgcn.model import LightGCN
from lightgcn.train import ROOT, evaluate, validate_checkpoint


def main() -> None:
    """Load one selected checkpoint and write its final full-catalog metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/ml20m_lightgcn")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    for name in ("checkpoint", "data_dir", "output"):
        setattr(args, name, (ROOT / getattr(args, name)).resolve())
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    data = load_dataset(args.data_dir, config.get("train_split", "train"))
    model = LightGCN(
        data.n_users,
        data.n_items,
        config["embedding_dim"],
        config["layers"],
        layer_weights=tuple(config["layer_weights"]) if "layer_weights" in config else None,
    ).to(device)
    validate_checkpoint(checkpoint, data, model)
    model.load_state_dict(checkpoint["state_dict"])
    adjacency = normalized_adjacency(data, device)
    result = {
        "selected_epoch": checkpoint["epoch"],
    }
    if data.train_split == "train":
        result["selection_validation"] = checkpoint.get("validation")
        result["full_validation"] = evaluate(model, adjacency, data, data.valid, data.valid_offsets, False)
    else:
        result.update({"train_split": data.train_split,
                       "selection_criterion": config["selection_criterion"],
                       "selected_main_epoch": checkpoint["training"]["main_epoch"],
                       "selected_train_loss": checkpoint["training"]["train_loss"]})
    result["test"] = evaluate(model, adjacency, data, data.test, data.test_offsets, True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
