"""Bounded synthetic acceptance workflow used by CI and local verification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
from torch.utils.data import DataLoader

from .config import AppConfig, DatasetConfig, EvaluationConfig, ModelConfig, TrainerConfig
from .data.dataset import InteractionDataset
from .data.preprocess import prepare_from_rows
from .device import select_device
from .model.multvae import MultiVAE
from .training.trainer import Trainer


def run_smoke(output_dir: str | Path) -> dict[str, object]:
    root = Path(output_dir)
    data_dir = root / "data"
    train_dir = root / "run"
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("BENTOML_HOME", str(root / "bentoml-home"))
    rows = [
        {"userId": user, "movieId": movie, "rating": 5.0}
        for user in range(1, 9)
        for movie in range(1, 9)
        if (user + movie) % 3 != 0
    ]
    prepared = prepare_from_rows(
        pd.DataFrame(rows),
        data_dir,
        config=DatasetConfig(
            name="synthetic",
            min_positive_ratings=3,
            n_validation_users=2,
            n_test_users=2,
            strict_user_counts=True,
        ),
    )
    config = AppConfig(
        seed=98765,
        device="cpu",
        output_dir=str(train_dir),
        data_dir=str(data_dir),
        model_version="smoke-v1",
        dataset=DatasetConfig(
            name="synthetic", n_validation_users=2, n_test_users=2, min_positive_ratings=3
        ),
        model=ModelConfig(hidden_dims=(16, 8, 16), latent_dim=8, dropout=0.0),
        trainer=TrainerConfig(
            epochs=2,
            batch_size=4,
            total_anneal_steps=10,
            max_train_batches=2,
            max_eval_batches=2,
        ),
        evaluation=EvaluationConfig(ks=(2, 4, 8)),
    )
    model = MultiVAE(
        n_items=prepared.n_items,
        hidden_dims=config.model.hidden_dims,
        latent_dim=config.model.latent_dim,
        dropout=config.model.dropout,
    )
    trainer = Trainer(
        model,
        DataLoader(InteractionDataset(prepared.train), batch_size=4, shuffle=True),
        DataLoader(
            InteractionDataset(
                prepared.validation_fold_in,
                eval=True,
                fold_in=prepared.validation_fold_in,
                fold_out=prepared.validation_fold_out,
            ),
            batch_size=4,
        ),
        output_dir=train_dir,
        device=select_device("cpu"),
        config=config,
        epochs=2,
        beta_cap=config.trainer.beta_cap,
        total_anneal_steps=config.trainer.total_anneal_steps,
        validation_ks=config.evaluation.ks,
        max_train_batches=config.trainer.max_train_batches,
        max_eval_batches=config.trainer.max_eval_batches,
        seed=config.seed,
    )
    history = trainer.run()
    checkpoint = train_dir / "best.pt"
    # Import after BENTOML_HOME is fixed so the smoke model cannot leak into a
    # developer's default model store.
    from starlette.testclient import TestClient

    from .serving.model_store import register_bento_model
    from .serving.service import create_bento_service

    model_version = f"smoke-{uuid4().hex[:12]}"
    registered = register_bento_model(
        checkpoint,
        data_dir,
        model_version=model_version,
        model_config=config.model.__dict__,
        extra_config={"smoke": True},
    )
    service = create_bento_service(str(registered.tag))
    with TestClient(service.to_asgi()) as client:
        assert client.get("/livez").status_code == 200
        assert client.get("/readyz").status_code == 200
        info = client.post("/model_info")
        assert info.status_code == 200
        assert info.json()["inference_backend"] == "onnxruntime"
        assert info.json()["execution_provider"] == "CPUExecutionProvider"
        assert info.json()["onnx_opset"] == 20
        known = prepared.item_ids[:5].tolist()
        response = client.post("/recommend", json={"movie_ids": known, "top_k": 2})
        assert response.status_code == 200, response.text
        assert all(item["movie_id"] not in known for item in response.json()["recommendations"])
    result = {
        "status": "PASS",
        "epochs": len(history),
        "model_tag": str(registered.tag),
    }
    (root / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/deep-recsys-smoke")
    args = parser.parse_args()
    result = run_smoke(args.output_dir)
    print(f"smoke acceptance: {result['status']}")


if __name__ == "__main__":  # pragma: no cover
    main()
