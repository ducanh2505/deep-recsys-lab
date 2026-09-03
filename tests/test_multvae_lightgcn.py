from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from deep_recsys_lab.evaluation.evaluator import evaluate_model
from deep_recsys_lab.evaluation.metrics import ranking_batch_stats
from deep_recsys_lab.lightgcn.config import LightGCNDataConfig
from deep_recsys_lab.lightgcn.data import prepare_lightgcn_from_frame, stable_hash
from deep_recsys_lab.model.multvae import MultiVAE
from deep_recsys_lab.multvae_lightgcn import (
    MultiVAELightGCNConfig,
    MultiVAESweepConfig,
    _rank_results,
    build_multvae_views,
    export_multvae_report,
    reproduce_multvae_lightgcn,
)


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"userId": user, "movieId": movie, "rating": 5.0}
            for user in range(1, 9)
            for movie in range(1, 13)
            if (user + movie) % 4 != 0
        ]
    )


def test_multvae_views_follow_lightgcn_split_without_leakage(tmp_path: Path) -> None:
    data = prepare_lightgcn_from_frame(
        _ratings(),
        tmp_path / "lightgcn-data",
        config=LightGCNDataConfig(name="synthetic", min_positive_ratings=5),
        seed=98_765,
    )
    views = build_multvae_views(data)
    assert views.train.shape == views.validation.shape == views.test.shape
    assert views.combined.shape == views.train.shape
    assert views.train.multiply(views.validation).nnz == 0
    assert views.train.multiply(views.test).nnz == 0
    assert views.validation.multiply(views.test).nnz == 0
    assert np.all(views.train.data == 1.0)
    assert np.all(views.combined.data == 1.0)
    assert views.combined.nnz == views.train.nnz + views.validation.nnz


def test_lightgcn_protocol_uses_truth_count_and_skips_empty_users() -> None:
    scores = torch.tensor(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )
    truth = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    lightgcn = ranking_batch_stats(scores, truth, 2, protocol="lightgcn")
    legacy = ranking_batch_stats(scores, truth, 2, protocol="multvae")
    assert lightgcn["recall"] == pytest.approx((2 / 3 + 1.0) / 2)
    assert legacy["recall"] == pytest.approx(1.0)
    assert lightgcn["evaluated_users"] == 2
    assert lightgcn["truth_edges"] == 5


def test_evaluator_reports_lightgcn_counts() -> None:
    model = MultiVAE(n_items=5, hidden_dims=(5, 2, 5), latent_dim=2, dropout=0.0)
    batches = [
        {
            "data": torch.tensor([[1.0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
            "ground_truth": torch.tensor([[0.0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]),
        }
    ]
    metrics = evaluate_model(model, batches, ks=(2,), ranking_protocol="lightgcn")
    assert metrics["evaluated_users"] == 1.0
    assert metrics["truth_edges"] == 1.0
    assert np.isfinite(metrics["recall@2"])
    assert np.isfinite(metrics["ndcg@2"])


def test_sweep_tie_break_is_deterministic() -> None:
    records = [
        {
            "trial": {"id": "z-trial"},
            "validation_metrics": {"recall@20": 0.5, "ndcg@20": 0.4},
        },
        {
            "trial": {"id": "a-trial"},
            "validation_metrics": {"recall@20": 0.5, "ndcg@20": 0.4},
        },
    ]
    assert [item["trial"]["id"] for item in _rank_results(records, 20)] == [
        "a-trial",
        "z-trial",
    ]


@pytest.mark.integration
def test_multvae_lightgcn_reproduce_and_report_is_resumable(tmp_path: Path) -> None:
    data_dir = tmp_path / "lightgcn-data"
    run_dir = tmp_path / "run"
    prepare_lightgcn_from_frame(
        _ratings(),
        data_dir,
        config=LightGCNDataConfig(name="synthetic", min_positive_ratings=5),
        seed=98_765,
    )
    config = MultiVAELightGCNConfig(
        seed=7,
        device="cpu",
        data_dir=str(data_dir),
        run_dir=str(run_dir),
        baseline_checkpoint=None,
        baseline_data_dir=None,
        lightgcn_metrics_path=None,
        batch_size=4,
        latent_dim=2,
        total_anneal_steps=10,
        round1_epochs=1,
        round2_epochs=2,
        max_epochs=3,
        round2_finalists=1,
        final_candidates=1,
        validation_every=1,
        patience=1,
        checkpoint_every=1,
        selection_k=2,
        evaluation_ks=(2, 3),
        sweep=MultiVAESweepConfig(
            architectures=((4, 2, 4), (3, 2, 3)),
            dropout=(0.0,),
            beta_cap=(0.2,),
            learning_rate=(0.01,),
            weight_decay=(0.0,),
        ),
    )
    result = reproduce_multvae_lightgcn(config)
    assert result["status"] == "verified"
    assert result["test_accessed"] is True
    selection = json.loads((run_dir / "selection.json").read_text())
    access = json.loads((run_dir / "test_access.json").read_text())
    assert access["after_model_selection"] is True
    assert access["selection_hash"] == stable_hash(selection)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["stage"] == "final"
    assert checkpoint["dataset_hash"] == stable_hash(
        {"base_dataset_hash": result["dataset_hash"], "phase": "train-plus-validation"}
    )
    second = reproduce_multvae_lightgcn(config)
    assert second["completed_at"] == result["completed_at"]
    with pytest.raises(ValueError, match="config"):
        reproduce_multvae_lightgcn(replace(config, batch_size=2))
    other_data_dir = tmp_path / "other-lightgcn-data"
    other_ratings = _ratings().assign(movieId=lambda frame: frame["movieId"] + 100)
    prepare_lightgcn_from_frame(
        other_ratings,
        other_data_dir,
        config=LightGCNDataConfig(name="synthetic-other", min_positive_ratings=5),
        seed=98_765,
    )
    with pytest.raises(ValueError, match="dataset"):
        reproduce_multvae_lightgcn(replace(config, data_dir=str(other_data_dir)))
    report = export_multvae_report(run_dir)
    assert report["status"] == "verified"
    assert report["report_hash"]
    assert report["training_curve"]
    assert report["final_training_curve"]
    assert {"loss", "nll", "kl", "beta"}.issubset(
        report["training_curve"][-1]["train"]
    )
    assert (run_dir / "report.json").exists()
    assert (run_dir / "report.md").exists()
