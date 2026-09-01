from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from deep_recsys_lab.config import ModelConfig
from deep_recsys_lab.lightgcn.checkpoint import load_checkpoint, save_checkpoint
from deep_recsys_lab.lightgcn.config import (
    LightGCNConfig,
    LightGCNDataConfig,
    LightGCNEvaluationConfig,
    LightGCNModelConfig,
    LightGCNReportConfig,
    LightGCNSweepConfig,
    LightGCNTrainingConfig,
)
from deep_recsys_lab.lightgcn.data import (
    load_lightgcn_data,
    prepare_lightgcn_from_frame,
)
from deep_recsys_lab.lightgcn.evaluation import BPRSampler, ranking_metrics
from deep_recsys_lab.lightgcn.experiment import Candidate, reproduce_lightgcn
from deep_recsys_lab.lightgcn.model import LightGCN, build_normalized_adjacency
from deep_recsys_lab.lightgcn.report import export_verified_report


def _matrix(rows: list[int], cols: list[int], shape: tuple[int, int]) -> sp.csr_matrix:
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape)


def _ratings() -> pd.DataFrame:
    rows = [
        {"userId": user, "movieId": movie, "rating": 5.0}
        for user in range(1, 13)
        for movie in range(1, 13)
        if (user + movie) % 3 != 0
    ]
    rows.append({"userId": 1, "movieId": 1, "rating": 4.5})
    rows.append({"userId": 99, "movieId": 1, "rating": 3.5})
    return pd.DataFrame(rows)


def test_split_is_deterministic_disjoint_and_filters_duplicates(tmp_path: Path) -> None:
    config = LightGCNDataConfig(name="synthetic", min_positive_ratings=5)
    left = prepare_lightgcn_from_frame(_ratings(), tmp_path / "left", config=config, seed=98_765)
    right = prepare_lightgcn_from_frame(_ratings(), tmp_path / "right", config=config, seed=98_765)
    assert left.dataset_hash == right.dataset_hash
    assert (left.train != right.train).nnz == 0
    assert (left.validation != right.validation).nnz == 0
    assert (left.test != right.test).nnz == 0
    assert left.train.multiply(left.validation).nnz == 0
    assert left.train.multiply(left.test).nnz == 0
    assert left.manifest["counts"]["positive_edges"] == len(_ratings()) - 2


def test_cold_start_heldout_edges_are_removed(tmp_path: Path) -> None:
    seed = 7
    ids = np.array([10, 20, 30, 40, 50])
    heldout_position = np.random.default_rng(seed + 1_000_003).permutation(5)[0]
    rare_item = int(ids[heldout_position])
    rows = [{"userId": 1, "movieId": int(movie), "rating": 5.0} for movie in ids]
    common = [int(movie) for movie in ids if movie != rare_item] + [60]
    rows.extend(
        {"userId": user, "movieId": movie, "rating": 5.0}
        for user in range(2, 20)
        for movie in common
    )
    data = prepare_lightgcn_from_frame(
        pd.DataFrame(rows),
        tmp_path / "cold",
        config=LightGCNDataConfig(name="synthetic", min_positive_ratings=5),
        seed=seed,
    )
    assert rare_item not in data.item_ids
    assert data.manifest["counts"]["test_cold_start_edges_removed"] >= 1
    assert data.test.shape[1] == len(data.item_ids)


def test_artifact_checksum_tampering_is_rejected(tmp_path: Path) -> None:
    data = prepare_lightgcn_from_frame(
        _ratings(),
        tmp_path / "prepared",
        config=LightGCNDataConfig(name="synthetic", min_positive_ratings=5),
        seed=4,
    )
    manifest = json.loads((data.root / "manifest.json").read_text(encoding="utf-8"))
    manifest["counts"]["users"] += 1
    (data.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        load_lightgcn_data(data.root)


def test_split_file_checksum_tampering_is_rejected(tmp_path: Path) -> None:
    data = prepare_lightgcn_from_frame(
        _ratings(),
        tmp_path / "prepared",
        config=LightGCNDataConfig(name="synthetic", min_positive_ratings=5),
        seed=5,
    )
    path = data.root / "train.npz"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_lightgcn_data(data.root)


def test_normalized_adjacency_is_symmetric_and_has_no_self_loops() -> None:
    interactions = _matrix([0, 0, 1], [0, 1, 0], (2, 2))
    adjacency = build_normalized_adjacency(interactions).to_dense()
    assert torch.allclose(adjacency, adjacency.T)
    assert torch.count_nonzero(torch.diag(adjacency)) == 0
    assert adjacency[0, 2] == pytest.approx(0.5)
    assert adjacency[0, 3] == pytest.approx(1 / np.sqrt(2))


def test_lightgcn_uniformly_averages_ego_and_propagated_layers() -> None:
    adjacency = build_normalized_adjacency(_matrix([0], [0], (1, 1)))
    model = LightGCN(1, 1, embedding_dim=1, layers=2, adjacency=adjacency)
    with torch.no_grad():
        model.user_embedding.weight.fill_(2.0)
        model.item_embedding.weight.fill_(4.0)
    users, items = model.propagated_embeddings()
    assert users.item() == pytest.approx(8 / 3)
    assert items.item() == pytest.approx(10 / 3)


def test_bpr_loss_includes_l2_only_on_ego_embeddings() -> None:
    adjacency = build_normalized_adjacency(_matrix([0, 1], [0, 1], (2, 3)))
    model = LightGCN(2, 3, embedding_dim=2, layers=1, adjacency=adjacency)
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 1])
    negatives = torch.tensor([2, 2])
    total, ranking, ego_l2 = model.bpr_loss(users, positives, negatives, l2=0.01)
    expected_l2 = (
        0.5
        * (
            model.user_embedding(users).square().sum()
            + model.item_embedding(positives).square().sum()
            + model.item_embedding(negatives).square().sum()
        )
        / 2
    )
    assert float(ego_l2.detach()) == pytest.approx(float(expected_l2.detach()))
    assert float(total.detach()) == pytest.approx(float((ranking + 0.01 * expected_l2).detach()))


def test_negative_sampler_is_deterministic_unobserved_and_resumable() -> None:
    observed = _matrix([0, 0, 1, 2], [0, 1, 2, 3], (3, 5))
    sampler = BPRSampler(observed, seed=12)
    first = sampler.sample(64)
    duplicate = BPRSampler(observed, seed=12).sample(64)
    assert all(np.array_equal(left, right) for left, right in zip(first, duplicate, strict=True))
    users, _positives, negatives = first
    assert observed[users, negatives].A1.sum() == 0
    state = sampler.state_dict()
    expected = sampler.sample(16)
    resumed = BPRSampler(observed, seed=999)
    resumed.load_state_dict(state)
    actual = resumed.sample(16)
    assert all(np.array_equal(left, right) for left, right in zip(expected, actual, strict=True))


def test_ranking_metric_goldens() -> None:
    truth = _matrix([0, 0, 1], [1, 3, 2], (2, 4))
    metrics = ranking_metrics(np.array([[1, 0], [0, 2]]), truth)
    expected_ndcg = ((1 / (1 + 1 / np.log2(3))) + (1 / np.log2(3))) / 2
    assert metrics["recall@2"] == pytest.approx(0.75)
    assert metrics["ndcg@2"] == pytest.approx(expected_ndcg)
    assert metrics["evaluated_users"] == 2


def test_checkpoint_resume_rejects_dataset_or_config_drift(tmp_path: Path) -> None:
    observed = _matrix([0, 1], [0, 1], (2, 3))
    candidate = Candidate(1, 1e-4)
    adjacency = build_normalized_adjacency(observed)
    model = LightGCN(2, 3, 4, candidate.layers, adjacency)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sampler = BPRSampler(observed, seed=3)
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        sampler,
        step=7,
        dataset_hash="dataset-a",
        config_hash="config-a",
        config={"layers": 1},
    )
    assert (
        load_checkpoint(
            checkpoint,
            model,
            optimizer,
            sampler,
            dataset_hash="dataset-a",
            config_hash="config-a",
        )["step"]
        == 7
    )
    with pytest.raises(ValueError, match="dataset hash"):
        load_checkpoint(
            checkpoint,
            model,
            optimizer,
            sampler,
            dataset_hash="dataset-b",
            config_hash="config-a",
        )
    with pytest.raises(ValueError, match="config hash"):
        load_checkpoint(
            checkpoint,
            model,
            optimizer,
            sampler,
            dataset_hash="dataset-a",
            config_hash="config-b",
        )


@pytest.mark.integration
def test_tiny_prepare_sweep_retrain_test_and_report(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    report_path = tmp_path / "public" / "lightgcn.json"
    data_config = LightGCNDataConfig(name="synthetic", min_positive_ratings=5)
    data = prepare_lightgcn_from_frame(_ratings(), data_dir, config=data_config, seed=98_765)
    config = LightGCNConfig(
        data_dir=str(data_dir),
        output_dir=str(run_dir),
        data=data_config,
        model=LightGCNModelConfig(embedding_dim=8, layers=1),
        training=LightGCNTrainingConfig(
            learning_rate=0.01,
            batch_size=32,
            threads=1,
            max_steps=4,
            minimum_steps=3,
            validation_every=1,
            patience=2,
            wall_time_hours=0.1,
            checkpoint_every=1,
        ),
        sweep=LightGCNSweepConfig(
            layers=(1, 2),
            l2_values=(1e-4,),
            round1_steps=1,
            round2_steps=2,
            finalists=1,
        ),
        evaluation=LightGCNEvaluationConfig(k=2, user_batch_size=4),
        report=LightGCNReportConfig(output=str(report_path)),
    )
    status = reproduce_lightgcn(data, config, run_dir)
    assert status["status"] == "verified"
    assert status["test_accessed"] is True
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    test_access = json.loads((run_dir / "test_access.json").read_text(encoding="utf-8"))
    assert test_access["after_model_selection"] is True
    assert datetime.fromisoformat(test_access["accessed_at"]) >= datetime.fromisoformat(
        selection["selected_at"]
    )
    second_status = reproduce_lightgcn(data, config, run_dir)
    assert second_status["completed_at"] == status["completed_at"]
    report = export_verified_report(run_dir, data, report_path)
    assert report["status"] == "verified"
    assert np.isfinite(report["test_metrics"]["recall@2"])
    assert report_path.exists()


def test_lightgcn_config_does_not_mutate_multivae_model_schema() -> None:
    config = LightGCNConfig()
    changed = replace(config, model=LightGCNModelConfig(embedding_dim=16, layers=4))
    assert changed.model.layers == 4
    assert "layers" not in ModelConfig.__dataclass_fields__
