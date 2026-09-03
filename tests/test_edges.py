from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from deep_recsys_lab.data.dataset import InteractionDataset
from deep_recsys_lab.data.types import PreparedData
from deep_recsys_lab.device import select_device
from deep_recsys_lab.evaluation.evaluator import evaluate_model, recommend
from deep_recsys_lab.model.multvae import MultiVAE
from deep_recsys_lab.reproducibility import capture_rng_state, restore_rng_state, seed_everything
from deep_recsys_lab.training.loss import kl_divergence, multinomial_nll
from deep_recsys_lab.training.schedule import beta_at_step
from deep_recsys_lab.training.trainer import Trainer, train_model


def test_device_rng_and_validation_edges() -> None:
    assert str(select_device("cpu")) == "cpu"
    assert str(select_device("auto")) in {"cpu", "cuda", "mps"}
    with pytest.raises(ValueError):
        select_device("tpu")
    seed_everything(11)
    state = capture_rng_state()
    expected = torch.rand(4)
    restore_rng_state(state)
    assert torch.equal(expected, torch.rand(4))
    with pytest.raises(ValueError):
        beta_at_step(-1, 200_000, 0.2)
    with pytest.raises(ValueError):
        beta_at_step(1, -1, 0.2)
    assert beta_at_step(1, 0, 0.2) == 0.2
    with pytest.raises(ValueError):
        multinomial_nll(torch.zeros(1, 2), torch.zeros(1, 3))
    with pytest.raises(ValueError):
        kl_divergence(torch.zeros(1, 2), torch.zeros(1, 3))


def test_dataset_edge_cases_and_model_config() -> None:
    matrix = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.float32)
    dataset = InteractionDataset(matrix, eval=True, prop=0.5)
    assert len(dataset) == 2
    assert dataset[1]["data"].sum().item() == 0
    with pytest.raises(ValueError):
        InteractionDataset(matrix, prop=1.0)
    with pytest.raises(ValueError):
        InteractionDataset(matrix, fold_in=matrix)
    with pytest.raises(ValueError):
        InteractionDataset(matrix, fold_in=matrix, fold_out=np.zeros((2, 4), dtype=np.float32))

    with pytest.raises(ValueError):
        MultiVAE(n_items=3, hidden_dims=(3, 2), latent_dim=2)
    with pytest.raises(ValueError):
        MultiVAE(n_items=3, hidden_dims=(3, 4, 3), latent_dim=2)
    with pytest.raises(ValueError):
        MultiVAE(n_items=3, input_normalization="bad")(torch.ones((1, 3)), sample=False)
    model = MultiVAE(n_items=3, hidden=4, dimz=2, p=0.0, input_normalization="row_sum")
    model.eval()
    output = model(torch.ones((1, 3)), sample=False)
    assert output.logits.shape == (1, 3)
    assert model.score(torch.ones((1, 3))).shape == (1, 3)
    assert MultiVAE(n_items=3, hidden_dims=(3, 2, 3), latent_dim=2, input_normalization="none")(
        torch.ones((1, 3)), sample=False
    ).logits.shape == (1, 3)
    with pytest.raises(ValueError):
        model(torch.ones((1, 4)), sample=False)


def test_evaluator_tuple_batches_and_recommendation() -> None:
    model = MultiVAE(n_items=5, hidden_dims=(5, 2, 5), latent_dim=2, dropout=0.0)
    data = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32)
    truth = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.float32)
    metrics = evaluate_model(model, [(data, truth)], ks=(1,), max_batches=1)
    assert set(metrics) == {"recall@1", "ndcg@1"}
    assert recommend(model, data, k=2)[0].shape == (1, 2)
    assert evaluate_model(model, [], ks=(1,)) == {"recall@1": 0.0, "ndcg@1": 0.0}


def test_trainer_resume_and_notebook_compatibility(
    prepared_data: PreparedData, tmp_path: Path
) -> None:
    train_loader = DataLoader(InteractionDataset(prepared_data.train), batch_size=2, shuffle=True)
    validation_loader = DataLoader(
        InteractionDataset(
            prepared_data.validation_fold_in,
            eval=True,
            fold_in=prepared_data.validation_fold_in,
            fold_out=prepared_data.validation_fold_out,
        ),
        batch_size=2,
    )
    run_dir = tmp_path / "run"
    first = Trainer(
        MultiVAE(n_items=prepared_data.n_items, hidden_dims=(8, 4, 8), latent_dim=4, dropout=0.0),
        train_loader,
        validation_loader,
        output_dir=run_dir,
        epochs=1,
        validation_ks=(2, 4),
        max_train_batches=1,
        max_eval_batches=1,
        seed=1,
    )
    assert len(first.run()) == 1
    resumed = Trainer(
        MultiVAE(n_items=prepared_data.n_items, hidden_dims=(8, 4, 8), latent_dim=4, dropout=0.0),
        train_loader,
        validation_loader,
        output_dir=run_dir,
        epochs=2,
        validation_ks=(2, 4),
        max_train_batches=1,
        max_eval_batches=1,
        resume_from=run_dir / "last.pt",
        seed=1,
    )
    assert len(resumed.run()) == 1
    history = train_model(
        MultiVAE(n_items=prepared_data.n_items, hidden_dims=(8, 4, 8), latent_dim=4, dropout=0.0),
        train_loader,
        validation_loader,
        epochs=1,
        device="cpu",
        checkpoint_path=tmp_path / "compat" / "model.pt",
        history_path=tmp_path / "compat" / "history.csv",
    )
    assert history and (tmp_path / "compat" / "model.pt").exists()
