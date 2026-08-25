from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from deep_recsys_lab.data.dataset import InteractionDataset
from deep_recsys_lab.evaluation.evaluator import evaluate_model
from deep_recsys_lab.model.multvae import MultiVAE
from deep_recsys_lab.reproducibility import seed_everything
from deep_recsys_lab.training.checkpoint import load_checkpoint, save_checkpoint
from deep_recsys_lab.training.loss import kl_divergence, multinomial_nll, multivae_loss
from deep_recsys_lab.training.schedule import beta_at_step


def test_multvae_shapes_sampling_and_eval_mode() -> None:
    seed_everything(98765)
    model = MultiVAE(n_items=12, hidden_dims=(8, 4, 8), latent_dim=4, dropout=0.5)
    data = torch.eye(12)[:3]
    model.train()
    first = model(data, sample=True)
    second = model(data, sample=True)
    assert first.logits.shape == (3, 12)
    assert first.mu is not None and first.mu.shape == (3, 4)
    assert first.logvar is not None and first.logvar.shape == (3, 4)
    assert not torch.equal(first.logits, second.logits)
    model.eval()
    assert torch.equal(model(data, sample=False).logits, model(data, sample=False).logits)
    assert model.architecture_config()["hidden_dims"] == [8, 4, 8]


def test_multinomial_loss_and_kl_are_finite() -> None:
    logits = torch.tensor([[0.0, 1.0, -1.0]])
    target = torch.tensor([[0.0, 1.0, 0.0]])
    nll = multinomial_nll(logits, target)
    assert torch.isclose(nll, -torch.log_softmax(logits, dim=1)[0, 1])
    mu = torch.zeros((1, 2))
    logvar = torch.zeros((1, 2))
    assert kl_divergence(mu, logvar).item() == 0.0
    terms = multivae_loss(logits, target, mu, logvar, beta=0.2)
    assert set(terms) == {"loss", "nll", "kl"}
    assert all(torch.isfinite(value) for value in terms.values())


def test_beta_schedule() -> None:
    assert beta_at_step(0, 200_000, 0.2) == 0.0
    assert beta_at_step(100_000, 200_000, 0.2) == 0.1
    assert beta_at_step(200_000, 200_000, 0.2) == 0.2
    assert beta_at_step(999_999, 200_000, 0.2) == 0.2


def test_checkpoint_restores_state_and_rng(tmp_path: Path) -> None:
    seed_everything(42)
    model = MultiVAE(n_items=6, hidden_dims=(6, 3, 6), latent_dim=3, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=3, global_step=11, beta=0.2)
    expected_random = torch.rand(3)
    restored_model = MultiVAE(n_items=6, hidden_dims=(6, 3, 6), latent_dim=3, dropout=0.0)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
    metadata = load_checkpoint(path, restored_model, restored_optimizer)
    assert metadata["epoch"] == 3
    assert metadata["global_step"] == 11
    assert metadata["model_config"]["n_items"] == 6
    assert torch.equal(expected_random, torch.rand(3))


def test_evaluation_sampling_and_metrics(prepared_data) -> None:
    model = MultiVAE(
        n_items=prepared_data.n_items, hidden_dims=(8, 4, 8), latent_dim=4, dropout=0.0
    )
    loader = DataLoader(
        InteractionDataset(
            prepared_data.validation_fold_in,
            eval=True,
            fold_in=prepared_data.validation_fold_in,
            fold_out=prepared_data.validation_fold_out,
        ),
        batch_size=2,
    )
    metrics = evaluate_model(model, loader, ks=(2, 4), device="cpu")
    assert set(metrics) == {"recall@2", "recall@4", "ndcg@2", "ndcg@4"}
    assert all(np.isfinite(value) for value in metrics.values())
