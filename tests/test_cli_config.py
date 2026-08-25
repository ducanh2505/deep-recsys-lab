from __future__ import annotations

from deep_recsys_lab.config import compose_hydra_config, to_app_config
from deep_recsys_lab.device import select_device
from deep_recsys_lab.model import build_model


def test_hydra_groups_compose_and_model_registry() -> None:
    config = to_app_config(
        compose_hydra_config(
            overrides=[
                "dataset=synthetic",
                "model.hidden_dims=[8,4,8]",
                "model.latent_dim=4",
                "trainer.epochs=1",
            ]
        )
    )
    assert config.dataset.name == "synthetic"
    assert config.model.hidden_dims == (8, 4, 8)
    assert config.trainer.epochs == 1
    assert str(select_device("cpu")) == "cpu"
    assert build_model("multvae", n_items=4, hidden_dims=(4, 2, 4), latent_dim=2).n_items == 4
