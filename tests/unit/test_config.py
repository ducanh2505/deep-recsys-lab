from __future__ import annotations

from pathlib import Path

from recsys.conf.schema import load_config


def test_packaged_config_group_and_dotlist_overrides(tmp_path: Path) -> None:
    config = load_config(
        workspace_root=tmp_path,
        overrides=["model=lightgcn", "model.parameters.layers=3", "evaluation.top_k=7"],
    )
    assert config.workspace_root == str(tmp_path)
    assert config.model.name == "lightgcn"
    assert config.model.parameters == {"embedding_dim": 32, "layers": 3, "l2": 0.0001}
    assert config.evaluation.top_k == 7


def test_external_config_precedes_cli_override(tmp_path: Path) -> None:
    external = tmp_path / "config.yaml"
    external.write_text(
        "model:\n"
        "  name: lightgcn\n"
        "  parameters:\n"
        "    layers: 4\n"
        "training:\n"
        "  epochs: 8\n"
        "evaluation:\n"
        "  top_k: 12\n"
    )
    config = load_config(external=external, overrides=["training.epochs=2"])
    assert config.training.epochs == 2
    assert config.evaluation.top_k == 12
    assert config.model.name == "lightgcn"
    assert config.model.parameters == {"embedding_dim": 32, "layers": 4, "l2": 0.0001}


def test_dataset_group_composes_from_package() -> None:
    config = load_config(overrides=["dataset=tabular", "dataset.path=events.csv"])
    assert config.dataset.name == "tabular"
    assert config.dataset.path == "events.csv"


def test_external_plugin_identity_starts_with_empty_parameters(tmp_path: Path) -> None:
    external = tmp_path / "plugin.yaml"
    external.write_text("model:\n  name: custom_model\n  parameters:\n    width: 5\n")
    config = load_config(external=external)
    assert config.model.name == "custom_model"
    assert config.model.parameters == {"width": 5}
