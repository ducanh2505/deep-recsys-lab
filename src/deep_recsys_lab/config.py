"""Hydra composition and small typed configuration adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
if not CONFIG_DIR.exists():
    CONFIG_DIR = Path.cwd() / "configs"


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "movielens20m"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    download_url: str = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    archive_name: str = "ml-20m.zip"
    ratings_file: str = "ratings.csv"
    movies_file: str = "movies.csv"
    positive_rating_threshold: float = 4.0
    min_positive_ratings: int = 5
    n_validation_users: int = 10_000
    n_test_users: int = 10_000
    fold_in_ratio: float = 0.8
    seed: int = 98_765
    strict_user_counts: bool = True


@dataclass(frozen=True)
class ModelConfig:
    name: str = "multvae"
    n_items: int | None = None
    hidden_dims: tuple[int, ...] = (600, 200, 600)
    latent_dim: int = 200
    dropout: float = 0.5
    input_normalization: str = "l2"


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 200
    batch_size: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    beta_cap: float = 0.2
    total_anneal_steps: int = 200_000
    validation_every: int = 1
    num_workers: int = 0
    gradient_clip_norm: float | None = None
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    precision: str = "fp32"
    resume_from: str | None = None
    save_every_epoch: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    ks: tuple[int, ...] = (20, 50, 100)
    mask_seen: bool = True
    max_batches: int | None = None


@dataclass(frozen=True)
class AppConfig:
    seed: int = 98_765
    device: str = "auto"
    output_dir: str = "outputs/multvae"
    data_dir: str = "data/processed"
    model_version: str = "multvae-v1"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: dict[str, Any] = field(default_factory=dict)


def compose_hydra_config(config_name: str = "config", overrides: Iterable[str] = ()) -> DictConfig:
    """Compose a config using Hydra's group/default-list mechanism."""

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name, overrides=list(overrides))


def _tuple_ints(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


def to_app_config(config: DictConfig | dict[str, Any]) -> AppConfig:
    """Convert a resolved Hydra config into a typed immutable config."""

    raw = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    assert isinstance(raw, dict)
    dataset_raw = dict(raw.get("dataset", {}))
    model_raw = dict(raw.get("model", {}))
    trainer_raw = dict(raw.get("trainer", {}))
    evaluation_raw = dict(raw.get("evaluation", {}))
    return AppConfig(
        seed=int(raw.get("seed", 98_765)),
        device=str(raw.get("device", "auto")),
        output_dir=str(raw.get("output_dir", "outputs/multvae")),
        data_dir=str(raw.get("data_dir", dataset_raw.get("processed_dir", "data/processed"))),
        model_version=str(raw.get("model_version", "multvae-v1")),
        dataset=DatasetConfig(
            **{
                key: value
                for key, value in dataset_raw.items()
                if key in DatasetConfig.__dataclass_fields__
            }
        ),
        model=ModelConfig(
            **{
                **{
                    key: value
                    for key, value in model_raw.items()
                    if key in ModelConfig.__dataclass_fields__
                },
                "hidden_dims": _tuple_ints(model_raw.get("hidden_dims", (600, 200, 600))),
            }
        ),
        trainer=TrainerConfig(
            **{
                key: value
                for key, value in trainer_raw.items()
                if key in TrainerConfig.__dataclass_fields__
            }
        ),
        evaluation=EvaluationConfig(
            **{
                **{
                    key: value
                    for key, value in evaluation_raw.items()
                    if key in EvaluationConfig.__dataclass_fields__
                },
                "ks": _tuple_ints(evaluation_raw.get("ks", (20, 50, 100))),
            }
        ),
        experiment=dict(raw.get("experiment", {})),
    )


def resolved_config_dict(config: DictConfig | AppConfig | dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe resolved config for manifests and run directories."""

    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=True)
        assert isinstance(value, dict)
        return cast(dict[str, Any], value)
    if isinstance(config, AppConfig):
        return {
            "seed": config.seed,
            "device": config.device,
            "output_dir": config.output_dir,
            "data_dir": config.data_dir,
            "model_version": config.model_version,
            "dataset": config.dataset.__dict__,
            "model": {**config.model.__dict__, "hidden_dims": list(config.model.hidden_dims)},
            "trainer": config.trainer.__dict__,
            "evaluation": {**config.evaluation.__dict__, "ks": list(config.evaluation.ks)},
            "experiment": config.experiment,
        }
    return cast(dict[str, Any], dict(config))
