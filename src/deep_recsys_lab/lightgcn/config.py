"""Typed Hydra configuration for the isolated LightGCN experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..config import compose_hydra_config


@dataclass(frozen=True)
class LightGCNDataConfig:
    name: str = "movielens20m"
    raw_dir: str = "data/raw"
    positive_rating_threshold: float = 4.0
    min_positive_ratings: int = 5
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    expected_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class LightGCNModelConfig:
    embedding_dim: int = 64
    layers: int = 3


@dataclass(frozen=True)
class LightGCNTrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 262_144
    threads: int = 8
    max_steps: int = 4_000
    minimum_steps: int = 500
    validation_every: int = 50
    patience: int = 10
    wall_time_hours: float = 8.0
    checkpoint_every: int = 50


@dataclass(frozen=True)
class LightGCNSweepConfig:
    # Empty values preserve the original single-architecture behavior and make
    # the base config backward compatible. A tuning config can provide several
    # embedding sizes and learning rates to form a Cartesian search space.
    embedding_dims: tuple[int, ...] = ()
    layers: tuple[int, ...] = (2, 3, 4)
    learning_rates: tuple[float, ...] = ()
    l2_values: tuple[float, ...] = (1e-5, 1e-4, 1e-3)
    round1_steps: int = 100
    round2_steps: int = 300
    finalists: int = 3


@dataclass(frozen=True)
class LightGCNEvaluationConfig:
    k: int = 20
    user_batch_size: int = 512


@dataclass(frozen=True)
class LightGCNReportConfig:
    output: str = "public/reports/lightgcn.json"


@dataclass(frozen=True)
class LightGCNConfig:
    seed: int = 98_765
    device: str = "cpu"
    data_dir: str = "data/lightgcn/ml-20m-v1"
    output_dir: str = "outputs/lightgcn/ml-20m-v1"
    data: LightGCNDataConfig = field(default_factory=LightGCNDataConfig)
    model: LightGCNModelConfig = field(default_factory=LightGCNModelConfig)
    training: LightGCNTrainingConfig = field(default_factory=LightGCNTrainingConfig)
    sweep: LightGCNSweepConfig = field(default_factory=LightGCNSweepConfig)
    evaluation: LightGCNEvaluationConfig = field(default_factory=LightGCNEvaluationConfig)
    report: LightGCNReportConfig = field(default_factory=LightGCNReportConfig)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe resolved representation."""

        return asdict(self)


def to_lightgcn_config(config: DictConfig | dict[str, Any]) -> LightGCNConfig:
    """Convert a Hydra config into the dedicated immutable schema."""

    raw = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    assert isinstance(raw, dict)
    data = dict(raw.get("data", {}))
    model = dict(raw.get("model", {}))
    training = dict(raw.get("training", {}))
    sweep = dict(raw.get("sweep", {}))
    evaluation = dict(raw.get("evaluation", {}))
    report = dict(raw.get("report", {}))
    return LightGCNConfig(
        seed=int(raw.get("seed", 98_765)),
        device=str(raw.get("device", "cpu")),
        data_dir=str(raw.get("data_dir", "data/lightgcn/ml-20m-v1")),
        output_dir=str(raw.get("output_dir", "outputs/lightgcn/ml-20m-v1")),
        data=LightGCNDataConfig(
            name=str(data.get("name", "movielens20m")),
            raw_dir=str(data.get("raw_dir", "data/raw")),
            positive_rating_threshold=float(data.get("positive_rating_threshold", 4.0)),
            min_positive_ratings=int(data.get("min_positive_ratings", 5)),
            validation_ratio=float(data.get("validation_ratio", 0.1)),
            test_ratio=float(data.get("test_ratio", 0.1)),
            expected_counts={
                str(k): int(v) for k, v in dict(data.get("expected_counts", {})).items()
            },
        ),
        model=LightGCNModelConfig(
            embedding_dim=int(model.get("embedding_dim", 64)),
            layers=int(model.get("layers", 3)),
        ),
        training=LightGCNTrainingConfig(
            learning_rate=float(training.get("learning_rate", 1e-3)),
            batch_size=int(training.get("batch_size", 262_144)),
            threads=int(training.get("threads", 8)),
            max_steps=int(training.get("max_steps", 4_000)),
            minimum_steps=int(training.get("minimum_steps", 500)),
            validation_every=int(training.get("validation_every", 50)),
            patience=int(training.get("patience", 10)),
            wall_time_hours=float(training.get("wall_time_hours", 8.0)),
            checkpoint_every=int(training.get("checkpoint_every", 50)),
        ),
        sweep=LightGCNSweepConfig(
            embedding_dims=tuple(
                int(value) for value in sweep.get("embedding_dims", ())
            ),
            layers=tuple(int(value) for value in sweep.get("layers", (2, 3, 4))),
            learning_rates=tuple(
                float(value) for value in sweep.get("learning_rates", ())
            ),
            l2_values=tuple(float(value) for value in sweep.get("l2_values", (1e-5, 1e-4, 1e-3))),
            round1_steps=int(sweep.get("round1_steps", 100)),
            round2_steps=int(sweep.get("round2_steps", 300)),
            finalists=int(sweep.get("finalists", 3)),
        ),
        evaluation=LightGCNEvaluationConfig(
            k=int(evaluation.get("k", 20)),
            user_batch_size=int(evaluation.get("user_batch_size", 512)),
        ),
        report=LightGCNReportConfig(
            output=str(report.get("output", "public/reports/lightgcn.json"))
        ),
    )


def compose_lightgcn_config(
    config_name: str = "lightgcn", overrides: list[str] | None = None
) -> LightGCNConfig:
    """Compose and validate the dedicated LightGCN Hydra config."""

    config = to_lightgcn_config(compose_hydra_config(config_name, overrides or ()))
    if config.device != "cpu":
        raise ValueError("LightGCN reproduction is pinned to CPU for sparse FP32 propagation")
    if (
        config.model.embedding_dim <= 0
        or not config.sweep.layers
        or min(config.sweep.layers) < 1
        or any(value <= 0 for value in config.sweep.embedding_dims)
    ):
        raise ValueError("embedding_dim and all layer counts must be positive")
    if any(value <= 0 for value in config.sweep.learning_rates):
        raise ValueError("all sweep learning rates must be positive")
    if any(value < 0 for value in config.sweep.l2_values):
        raise ValueError("all sweep L2 values must be non-negative")
    if config.training.batch_size <= 0 or config.training.wall_time_hours <= 0:
        raise ValueError("batch_size and wall_time_hours must be positive")
    if (
        config.training.learning_rate <= 0
        or config.training.max_steps <= 0
        or config.training.minimum_steps <= 0
        or config.training.minimum_steps > config.training.max_steps
        or config.training.validation_every <= 0
        or config.training.patience <= 0
        or config.training.checkpoint_every <= 0
    ):
        raise ValueError("training horizons and intervals must be positive and ordered")
    if (
        config.sweep.round1_steps <= 0
        or config.sweep.round1_steps > config.sweep.round2_steps
        or config.sweep.round2_steps <= 0
        or config.sweep.finalists <= 0
    ):
        raise ValueError("sweep stages must be positive and ordered")
    if config.data.validation_ratio <= 0 or config.data.test_ratio <= 0:
        raise ValueError("validation_ratio and test_ratio must be positive")
    if config.data.validation_ratio + config.data.test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be less than one")
    return config


def resolve_path(value: str) -> Path:
    """Resolve a configured path without requiring the CLI working directory."""

    return Path(value).expanduser().resolve()
