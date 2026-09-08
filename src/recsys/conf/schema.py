"""Structured configuration schema composed through Hydra."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_module
from hydra.errors import MissingConfigException
from omegaconf import OmegaConf


@dataclass
class ColumnsConfig:
    user_id: str = "user_id"
    item_id: str = "item_id"
    value: str | None = None
    timestamp: str | None = None


@dataclass
class SplitConfig:
    strategy: str = "user_holdout"
    validation_ratio: float = 0.1
    test_ratio: float = 0.2
    seed: int = 42


@dataclass
class DatasetConfig:
    name: str = "synthetic"
    path: str | None = None
    format: str = "csv"
    columns: ColumnsConfig = field(default_factory=ColumnsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "multivae"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    components: list[str] = field(default_factory=lambda: ["popularity", "itemknn"])
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class FusionConfig:
    name: str = "rrf"
    weights: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankerConfig:
    name: str = "none"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 2
    batch_size: int = 128
    learning_rate: float = 0.001
    device: str = "cpu"


@dataclass
class EvaluationConfig:
    top_k: int = 10


@dataclass
class ServingConfig:
    host: str = "0.0.0.0"
    port: int = 3000
    max_body_bytes: int = 65536


@dataclass
class PlatformConfig:
    workspace_root: str = "."
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)


def load_config(
    *,
    external: Path | None = None,
    overrides: list[str] | None = None,
    workspace_root: Path | None = None,
) -> PlatformConfig:
    """Compose packaged defaults, then external YAML, then dotlist overrides."""

    external_layer = OmegaConf.load(external) if external is not None else None
    selected_groups: dict[str, Any] = {}
    external_groups: dict[str, Any] = {}

    def group_schema(key: str) -> Any:
        value = OmegaConf.structured(DatasetConfig if key == "dataset" else ModelConfig)
        OmegaConf.set_struct(OmegaConf.select(value, "parameters"), False)
        return value

    with initialize_config_module(config_module="recsys.conf", version_base=None):
        packaged = compose(config_name="config")
        for key in ("dataset", "model"):
            external_name = (
                OmegaConf.select(external_layer, f"{key}.name")
                if external_layer is not None
                else None
            )
            if isinstance(external_name, str) and external_name != packaged[key].name:
                schema = group_schema(key)
                try:
                    selected = compose(config_name="config", overrides=[f"{key}={external_name}"])
                except MissingConfigException:
                    external_groups[key] = schema
                else:
                    external_groups[key] = OmegaConf.merge(schema, selected[key])
        for override in overrides or []:
            key = override.partition("=")[0]
            if key in {"dataset", "model"}:
                selected = compose(config_name="config", overrides=[override])
                schema = group_schema(key)
                selected_groups[override] = OmegaConf.to_object(
                    OmegaConf.merge(schema, selected[key])
                )
    extension_paths = (
        "dataset.parameters",
        "model.parameters",
        "retrieval.parameters",
        "fusion.weights",
        "fusion.parameters",
        "reranker.parameters",
    )
    structured = OmegaConf.structured(PlatformConfig)
    for extension_path in extension_paths:
        OmegaConf.set_struct(OmegaConf.select(structured, extension_path), False)
    merged = OmegaConf.merge(structured, packaged)
    for key, group in external_groups.items():
        OmegaConf.update(merged, key, group, merge=False)
    layers: list[Any] = [merged]
    if external_layer is not None:
        layers.append(external_layer)
    if workspace_root is not None:
        layers.append(OmegaConf.create({"workspace_root": str(workspace_root)}))
    merged = OmegaConf.merge(*layers)
    for extension_path in extension_paths:
        OmegaConf.set_struct(OmegaConf.select(merged, extension_path), False)
    for override in overrides or []:
        key = override.partition("=")[0]
        if override in selected_groups:
            OmegaConf.update(merged, key, selected_groups[override], merge=False)
            OmegaConf.set_struct(OmegaConf.select(merged, f"{key}.parameters"), False)
        else:
            merged = OmegaConf.merge(merged, OmegaConf.from_dotlist([override]))
    OmegaConf.resolve(merged)
    value = OmegaConf.to_object(merged)
    if not isinstance(value, PlatformConfig):
        raise TypeError("Hydra did not produce a PlatformConfig")
    return value
