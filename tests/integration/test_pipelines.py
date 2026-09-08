from __future__ import annotations

from pathlib import Path

import pytest

from recsys.artifacts import load_artifact
from recsys.conf.schema import load_config
from recsys.experiments import run_pipeline
from recsys.serving import RecommendationEngine, RecommendationRequest


def tiny_overrides(model: str) -> list[str]:
    values = [
        f"model={model}",
        "dataset.parameters.users=6",
        "dataset.parameters.items=18",
        "dataset.parameters.interactions_per_user=6",
        "training.epochs=1",
        "training.batch_size=8",
        "evaluation.top_k=3",
    ]
    if model == "multivae":
        values.extend(["model.parameters.hidden_dim=8", "model.parameters.latent_dim=4"])
    if model == "lightgcn":
        values.extend(["model.parameters.embedding_dim=4", "model.parameters.steps=1"])
    if model == "hybrid":
        values.extend(
            [
                "retrieval.components=[popularity,itemknn]",
                "retrieval.parameters.itemknn.neighbours=4",
            ]
        )
    return values


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model", "runtime"),
    [("multivae", "onnx_dense"), ("lightgcn", "embedding"), ("hybrid", "hybrid")],
)
def test_synthetic_pipeline_round_trip(tmp_path: Path, model: str, runtime: str) -> None:
    config = load_config(workspace_root=tmp_path, overrides=tiny_overrides(model))
    result = run_pipeline(config)
    artifact = load_artifact(result["artifact"])
    assert artifact.manifest.runtime == runtime
    assert result["metrics"]["evaluated_users"] == 6
    engine = RecommendationEngine(artifact)
    if "known_user" in artifact.manifest.capabilities:
        request = RecommendationRequest(user_id=artifact.user_ids[0], top_k=3)
    else:
        request = RecommendationRequest(interactions=[{"item_id": artifact.item_ids[0]}], top_k=3)
    response = engine.recommend(request)
    assert response.artifact_id == artifact.manifest.artifact_id
    assert len(response.recommendations) == 3


@pytest.mark.integration
@pytest.mark.parametrize("model", ["bpr", "sasrec", "two_tower"])
def test_additional_model_runtime_round_trip(tmp_path: Path, model: str) -> None:
    config = load_config(
        workspace_root=tmp_path,
        overrides=[
            *tiny_overrides(model),
            "model.parameters.embedding_dim=4",
            "model.parameters.steps=1",
        ],
    )
    result = run_pipeline(config)
    artifact = load_artifact(result["artifact"])
    engine = RecommendationEngine(artifact)
    request = RecommendationRequest(user_id=artifact.user_ids[0], top_k=2)
    assert len(engine.recommend(request).recommendations) == 2


@pytest.mark.integration
def test_learned_hybrid_fusion_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("lightgbm")
    config = load_config(
        workspace_root=tmp_path,
        overrides=[
            *tiny_overrides("hybrid"),
            "fusion.name=learned_lightgbm",
            "fusion.parameters.num_boost_round=2",
            "fusion.parameters.min_data_in_leaf=1",
        ],
    )
    result = run_pipeline(config)
    artifact = load_artifact(result["artifact"])
    assert (artifact.root / "fusion_model.txt").is_file()
    request = RecommendationRequest(interactions=[{"item_id": artifact.item_ids[0]}], top_k=2)
    assert len(RecommendationEngine(artifact).recommend(request).recommendations) == 2


@pytest.mark.integration
def test_sequential_retriever_supports_both_hybrid_query_modes(tmp_path: Path) -> None:
    config = load_config(
        workspace_root=tmp_path,
        overrides=[
            *tiny_overrides("hybrid"),
            "retrieval.components=[popularity,markov]",
        ],
    )
    artifact = load_artifact(run_pipeline(config)["artifact"])
    assert {mode.value for mode in artifact.manifest.capabilities} == {"known_user", "history"}
    engine = RecommendationEngine(artifact)
    known = RecommendationRequest(user_id=artifact.user_ids[0], top_k=2)
    history = RecommendationRequest(interactions=[{"item_id": artifact.item_ids[0]}], top_k=2)
    assert len(engine.recommend(known).recommendations) == 2
    assert len(engine.recommend(history).recommendations) == 2
