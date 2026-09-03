from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import onnx
import pytest
import torch
from bentoml.exceptions import BentoMLException
from onnx import numpy_helper
from pydantic import ValidationError
from starlette.testclient import TestClient

from deep_recsys_lab.data.types import PreparedData
from deep_recsys_lab.model.multvae import MultiVAE
from deep_recsys_lab.serving.model_store import (
    ARTIFACT_SCHEMA_VERSION,
    EXECUTION_PROVIDER,
    INFERENCE_BACKEND,
    ONNX_FILENAME,
    ONNX_OPSET_VERSION,
    PARITY_ATOL,
    PARITY_RTOL,
    SCORE_NOTE,
    ModelArtifactError,
    load_model_artifact,
    register_bento_model,
    validate_model_artifact,
)
from deep_recsys_lab.serving.schemas import (
    BatchRecommendationResult,
    Recommendation,
    RecommendationRequest,
)
from deep_recsys_lab.serving.service import (
    ADAPTIVE_MAX_BATCH_SIZE,
    ADAPTIVE_MAX_LATENCY_MS,
    CATALOG_EXHAUSTED_ERROR,
    INFERENCE_SERVICE_NAME,
    MAX_CONTENT_LENGTH,
    UNKNOWN_MOVIE_ERROR,
    Predictor,
    create_bento_service,
    validate_explicit_model_tag,
)
from deep_recsys_lab.training.checkpoint import save_checkpoint


def _checkpoint(
    prepared_data: PreparedData,
    tmp_path: Path,
    *,
    n_items: int | None = None,
) -> Path:
    model = MultiVAE(
        n_items=n_items or prepared_data.n_items,
        hidden_dims=(8, 4, 8),
        latent_dim=4,
        dropout=0.0,
    )
    path = tmp_path / f"checkpoint-{uuid4().hex}.pt"
    save_checkpoint(
        path,
        model,
        torch.optim.Adam(model.parameters()),
        metrics={"ndcg@100": 0.2},
    )
    return path


def _register(
    prepared_data: PreparedData,
    tmp_path: Path,
    *,
    version: str | None = None,
) -> Any:
    root = prepared_data.root
    assert root is not None
    return register_bento_model(
        _checkpoint(prepared_data, tmp_path),
        root,
        model_version=version or f"test-{uuid4().hex}",
    )


def _refresh_manifest_file(root: Path, filename: str) -> None:
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = root / filename
    manifest["files"][filename] = {
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "size_bytes": payload.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _inference_service(service: Any) -> Any:
    return service.find_dependent_by_name(INFERENCE_SERVICE_NAME)


def test_registration_validates_and_loads_cpu_artifact(
    prepared_data: PreparedData, tmp_path: Path
) -> None:
    registered = _register(prepared_data, tmp_path, version=f"valid-{uuid4().hex}")
    manifest = validate_model_artifact(registered.path, model_tag=str(registered.tag))
    loaded = load_model_artifact(registered.path, model_tag=str(registered.tag))

    assert manifest["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["mapping"]["count"] == prepared_data.n_items
    assert manifest["runtime"]["opset_version"] == ONNX_OPSET_VERSION
    assert manifest["runtime"]["execution_provider"] == EXECUTION_PROVIDER
    assert manifest["exporter"]["framework"] == "pytorch"
    assert manifest["exporter"]["onnxscript_version"]
    assert loaded.scorer.n_items == prepared_data.n_items
    assert loaded.scorer.session.get_providers() == [EXECUTION_PROVIDER]
    assert loaded.metrics == {"ndcg@100": 0.2}
    assert list(loaded.metadata) == prepared_data.item_ids.tolist()
    assert (Path(registered.path) / ONNX_FILENAME).is_file()
    assert not (Path(registered.path) / "weights.pt").exists()
    onnx.checker.check_model(onnx.load(Path(registered.path) / ONNX_FILENAME))

    single = np.zeros((1, prepared_data.n_items), dtype=np.float32)
    batch = np.zeros((3, prepared_data.n_items), dtype=np.float32)
    assert loaded.scorer.score(single).shape == single.shape
    assert loaded.scorer.score(batch).shape == batch.shape


def test_duplicate_model_tag_is_immutable(prepared_data: PreparedData, tmp_path: Path) -> None:
    version = f"duplicate-{uuid4().hex}"
    _register(prepared_data, tmp_path, version=version)
    with pytest.raises(ModelArtifactError, match="already exists"):
        _register(prepared_data, tmp_path, version=version)


def test_checksum_tampering_fails_validation(prepared_data: PreparedData, tmp_path: Path) -> None:
    registered = _register(prepared_data, tmp_path)
    model_path = Path(registered.path) / ONNX_FILENAME
    model_path.write_bytes(model_path.read_bytes() + b"tampered")

    with pytest.raises(ModelArtifactError, match="checksum mismatch"):
        validate_model_artifact(registered.path, model_tag=str(registered.tag))


def test_invalid_onnx_with_valid_checksum_fails_validation(
    prepared_data: PreparedData, tmp_path: Path
) -> None:
    registered = _register(prepared_data, tmp_path)
    root = Path(registered.path)
    (root / ONNX_FILENAME).write_bytes(b"not an ONNX graph")
    _refresh_manifest_file(root, ONNX_FILENAME)

    with pytest.raises(ModelArtifactError, match="initialize ONNX Runtime"):
        validate_model_artifact(root, model_tag=str(registered.tag))


def test_incompatible_onnx_metadata_fails_validation(
    prepared_data: PreparedData, tmp_path: Path
) -> None:
    registered = _register(prepared_data, tmp_path)
    root = Path(registered.path)
    model_path = root / ONNX_FILENAME
    model = onnx.load(model_path)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    metadata["onnx_opset"] = "19"
    onnx.helper.set_model_props(model, metadata)
    onnx.save(model, model_path)
    _refresh_manifest_file(root, ONNX_FILENAME)

    with pytest.raises(ModelArtifactError, match="metadata is missing or incompatible"):
        validate_model_artifact(root, model_tag=str(registered.tag))


def test_incompatible_onnx_io_contract_fails_validation(
    prepared_data: PreparedData, tmp_path: Path
) -> None:
    registered = _register(prepared_data, tmp_path)
    root = Path(registered.path)
    model_path = root / ONNX_FILENAME
    model = onnx.load(model_path)
    model.graph.output[0].name = "scores"
    for node in model.graph.node:
        for output_index, output_name in enumerate(node.output):
            if output_name == "logits":
                node.output[output_index] = "scores"
    onnx.save(model, model_path)
    _refresh_manifest_file(root, ONNX_FILENAME)

    with pytest.raises(ModelArtifactError, match="output contract is incompatible"):
        validate_model_artifact(root, model_tag=str(registered.tag))


def test_non_finite_onnx_output_fails_validation(
    prepared_data: PreparedData, tmp_path: Path
) -> None:
    registered = _register(prepared_data, tmp_path)
    root = Path(registered.path)
    model_path = root / ONNX_FILENAME
    original = onnx.load(model_path)
    shape: list[str | int] = ["batch", prepared_data.n_items]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Mul", ["interactions", "nan"], ["logits"])],
        "non_finite_test",
        [onnx.helper.make_tensor_value_info("interactions", onnx.TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, shape)],
        [numpy_helper.from_array(np.asarray(np.nan, dtype=np.float32), name="nan")],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", ONNX_OPSET_VERSION)],
    )
    onnx.helper.set_model_props(
        model, {entry.key: entry.value for entry in original.metadata_props}
    )
    onnx.save(model, model_path)
    _refresh_manifest_file(root, ONNX_FILENAME)

    with pytest.raises(ModelArtifactError, match="non-finite scores"):
        validate_model_artifact(root, model_tag=str(registered.tag))


def test_schema_v2_artifact_requires_new_registration(
    prepared_data: PreparedData, tmp_path: Path
) -> None:
    registered = _register(prepared_data, tmp_path)
    root = Path(registered.path)
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_schema_version"] = "2.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="re-register the original checkpoint"):
        validate_model_artifact(root, model_tag=str(registered.tag))


def test_malformed_mapping_fails_validation(prepared_data: PreparedData, tmp_path: Path) -> None:
    registered = _register(prepared_data, tmp_path)
    root = Path(registered.path)
    item_ids = json.loads((root / "item_ids.json").read_text(encoding="utf-8"))
    item_ids[-1] = item_ids[0]
    (root / "item_ids.json").write_text(json.dumps(item_ids), encoding="utf-8")
    _refresh_manifest_file(root, "item_ids.json")

    with pytest.raises(ModelArtifactError, match="not unique"):
        validate_model_artifact(root, model_tag=str(registered.tag))


def test_incompatible_checkpoint_shapes_fail_registration(
    prepared_data: PreparedData, tmp_path: Path
) -> None:
    incompatible = _checkpoint(prepared_data, tmp_path, n_items=prepared_data.n_items + 1)
    expected_architecture = {
        "name": "multvae",
        "n_items": prepared_data.n_items,
        "hidden_dims": [8, 4, 8],
        "latent_dim": 4,
        "dropout": 0.0,
        "input_normalization": "l2",
    }

    with pytest.raises(ModelArtifactError, match="shapes do not match"):
        assert prepared_data.root is not None
        register_bento_model(
            incompatible,
            prepared_data.root,
            model_version=f"incompatible-{uuid4().hex}",
            model_config=expected_architecture,
        )


def test_registered_onnx_matches_pytorch_for_dynamic_batches(
    prepared_data: PreparedData, tmp_path: Path
) -> None:
    checkpoint = _checkpoint(prepared_data, tmp_path)
    assert prepared_data.root is not None
    registered = register_bento_model(
        checkpoint,
        prepared_data.root,
        model_version=f"parity-{uuid4().hex}",
    )
    loaded = load_model_artifact(registered.path, model_tag=str(registered.tag))
    model = MultiVAE(
        n_items=prepared_data.n_items,
        hidden_dims=(8, 4, 8),
        latent_dim=4,
        dropout=0.0,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    interactions = prepared_data.train[:3].toarray().astype(np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(interactions), sample=False).logits.numpy()
    actual = loaded.scorer.score(interactions)

    np.testing.assert_allclose(actual, expected, rtol=PARITY_RTOL, atol=PARITY_ATOL)


def test_missing_and_corrupt_models_fail_service_startup(
    prepared_data: PreparedData, tmp_path: Path
) -> None:
    missing_tag = f"deep_recsys:missing-{uuid4().hex}"
    with pytest.raises(BentoMLException, match="could not be found"):
        _inference_service(create_bento_service(missing_tag)).inner()

    registered = _register(prepared_data, tmp_path)
    model_path = Path(registered.path) / ONNX_FILENAME
    model_path.write_bytes(model_path.read_bytes() + b"tampered")
    service = create_bento_service(str(registered.tag))
    with pytest.raises(ModelArtifactError, match="checksum mismatch"):
        _inference_service(service).inner()


def test_service_graph_configures_adaptive_inference_dependency(
    prepared_data: PreparedData, tmp_path: Path
) -> None:
    registered = _register(prepared_data, tmp_path)
    service = create_bento_service(str(registered.tag))
    inference = _inference_service(service)
    batch_api = inference.apis["recommend_batch"]

    assert service.name == "deep_recsys_service"
    assert service.dependencies["inference"].on is inference
    assert service.models == []
    assert inference.name == INFERENCE_SERVICE_NAME
    assert [str(model.tag) for model in inference.models] == [str(registered.tag)]
    assert inference.config["workers"] == 1
    assert inference.config["metrics"]["enabled"] is True
    assert inference.config["traffic"]["max_concurrency"] == 8
    assert batch_api.batchable is True
    assert batch_api.max_batch_size == ADAPTIVE_MAX_BATCH_SIZE
    assert batch_api.max_latency_ms == ADAPTIVE_MAX_LATENCY_MS


def test_internal_batch_result_requires_success_xor_error() -> None:
    recommendation = Recommendation(movie_id=1, title="Movie 1", genres="Drama", score=0.5)

    assert (
        BatchRecommendationResult(model_version="v1", recommendations=[recommendation]).error
        is None
    )
    assert BatchRecommendationResult(model_version="v1", error="invalid").recommendations == []
    with pytest.raises(ValidationError, match="must contain recommendations"):
        BatchRecommendationResult(model_version="v1")
    with pytest.raises(ValidationError, match="cannot contain recommendations"):
        BatchRecommendationResult(
            model_version="v1",
            recommendations=[recommendation],
            error="invalid",
        )


def test_service_contract_auth_observability_and_deterministic_ranking(
    prepared_data: PreparedData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered = _register(
        prepared_data,
        tmp_path,
        version=f"api-{uuid4().hex}",
    )
    monkeypatch.setenv("DEEP_RECSYS_API_KEY", "secret")
    service = create_bento_service(str(registered.tag))
    known = [int(value) for value in prepared_data.item_ids[:5]]

    with TestClient(service.to_asgi()) as client:
        assert client.get("/livez").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert client.get("/").status_code == 200
        assert client.get("/docs.json").status_code == 200
        assert client.post("/model_info").status_code == 401
        assert client.post("/recommend", json={"movie_ids": known, "top_k": 3}).status_code == 401

        headers = {"X-API-Key": "secret", "X-Request-ID": "request-123"}
        first = client.post(
            "/recommend",
            headers=headers,
            json={"movie_ids": known, "top_k": 3},
        )
        second = client.post(
            "/recommend",
            headers={"X-API-Key": "secret"},
            json={"movie_ids": known, "top_k": 3},
        )
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        body = first.json()
        assert body["request_id"] == "request-123"
        assert first.headers["X-Request-ID"] == "request-123"
        assert UUID(second.json()["request_id"])
        assert second.headers["X-Request-ID"] == second.json()["request_id"]
        assert body["model_version"] == str(registered.tag.version)
        assert body["recommendations"] == second.json()["recommendations"]
        assert all(item["movie_id"] not in known for item in body["recommendations"])
        assert all(item["title"].startswith("Movie ") for item in body["recommendations"])
        scores = [item["score"] for item in body["recommendations"]]
        assert scores == sorted(scores, reverse=True)

        info = client.post("/model_info", headers={"X-API-Key": "secret"})
        assert info.status_code == 200, info.text
        assert info.json() == {
            "model_version": str(registered.tag.version),
            "model_tag": str(registered.tag),
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "item_count": prepared_data.n_items,
            "mapping_hash": load_model_artifact(
                registered.path, model_tag=str(registered.tag)
            ).manifest["mapping"]["sha256"],
            "score_note": SCORE_NOTE,
            "inference_backend": INFERENCE_BACKEND,
            "execution_provider": EXECUTION_PROVIDER,
            "onnx_opset": ONNX_OPSET_VERSION,
        }
        invalid_payloads: list[dict[str, Any]] = [
            {"movie_ids": [1, 2, 3, 4], "top_k": 1},
            {"movie_ids": [1, 2, 3, 4, 4], "top_k": 1},
            {"movie_ids": [1, 2, 3, 4, True], "top_k": 1},
            {"movie_ids": [1, 2, 3, 4, 5], "top_k": 0},
            {"movie_ids": [1, 2, 3, 4, 5], "top_k": 101},
            {"movie_ids": list(range(1, 502)), "top_k": 1},
        ]
        for payload in invalid_payloads:
            invalid = client.post("/recommend", headers=headers, json=payload)
            assert invalid.status_code == 400, invalid.text

        unknown = client.post(
            "/recommend",
            headers=headers,
            json={"movie_ids": [*known, 999_999], "top_k": 1},
        )
        all_items = client.post(
            "/recommend",
            headers=headers,
            json={
                "movie_ids": [int(value) for value in prepared_data.item_ids],
                "top_k": 1,
            },
        )
        oversized = client.post(
            "/recommend",
            headers=headers,
            json={
                "movie_ids": [1, 2, 3, 4, 5],
                "top_k": 1,
                "padding": "x" * MAX_CONTENT_LENGTH,
            },
        )
        assert unknown.status_code == 400, unknown.text
        assert all_items.status_code == 400, all_items.text
        assert oversized.status_code == 413, oversized.text


def test_vectorized_batch_matches_sequential_and_scores_once(
    prepared_data: PreparedData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered = _register(prepared_data, tmp_path)
    predictor = Predictor(load_model_artifact(registered.path, model_tag=str(registered.tag)))
    item_ids = [int(value) for value in prepared_data.item_ids]
    requests = [
        RecommendationRequest(movie_ids=item_ids[:5], top_k=3),
        RecommendationRequest(movie_ids=item_ids[2:8], top_k=100),
    ]
    expected = [predictor.recommend(request.movie_ids, request.top_k) for request in requests]
    original_score = predictor.loaded.scorer.score
    scored_inputs: list[np.ndarray] = []

    def score_once(interactions: np.ndarray) -> np.ndarray:
        scored_inputs.append(interactions.copy())
        return original_score(interactions)

    monkeypatch.setattr(predictor.loaded.scorer, "score", score_once)
    results = predictor.recommend_batch(requests)

    assert [result.error for result in results] == [None, None]
    assert [result.recommendations for result in results] == expected
    assert len(scored_inputs) == 1
    assert scored_inputs[0].shape == (2, prepared_data.n_items)
    assert scored_inputs[0].dtype == np.float32
    assert scored_inputs[0].flags.c_contiguous
    np.testing.assert_array_equal(scored_inputs[0][0], np.isin(item_ids, item_ids[:5]))
    np.testing.assert_array_equal(scored_inputs[0][1], np.isin(item_ids, item_ids[2:8]))


def test_vectorized_batch_isolates_domain_errors_and_skips_invalid_batches(
    prepared_data: PreparedData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered = _register(prepared_data, tmp_path)
    predictor = Predictor(load_model_artifact(registered.path, model_tag=str(registered.tag)))
    item_ids = [int(value) for value in prepared_data.item_ids]
    valid = RecommendationRequest(movie_ids=item_ids[:5], top_k=2)
    unknown = RecommendationRequest(movie_ids=[*item_ids[:5], 999_999], top_k=1)
    exhausted = RecommendationRequest(movie_ids=item_ids, top_k=1)
    original_score = predictor.loaded.scorer.score
    scored_shapes: list[tuple[int, ...]] = []

    def track_score(interactions: np.ndarray) -> np.ndarray:
        scored_shapes.append(interactions.shape)
        return original_score(interactions)

    monkeypatch.setattr(predictor.loaded.scorer, "score", track_score)
    results = predictor.recommend_batch([valid, unknown, exhausted])

    assert results[0].error is None
    assert len(results[0].recommendations) == 2
    assert results[1].error == UNKNOWN_MOVIE_ERROR
    assert results[1].recommendations == []
    assert results[2].error == CATALOG_EXHAUSTED_ERROR
    assert results[2].recommendations == []
    assert scored_shapes == [(1, prepared_data.n_items)]

    invalid_results = predictor.recommend_batch([unknown, exhausted])
    assert [result.error for result in invalid_results] == [
        UNKNOWN_MOVIE_ERROR,
        CATALOG_EXHAUSTED_ERROR,
    ]
    assert predictor.recommend_batch([]) == []
    assert scored_shapes == [(1, prepared_data.n_items)]


def test_predictor_is_deterministic_under_concurrency(
    prepared_data: PreparedData, tmp_path: Path
) -> None:
    registered = _register(prepared_data, tmp_path)
    predictor = Predictor(load_model_artifact(registered.path, model_tag=str(registered.tag)))
    known = [int(value) for value in prepared_data.item_ids[:5]]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: predictor.recommend(known, 3), range(32)))

    expected = results[0]
    assert all(result == expected for result in results)


@pytest.mark.parametrize(
    "tag",
    ["deep_recsys:latest", "deep_recsys", "other:model-v1", "deep_recsys:"],
)
def test_explicit_model_tag_is_required(tag: str) -> None:
    with pytest.raises(ValueError, match="must be explicit"):
        validate_explicit_model_tag(tag)
