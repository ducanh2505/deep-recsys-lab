"""Versioned ONNX artifacts and CPU inference sessions for BentoML serving."""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import bentoml
import numpy as np
import onnxruntime as ort
from bentoml.exceptions import NotFound

from .ranking import topk_unseen

if TYPE_CHECKING:
    import torch

    from ..data.types import PreparedData
    from ..model.base import BaseRecommender

MODEL_NAME = "deep_recsys"
ARTIFACT_SCHEMA_VERSION = "3.0"
SCORE_NOTE = "scores are uncalibrated ranking logits; do not interpret as probabilities"
ONNX_FILENAME = "model.onnx"
ONNX_INPUT_NAME = "interactions"
ONNX_OUTPUT_NAME = "logits"
ONNX_OPSET_VERSION = 20
INFERENCE_BACKEND = "onnxruntime"
EXECUTION_PROVIDER = "CPUExecutionProvider"
PAYLOAD_FILES = (ONNX_FILENAME, "item_ids.json", "movies.json", "metrics.json", "config.json")
PARITY_RTOL = 1e-4
PARITY_ATOL = 2e-5


class ModelArtifactError(ValueError):
    """Raised when a registered serving artifact is invalid or incompatible."""


class OnnxScorer:
    """Validated ONNX Runtime session with the recommender score contract."""

    def __init__(self, path: str | Path, *, n_items: int) -> None:
        self.path = Path(path)
        self.n_items = int(n_items)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 0
        try:
            self.session = ort.InferenceSession(
                self.path,
                sess_options=options,
                providers=[EXECUTION_PROVIDER],
            )
        except Exception as exc:
            raise ModelArtifactError(f"unable to initialize ONNX Runtime session: {exc}") from exc
        self._validate_contract()

    def _validate_contract(self) -> None:
        providers = self.session.get_providers()
        if providers != [EXECUTION_PROVIDER]:
            raise ModelArtifactError(
                f"ONNX execution providers are incompatible: expected {EXECUTION_PROVIDER}"
            )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ModelArtifactError("ONNX model must expose exactly one input and one output")
        model_input = inputs[0]
        model_output = outputs[0]
        expected_shape: list[str | int] = ["batch", self.n_items]
        if (
            model_input.name != ONNX_INPUT_NAME
            or model_input.type != "tensor(float)"
            or list(model_input.shape) != expected_shape
        ):
            raise ModelArtifactError("ONNX input contract is incompatible with the item catalog")
        if (
            model_output.name != ONNX_OUTPUT_NAME
            or model_output.type != "tensor(float)"
            or list(model_output.shape) != expected_shape
        ):
            raise ModelArtifactError("ONNX output contract is incompatible with the item catalog")

        expected_metadata = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "inference_backend": INFERENCE_BACKEND,
            "execution_provider": EXECUTION_PROVIDER,
            "onnx_opset": str(ONNX_OPSET_VERSION),
            "n_items": str(self.n_items),
            "input_name": ONNX_INPUT_NAME,
            "output_name": ONNX_OUTPUT_NAME,
        }
        metadata = self.session.get_modelmeta().custom_metadata_map
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ModelArtifactError("ONNX model metadata is missing or incompatible")

    def score(self, interactions: np.ndarray) -> np.ndarray:
        """Return finite float32 logits for a non-empty interaction batch."""

        values = np.asarray(interactions, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != self.n_items:
            raise ValueError(f"expected [batch, {self.n_items}] interactions")
        inputs = np.ascontiguousarray(values)
        try:
            result = self.session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: inputs})[0]
        except Exception as exc:
            raise ModelArtifactError(f"ONNX inference failed: {exc}") from exc
        scores = np.asarray(result, dtype=np.float32)
        if scores.shape != inputs.shape:
            raise ModelArtifactError("ONNX inference returned an incompatible output shape")
        if not bool(np.isfinite(scores).all()):
            raise ModelArtifactError("ONNX inference returned non-finite scores")
        return scores


@dataclass(frozen=True)
class LoadedModel:
    """Validated ONNX scorer and catalog state loaded from a Bento artifact."""

    root: Path
    model_tag: str
    scorer: OnnxScorer
    item_ids: np.ndarray
    metadata: dict[int, dict[str, Any]]
    manifest: dict[str, Any]
    metrics: dict[str, Any]
    config: dict[str, Any]

    @property
    def model_version(self) -> str:
        return str(self.manifest["model_version"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_mapping_hash(item_ids: list[int]) -> str:
    payload = json.dumps(item_ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    import torch

    checkpoint = Path(path)
    try:
        payload = cast(
            dict[str, Any], torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
    except (RuntimeError, pickle.UnpicklingError, TypeError) as exc:
        try:
            payload = cast(
                dict[str, Any], torch.load(checkpoint, map_location="cpu", weights_only=False)
            )
        except Exception as fallback_exc:  # pragma: no cover - defensive
            raise ModelArtifactError(
                f"unable to read checkpoint {checkpoint}: {fallback_exc}"
            ) from exc
    if isinstance(payload, dict):
        return payload
    return {"model_state": payload}


def _state_from_payload(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    import torch

    state = payload.get("model_state", payload.get("state_dict", payload))
    if not isinstance(state, dict) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ModelArtifactError("checkpoint does not contain a torch state dictionary")
    return cast(dict[str, torch.Tensor], state)


def _model_config_from_payload(
    payload: dict[str, Any],
    n_items: int,
    model_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if model_config:
        result = dict(model_config)
    elif isinstance(payload.get("model_config"), dict) and payload["model_config"]:
        result = dict(payload["model_config"])
    else:
        config = payload.get("config", {})
        result = dict(config.get("model", {})) if isinstance(config, dict) else {}
    result.setdefault("name", "multvae")
    result.setdefault("n_items", n_items)
    result["n_items"] = int(result["n_items"] or n_items)
    result.pop("_target_", None)
    return result


def _read_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(f"{description} is missing or invalid") from exc


def _validate_model_version(model_version: str) -> str:
    version = model_version.strip()
    if not version or version.lower() == "latest" or ":" in version:
        raise ModelArtifactError("model_version must be an explicit Bento version, not 'latest'")
    return version


def _runtime_manifest(n_items: int) -> dict[str, Any]:
    return {
        "format": "onnx",
        "inference_backend": INFERENCE_BACKEND,
        "execution_provider": EXECUTION_PROVIDER,
        "opset_version": ONNX_OPSET_VERSION,
        "input": {
            "name": ONNX_INPUT_NAME,
            "dtype": "float32",
            "shape": ["batch", n_items],
        },
        "output": {
            "name": ONNX_OUTPUT_NAME,
            "dtype": "float32",
            "shape": ["batch", n_items],
        },
    }


def _validate_artifact_files(
    root: str | Path, *, model_tag: str
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    artifact_root = Path(root)
    manifest = cast(
        dict[str, Any], _read_json(artifact_root / "artifact_manifest.json", "artifact manifest")
    )
    schema = manifest.get("artifact_schema_version")
    if schema != ARTIFACT_SCHEMA_VERSION:
        detail = (
            "; re-register the original checkpoint under a new immutable version"
            if schema == "2.0"
            else ""
        )
        raise ModelArtifactError(f"unsupported artifact schema: {schema}{detail}")
    if manifest.get("model_tag") != model_tag:
        raise ModelArtifactError("artifact model tag does not match the resolved Bento model")
    tag_name, separator, tag_version = model_tag.partition(":")
    if (
        tag_name != MODEL_NAME
        or separator != ":"
        or not tag_version
        or tag_version.lower() == "latest"
        or manifest.get("model_name") != MODEL_NAME
        or manifest.get("model_version") != tag_version
    ):
        raise ModelArtifactError("artifact identity is not an explicit matching model version")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(PAYLOAD_FILES):
        raise ModelArtifactError("artifact manifest does not list the expected payload files")
    for filename, expected in files.items():
        if not isinstance(expected, dict):
            raise ModelArtifactError(f"artifact checksum entry is invalid: {filename}")
        path = artifact_root / filename
        if not path.is_file():
            raise ModelArtifactError(f"artifact file is missing: {filename}")
        if _sha256(path) != expected.get("sha256"):
            raise ModelArtifactError(f"artifact checksum mismatch: {filename}")
        if path.stat().st_size != expected.get("size_bytes"):
            raise ModelArtifactError(f"artifact size mismatch: {filename}")

    item_ids_raw = _read_json(artifact_root / "item_ids.json", "item mapping")
    movies = _read_json(artifact_root / "movies.json", "movie metadata")
    config = _read_json(artifact_root / "config.json", "model configuration")
    metrics = _read_json(artifact_root / "metrics.json", "model metrics")
    if (
        not isinstance(item_ids_raw, list)
        or not item_ids_raw
        or any(isinstance(value, bool) or not isinstance(value, int) for value in item_ids_raw)
        or len(set(item_ids_raw)) != len(item_ids_raw)
    ):
        raise ModelArtifactError("item mapping is empty, non-integer, or not unique")
    item_ids = [int(value) for value in item_ids_raw]
    mapping = manifest.get("mapping", {})
    if not isinstance(mapping, dict) or mapping.get("count") != len(item_ids):
        raise ModelArtifactError("mapping size does not match the artifact manifest")
    if mapping.get("sha256") != _canonical_mapping_hash(item_ids):
        raise ModelArtifactError("mapping hash does not match the artifact manifest")
    if (
        not isinstance(movies, list)
        or len(movies) != len(item_ids)
        or any(not isinstance(row, dict) for row in movies)
    ):
        raise ModelArtifactError("movie metadata length does not match the item mapping")
    movie_rows = cast(list[dict[str, Any]], movies)
    try:
        metadata_ids = [int(row["movie_id"]) for row in movie_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelArtifactError("movie metadata rows do not contain valid movie IDs") from exc
    if metadata_ids != item_ids:
        raise ModelArtifactError("movie metadata order does not match the item mapping")
    if any(
        not isinstance(row.get("title"), str) or not isinstance(row.get("genres"), str)
        for row in movie_rows
    ):
        raise ModelArtifactError("movie metadata titles and genres must be strings")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ModelArtifactError("model configuration does not define an architecture")
    if not isinstance(metrics, dict):
        raise ModelArtifactError("model metrics must be a JSON object")

    config_dict = cast(dict[str, Any], config)
    metrics_dict = cast(dict[str, Any], metrics)
    model_config = cast(dict[str, Any], config_dict["model"])
    if model_config.get("n_items") != len(item_ids) or manifest.get("model") != model_config:
        raise ModelArtifactError("model architecture does not match the catalog or manifest")
    if manifest.get("runtime") != _runtime_manifest(len(item_ids)):
        raise ModelArtifactError("ONNX runtime manifest is missing or incompatible")
    exporter = manifest.get("exporter")
    expected_exporter_fields = {
        "framework",
        "torch_version",
        "onnx_version",
        "onnxscript_version",
        "onnxruntime_version",
    }
    if (
        not isinstance(exporter, dict)
        or set(exporter) != expected_exporter_fields
        or exporter.get("framework") != "pytorch"
        or any(not isinstance(value, str) or not value for value in exporter.values())
    ):
        raise ModelArtifactError("ONNX exporter metadata is missing or incompatible")
    return manifest, np.asarray(item_ids, dtype=np.int64), movie_rows, config_dict, metrics_dict


def _load_scorer(root: Path, n_items: int) -> OnnxScorer:
    scorer = OnnxScorer(root / ONNX_FILENAME, n_items=n_items)
    scorer.score(np.zeros((1, n_items), dtype=np.float32))
    return scorer


def validate_model_artifact(root: str | Path, *, model_tag: str) -> dict[str, Any]:
    """Validate schema-v3 checksums, catalog alignment, and the ONNX session."""

    manifest, item_ids, _, _, _ = _validate_artifact_files(root, model_tag=model_tag)
    _load_scorer(Path(root), len(item_ids))
    return manifest


def load_model_artifact(root: str | Path, *, model_tag: str) -> LoadedModel:
    """Load a checksum-validated ONNX Runtime CPU artifact."""

    artifact_root = Path(root)
    manifest, item_ids, movies, config, metrics = _validate_artifact_files(
        artifact_root, model_tag=model_tag
    )
    scorer = _load_scorer(artifact_root, len(item_ids))
    metadata = {int(row["movie_id"]): row for row in movies}
    return LoadedModel(
        root=artifact_root,
        model_tag=model_tag,
        scorer=scorer,
        item_ids=item_ids,
        metadata=metadata,
        manifest=manifest,
        metrics=metrics,
        config=config,
    )


def _export_onnx_model(model: BaseRecommender, path: Path) -> None:
    import onnx
    import torch

    class LogitsWrapper(torch.nn.Module):
        def __init__(self, recommender: BaseRecommender) -> None:
            super().__init__()
            self.recommender = recommender

        def forward(self, interactions: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.recommender(interactions, sample=False).logits)

    wrapper = LogitsWrapper(model).eval()
    example = torch.zeros((1, model.n_items), dtype=torch.float32)
    try:
        torch.onnx.export(
            wrapper,
            (example,),
            path,
            input_names=[ONNX_INPUT_NAME],
            output_names=[ONNX_OUTPUT_NAME],
            opset_version=ONNX_OPSET_VERSION,
            dynamo=True,
            dynamic_shapes={
                ONNX_INPUT_NAME: {0: torch.export.Dim("batch", min=1)},
            },
            external_data=False,
            optimize=True,
        )
        onnx_model = onnx.load(path, load_external_data=False)
        opsets = {entry.domain: entry.version for entry in onnx_model.opset_import}
        if opsets.get("") != ONNX_OPSET_VERSION:
            raise ModelArtifactError("exported ONNX model uses an unexpected opset")
        onnx.helper.set_model_props(
            onnx_model,
            {
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "inference_backend": INFERENCE_BACKEND,
                "execution_provider": EXECUTION_PROVIDER,
                "onnx_opset": str(ONNX_OPSET_VERSION),
                "n_items": str(model.n_items),
                "input_name": ONNX_INPUT_NAME,
                "output_name": ONNX_OUTPUT_NAME,
            },
        )
        onnx.checker.check_model(onnx_model, full_check=True)
        onnx.save_model(onnx_model, path, save_as_external_data=False)
    except ModelArtifactError:
        raise
    except Exception as exc:
        raise ModelArtifactError(f"unable to export a valid ONNX model: {exc}") from exc


def _parity_inputs(data: PreparedData) -> np.ndarray:
    rows = [np.zeros((1, data.n_items), dtype=np.float32)]
    seed_history = np.zeros((1, data.n_items), dtype=np.float32)
    seed_history[0, : min(5, data.n_items)] = 1.0
    rows.append(seed_history)
    count = min(6, data.train.shape[0])
    if count:
        rows.append(data.train[:count].toarray().astype(np.float32, copy=False))
    return np.ascontiguousarray(np.concatenate(rows, axis=0))


def _validate_export_parity(
    model: BaseRecommender,
    scorer: OnnxScorer,
    data: PreparedData,
) -> None:
    import torch

    interactions = _parity_inputs(data)
    tensor = torch.from_numpy(interactions)
    with torch.inference_mode():
        expected = model(tensor, sample=False).logits.detach().cpu().numpy()
    actual = scorer.score(interactions)
    if expected.shape != actual.shape or not bool(np.isfinite(expected).all()):
        raise ModelArtifactError("PyTorch parity reference is invalid")
    if not np.allclose(expected, actual, rtol=PARITY_RTOL, atol=PARITY_ATOL):
        max_error = float(np.max(np.abs(expected - actual)))
        raise ModelArtifactError(
            f"ONNX numerical parity failed: maximum absolute error={max_error:.8g}"
        )

    for row_index, row in enumerate(interactions):
        available = int((~row.astype(bool)).sum())
        if available == 0:
            continue
        k = min(100, available)
        expected_indices, _ = topk_unseen(expected[row_index : row_index + 1], row[None, :], k)
        actual_indices, _ = topk_unseen(actual[row_index : row_index + 1], row[None, :], k)
        if not np.array_equal(expected_indices, actual_indices):
            raise ModelArtifactError("ONNX unseen top-k ordering does not match PyTorch")


def register_bento_model(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    *,
    model_version: str,
    model_config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    extra_config: dict[str, Any] | None = None,
) -> bentoml.Model:
    """Export and register an immutable schema-v3 ONNX Bento model."""

    import onnx
    import onnxscript
    import torch

    from ..data.dataset import load_prepared_data
    from ..model import build_model

    version = _validate_model_version(model_version)
    tag = f"{MODEL_NAME}:{version}"
    try:
        bentoml.models.get(tag)
    except NotFound:
        pass
    else:
        raise ModelArtifactError(f"Bento model already exists: {tag}")

    data = load_prepared_data(data_dir)
    payload = _load_checkpoint_payload(checkpoint_path)
    state = _state_from_payload(payload)
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise ModelArtifactError("checkpoint contains non-finite model weights")
    architecture = _model_config_from_payload(payload, data.n_items, model_config)
    if int(architecture["n_items"]) != data.n_items:
        raise ModelArtifactError("checkpoint item dimension does not match prepared data")
    constructor = dict(architecture)
    model_name = str(constructor.pop("name", "multvae"))
    model = build_model(model_name, **constructor)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise ModelArtifactError(
            f"checkpoint shapes do not match model architecture: {exc}"
        ) from exc
    model.eval()

    item_ids = [int(value) for value in data.item_ids.tolist()]
    metric_payload = metrics if metrics is not None else payload.get("metrics", {}) or {}
    if not isinstance(metric_payload, dict):
        raise ModelArtifactError("model metrics must be a JSON object")
    config_payload: dict[str, Any] = {
        "model": model.architecture_config(),
        "source_checkpoint": Path(checkpoint_path).name,
        "registration": extra_config or {},
    }
    try:
        json.dumps(metric_payload, allow_nan=False)
        json.dumps(config_payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ModelArtifactError("metrics and configuration must be finite JSON values") from exc
    mapping_hash = _canonical_mapping_hash(item_ids)
    model_metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": version,
        "item_count": len(item_ids),
        "mapping_sha256": mapping_hash,
        "score_note": SCORE_NOTE,
        "inference_backend": INFERENCE_BACKEND,
        "execution_provider": EXECUTION_PROVIDER,
        "onnx_opset": ONNX_OPSET_VERSION,
    }
    with bentoml.models.create(
        tag,
        labels={"framework": "onnx", "model_type": model_name},
        metadata=model_metadata,
    ) as model_ref:
        root = Path(model_ref.path)
        _export_onnx_model(model, root / ONNX_FILENAME)
        scorer = OnnxScorer(root / ONNX_FILENAME, n_items=data.n_items)
        _validate_export_parity(model, scorer, data)
        (root / "item_ids.json").write_text(json.dumps(item_ids, indent=2), encoding="utf-8")
        (root / "movies.json").write_text(
            json.dumps(data.movies, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (root / "metrics.json").write_text(
            json.dumps(metric_payload, sort_keys=True, indent=2), encoding="utf-8"
        )
        (root / "config.json").write_text(
            json.dumps(config_payload, sort_keys=True, indent=2), encoding="utf-8"
        )
        checksums = {
            filename: {
                "sha256": _sha256(root / filename),
                "size_bytes": (root / filename).stat().st_size,
            }
            for filename in PAYLOAD_FILES
        }
        manifest = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_name": MODEL_NAME,
            "model_version": version,
            "model_tag": str(model_ref.tag),
            "created_by": "deep-recsys-lab",
            "python": sys.version,
            "platform": platform.platform(),
            "exporter": {
                "framework": "pytorch",
                "torch_version": torch.__version__,
                "onnx_version": onnx.__version__,
                "onnxscript_version": onnxscript.__version__,
                "onnxruntime_version": ort.__version__,
            },
            "runtime": _runtime_manifest(data.n_items),
            "model": model.architecture_config(),
            "mapping": {"count": len(item_ids), "sha256": mapping_hash},
            "files": checksums,
            "metrics_file": "metrics.json",
            "uncalibrated_score_note": SCORE_NOTE,
        }
        (root / "artifact_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
        )
        validate_model_artifact(root, model_tag=str(model_ref.tag))
    return bentoml.models.get(tag)
