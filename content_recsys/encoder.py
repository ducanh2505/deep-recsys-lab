"""Frozen Qwen features and trainable projection/LoRA movie encoders."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from content_recsys.common import MODEL_ID, REVISION, emit, exclusive_lock, sha256, write_json
from content_recsys.data import BLOCK_WEIGHTS, load_prepared


class TextEncoder(nn.Module):
    """Encode cached token IDs with left padding and last-token pooling."""

    def __init__(self, device: torch.device, lora: bool = False) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, padding_side="left")
        self.backbone = AutoModel.from_pretrained(MODEL_ID, revision=REVISION,
                                                  dtype=torch.float32, attn_implementation="sdpa")
        self.backbone.config.use_cache = False
        self.backbone.requires_grad_(False)
        if lora:
            config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16,
                                lora_dropout=0.0, target_modules=["q_proj", "v_proj"])
            self.backbone = get_peft_model(self.backbone, config)
            self.backbone.enable_input_require_grads()
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.to(device)

    def forward(self, sequences: list[torch.Tensor]) -> torch.Tensor:
        """Pool and normalize a nonempty batch of token sequences."""
        device = next(self.parameters()).device
        batch = self.tokenizer.pad({"input_ids": [tokens.tolist() for tokens in sequences]},
                                   padding=True, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        hidden = self.backbone(**batch).last_hidden_state
        result = F.normalize(hidden[:, -1, :], dim=1)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Non-finite Qwen embedding")
        return result


def ensure_frozen(prepared: Path, device: torch.device, batch_size: int = 8) -> dict:
    """Lock and resume frozen feature preparation."""
    with exclusive_lock(prepared / "frozen" / ".encoding.lock"):
        return _ensure_frozen(prepared, device, batch_size)


def _ensure_frozen(prepared: Path, device: torch.device, batch_size: int = 8) -> dict:
    """Resume catalog encoding at the last completed batch of each text view."""
    started = time.monotonic()
    emit(prepared, "frozen_preparation", stage="verify_prepared_inputs", device=str(device))
    data, _, tokens, manifest = load_prepared(prepared)
    cache = prepared / "frozen"
    cache.mkdir(exist_ok=True)
    result_path = cache / "manifest.json"
    if result_path.exists():
        emit(cache, "frozen_preparation", stage="verify_cached_embeddings")
        result = json.loads(result_path.read_text())
        if result["prepared_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Frozen features do not match prepared inputs")
        for name, digest in result["artifacts"].items():
            if sha256(cache / name) != digest:
                raise ValueError(f"Corrupt frozen cache: {name}")
        emit(cache, "frozen_preparation", stage="complete", cached=True,
             elapsed_seconds=time.monotonic() - started)
        return result
    emit(cache, "frozen_preparation", stage="load_model", model=MODEL_ID, revision=REVISION,
         device=str(device))
    encoder = TextEncoder(device)
    encoder.eval()
    emit(cache, "frozen_preparation", stage="model_loaded", elapsed_seconds=time.monotonic() - started)
    for view, sequences in tokens.items():
        filename = cache / f"{view}.npy"
        progress_path = cache / f"{view}_progress.json"
        order = sorted((i for i, ids in enumerate(sequences) if len(ids)), key=lambda i: len(sequences[i]))
        done = json.loads(progress_path.read_text())["completed"] if progress_path.exists() else 0
        if done and not filename.exists():
            raise ValueError("Encoding progress exists without the vector cache")
        if filename.exists():
            values = np.lib.format.open_memmap(filename, mode="r+")
            if values.shape != (data.n_items, 1024) or values.dtype != np.float32:
                raise ValueError("Frozen cache has an incompatible shape or dtype")
        else:
            values = np.lib.format.open_memmap(filename, mode="w+", dtype=np.float32, shape=(data.n_items, 1024))
            values[:] = 0
        view_started = time.monotonic()
        last_report = view_started
        emit(cache, "frozen_encoding", view=view, completed=done, total=len(order),
             percent=100.0 * done / len(order) if order else 100.0,
             resumed=done > 0, elapsed_seconds=time.monotonic() - started)
        with torch.inference_mode():
            for offset in range(done, len(order), batch_size):
                indices = order[offset:offset + batch_size]
                values[indices] = encoder([sequences[i] for i in indices]).cpu().numpy()
                values.flush()
                write_json(progress_path, {"completed": offset + len(indices), "total": len(order)})
                completed = offset + len(indices)
                now = time.monotonic()
                if offset == done or now - last_report >= 10 or completed == len(order):
                    rate = (completed - done) / max(now - view_started, 1e-9)
                    emit(cache, "frozen_encoding", view=view, completed=completed,
                         total=len(order), percent=100.0 * completed / len(order),
                         movies_per_second=rate, eta_seconds=(len(order) - completed) / rate,
                         elapsed_seconds=now - started)
                    last_report = now
        del values
    emit(cache, "frozen_preparation", stage="combine_multiview_embeddings")
    context = torch.from_numpy(np.load(prepared / "context.npy"))
    pieces = [torch.from_numpy(np.load(cache / f"{view}.npy")) for view in ("plot", "topic", "people")]
    pieces.append(F.normalize(context, dim=1))
    multi = F.normalize(torch.cat([value * weight**0.5 for value, weight in zip(pieces, BLOCK_WEIGHTS)], dim=1), dim=1)
    np.save(cache / "multi.npy", multi.numpy())
    emit(cache, "frozen_preparation", stage="hash_embeddings")
    result = {"prepared_fingerprint": manifest["fingerprint"], "model": MODEL_ID, "revision": REVISION,
              "elapsed_seconds": time.monotonic() - started,
              "artifacts": {f"{view}.npy": sha256(cache / f"{view}.npy")
                            for view in ("single", "plot", "topic", "people", "multi")}}
    write_json(result_path, result)
    emit(cache, "frozen_preparation", stage="complete", cached=False,
         elapsed_seconds=time.monotonic() - started)
    return result


class ContentModel(nn.Module):
    """Produce normalized 512-dimensional movie vectors from either layout."""

    def __init__(self, prepared: Path, layout: str, mode: str, device: torch.device,
                 text_microbatch: int = 4) -> None:
        super().__init__()
        _, _, self.tokens, _ = load_prepared(prepared)
        self.layout, self.mode, self.device = layout, mode, device
        self.text_microbatch = text_microbatch
        self.raw = torch.from_numpy(np.load(prepared / "frozen" / f"{layout}.npy"))
        self.context = F.normalize(torch.from_numpy(np.load(prepared / "context.npy")), dim=1)
        self.projection = nn.Sequential(nn.Linear(self.raw.shape[1], 512), nn.GELU(), nn.Linear(512, 512)).to(device)
        self.text = TextEncoder(device, lora=True) if mode == "lora" else None

    def features(self, indices: np.ndarray) -> torch.Tensor:
        """Recompute candidate text features only when adapters are trainable."""
        if self.text is None:
            return self.raw[indices].to(self.device)
        blocks = []
        for view in (("single",) if self.layout == "single" else ("plot", "topic", "people")):
            sequences = self.tokens[view]
            present = [k for k, index in enumerate(indices) if len(sequences[int(index)])]
            block = torch.zeros((len(indices), 1024), device=self.device)
            for start in range(0, len(present), self.text_microbatch):
                positions = present[start:start + self.text_microbatch]
                value = self.text([sequences[int(indices[k])] for k in positions])
                block = block.index_copy(0, torch.tensor(positions, device=self.device), value)
            blocks.append(block)
        if self.layout == "single":
            return blocks[0]
        blocks.append(self.context[indices].to(self.device))
        return F.normalize(torch.cat([block * weight**0.5 for block, weight in zip(blocks, BLOCK_WEIGHTS)], dim=1), dim=1)

    def forward(self, indices: np.ndarray) -> torch.Tensor:
        """Return learned movie embeddings for candidate item indices."""
        return F.normalize(self.projection(self.features(indices)), dim=1)

    @torch.no_grad()
    def catalog(self, batch_size: int = 8, initial: bool = False) -> torch.Tensor:
        """Export the full catalog, optionally reusing unchanged initial Qwen features."""
        self.eval()
        results = []
        for start in range(0, len(self.raw), batch_size):
            indices = np.arange(start, min(start + batch_size, len(self.raw)))
            if initial or self.text is None:
                value = F.normalize(self.projection(self.raw[indices].to(self.device)), dim=1)
            else:
                value = self(indices)
            results.append(value.cpu())
        return torch.cat(results)

    def learned_state(self) -> dict[str, torch.Tensor]:
        """Save only projection and adapter parameters, never the frozen backbone."""
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items() if name in trainable}

    def restore_learned(self, state: dict) -> None:
        """Reject incomplete or incompatible learned parameter checkpoints."""
        expected = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        if expected != set(state):
            raise ValueError("Projection/adapter parameter mapping differs from checkpoint")
        self.load_state_dict(state, strict=False)
