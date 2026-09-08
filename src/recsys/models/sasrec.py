"""Self-attentive sequential recommendation plugin."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import nn

from recsys.artifacts import RuntimeSpec
from recsys.core.types import QueryMode
from recsys.datasets import ordered_train_item_indices

from .base import ModelPlugin, TrainingContext


class CausalAttentionBlock(nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.dimension = dimension
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)
        self.attention_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.GELU(),
            nn.Linear(dimension * 2, dimension),
        )
        self.feed_forward_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        query = self.query(value)
        key = self.key(value)
        content = self.value(value)
        scores = query @ key.transpose(1, 2) / self.dimension**0.5
        length = value.shape[1]
        causal = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=value.device), diagonal=1
        )
        scores = scores.masked_fill(causal, -10_000.0)
        scores = scores.masked_fill(padding.unsqueeze(1), -10_000.0)
        attended = torch.softmax(scores, dim=-1) @ content
        value = self.attention_norm(value + self.dropout(self.output(attended)))
        value = self.feed_forward_norm(value + self.dropout(self.feed_forward(value)))
        return value.masked_fill(padding.unsqueeze(-1), 0.0)


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        max_length: int,
        dimension: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if max_length < 1 or dimension < 1 or layers < 1:
            raise ValueError("SASRec dimensions and layers must be positive")
        self.n_items = n_items
        self.max_length = max_length
        self.items = nn.Embedding(n_items + 1, dimension, padding_idx=0)
        self.positions = nn.Embedding(max_length, dimension)
        self.blocks = nn.ModuleList(CausalAttentionBlock(dimension, dropout) for _ in range(layers))
        nn.init.normal_(self.items.weight, std=0.02)
        nn.init.normal_(self.positions.weight, std=0.02)
        with torch.no_grad():
            self.items.weight[0].zero_()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        tracing = torch.jit.is_tracing()  # type: ignore[attr-defined,no-untyped-call]
        if not tracing and (sequence.ndim != 2 or sequence.shape[1] != self.max_length):
            raise ValueError(f"expected [batch, {self.max_length}] sequences")
        padding = sequence.eq(0)
        position = torch.arange(self.max_length, device=sequence.device).unsqueeze(0)
        value = self.items(sequence) + self.positions(position)
        value = value.masked_fill(padding.unsqueeze(-1), 0.0)
        for block in self.blocks:
            value = block(value, padding)
        return cast(torch.Tensor, value[:, -1] @ self.items.weight[1:].T)


def _sequences(
    context: TrainingContext, max_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    known = np.zeros((context.data.shape[0], max_length), dtype=np.int64)
    examples: list[np.ndarray] = []
    targets: list[int] = []
    for user_index, values in enumerate(ordered_train_item_indices(context.data)):
        encoded = [item + 1 for item in values]
        known[user_index, -min(len(encoded), max_length) :] = encoded[-max_length:]
        for stop in range(1, len(encoded)):
            row = np.zeros(max_length, dtype=np.int64)
            prefix = encoded[max(0, stop - max_length) : stop]
            row[-len(prefix) :] = prefix
            examples.append(row)
            targets.append(encoded[stop] - 1)
    if not examples:
        raise ValueError("SASRec training requires at least one ordered item transition")
    return known, np.stack(examples), np.asarray(targets, dtype=np.int64)


class SASRecPlugin(ModelPlugin):
    name = "sasrec"

    def train(self, context: TrainingContext) -> RuntimeSpec:
        torch.manual_seed(context.training.seed)
        parameters = context.parameters
        max_length = int(parameters.get("max_length", 50))
        model = SASRec(
            context.data.shape[1],
            max_length,
            int(parameters.get("embedding_dim", 32)),
            int(parameters.get("layers", 2)),
            float(parameters.get("dropout", 0.1)),
        )
        known, examples, targets = _sequences(context, max_length)
        device = torch.device(context.training.device)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=context.training.learning_rate)
        rng = np.random.default_rng(context.training.seed)
        batch_size = max(1, min(context.training.batch_size, len(examples)))
        model.train()
        for _ in range(max(1, context.training.epochs)):
            order = rng.permutation(len(examples))
            for start in range(0, len(order), batch_size):
                selected = order[start : start + batch_size]
                sequence = torch.from_numpy(examples[selected]).to(device)
                target = torch.from_numpy(targets[selected]).to(device)
                loss = nn.functional.cross_entropy(model(sequence), target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
        model.eval().cpu()
        context.scratch.mkdir(parents=True, exist_ok=True)
        onnx_path = context.scratch / "model.onnx"
        example = torch.zeros((1, max_length), dtype=torch.int64)
        example[0, -1] = 1
        torch.onnx.export(
            model,
            (example,),
            onnx_path,
            input_names=["sequence"],
            output_names=["scores"],
            dynamic_axes={"sequence": {0: "batch"}, "scores": {0: "batch"}},
            opset_version=20,
            dynamo=False,
        )
        return RuntimeSpec(
            plugin=self.name,
            runtime="sequential_onnx",
            capabilities=(QueryMode.KNOWN_USER, QueryMode.HISTORY),
            arrays={"known_sequences": known},
            files={"model.onnx": onnx_path},
            metadata={"input": "sequence", "output": "scores", "max_length": max_length},
        )
