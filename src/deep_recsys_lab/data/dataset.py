"""PyTorch datasets for dense batches from sparse interaction matrices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix, load_npz
from torch.utils.data import Dataset

from .types import MATRIX_FILES, PreparedData


def _as_csr(matrix: csr_matrix | np.ndarray) -> csr_matrix:
    if isinstance(matrix, csr_matrix):
        return matrix.tocsr().astype(np.float32)
    return csr_matrix(np.asarray(matrix, dtype=np.float32))


class InteractionDataset(Dataset[dict[str, torch.Tensor]]):
    """Return dense interaction rows, with deterministic evaluation holdouts.

    When ``eval=False`` the complete input row is returned. When ``eval=True``
    each row is split once, at construction time, into fold-in data and
    fold-out ground truth. This avoids sampling changes between epochs and
    makes a repeated evaluation bit-for-bit reproducible.
    """

    def __init__(
        self,
        matrix: csr_matrix | np.ndarray,
        *,
        eval: bool = False,
        prop: float = 0.2,
        seed: int = 98_765,
        fold_in: csr_matrix | np.ndarray | None = None,
        fold_out: csr_matrix | np.ndarray | None = None,
    ) -> None:
        self.matrix = _as_csr(matrix)
        self.eval = eval
        if not 0 <= prop < 1:
            raise ValueError("prop must be in [0, 1)")
        self.prop = float(prop)
        self.seed = int(seed)
        if (fold_in is None) != (fold_out is None):
            raise ValueError("fold_in and fold_out must be provided together")
        if fold_in is not None and fold_out is not None:
            self._fold_in = _as_csr(fold_in)
            self._fold_out = _as_csr(fold_out)
            if (
                self._fold_in.shape != self.matrix.shape
                or self._fold_out.shape != self.matrix.shape
            ):
                raise ValueError("fold_in and fold_out must match the input matrix shape")
        elif eval:
            self._fold_in, self._fold_out = self._split_rows(self.matrix, prop, seed)
        else:
            self._fold_in = self.matrix
            self._fold_out = csr_matrix(self.matrix.shape, dtype=np.float32)

    @staticmethod
    def _split_rows(matrix: csr_matrix, prop: float, seed: int) -> tuple[csr_matrix, csr_matrix]:
        fold_in = matrix.copy().tolil()
        fold_out = csr_matrix(matrix.shape, dtype=np.float32).tolil()
        for row_idx in range(matrix.shape[0]):
            start, end = matrix.indptr[row_idx : row_idx + 2]
            indices = matrix.indices[start:end]
            if len(indices) < 2 or prop == 0:
                continue
            n_out = min(len(indices) - 1, max(1, int(np.floor(len(indices) * prop))))
            rng = np.random.default_rng(seed + row_idx * 1_000_003)
            selected = np.sort(rng.choice(len(indices), size=n_out, replace=False))
            out_items = indices[selected]
            fold_in[row_idx, out_items] = 0
            fold_out[row_idx, out_items] = 1
        return fold_in.tocsr(), fold_out.tocsr()

    @property
    def fold_in(self) -> csr_matrix:
        return self._fold_in

    @property
    def fold_out(self) -> csr_matrix:
        return self._fold_out

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = torch.from_numpy(self._fold_in.getrow(index).toarray().ravel().astype(np.float32))
        truth = torch.from_numpy(self._fold_out.getrow(index).toarray().ravel().astype(np.float32))
        return {"data": data, "ground_truth": truth}


def load_prepared_data(root: str | Path) -> PreparedData:
    """Load and minimally validate a versioned processed-data directory."""

    root_path = Path(root)
    manifest_path = root_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"prepared-data manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError(f"unsupported prepared-data schema: {manifest.get('schema_version')}")
    matrices = {
        name: load_npz(root_path / filename).tocsr().astype(np.float32)
        for name, filename in MATRIX_FILES.items()
    }
    n_items = matrices["train"].shape[1]
    expected_files = set(MATRIX_FILES.values()) | {"item_ids.npy", "movies.json"}
    recorded_files = manifest.get("files", {})
    if set(recorded_files) != expected_files:
        raise ValueError("prepared-data manifest does not list the expected files")
    import hashlib

    for filename, expected in recorded_files.items():
        digest = hashlib.sha256((root_path / filename).read_bytes()).hexdigest()
        if digest != expected.get("sha256"):
            raise ValueError(f"prepared-data checksum mismatch: {filename}")
    if any(matrix.shape[1] != n_items for matrix in matrices.values()):
        raise ValueError("all prepared matrices must have the same item dimension")
    item_ids = np.load(root_path / "item_ids.npy", allow_pickle=False)
    if len(item_ids) != n_items or len(np.unique(item_ids)) != len(item_ids):
        raise ValueError("item mapping length or uniqueness is invalid")
    movies_path = root_path / "movies.json"
    movies = json.loads(movies_path.read_text(encoding="utf-8")) if movies_path.exists() else []
    return PreparedData(
        train=matrices["train"],
        validation_fold_in=matrices["validation_fold_in"],
        validation_fold_out=matrices["validation_fold_out"],
        test_fold_in=matrices["test_fold_in"],
        test_fold_out=matrices["test_fold_out"],
        item_ids=item_ids,
        movies=movies,
        manifest=manifest,
        root=root_path,
    )
