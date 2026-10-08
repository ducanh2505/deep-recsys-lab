#!/usr/bin/env bash
# After cloning the source on Kaggle, run: bash scripts/kaggle_prepare.sh
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE_SHA256="96f243c338a8665f6bcc89c53edf6ee39162a846940de6b7c8c48aeada765ff3"
CATALOG_INPUT="${CATALOG_INPUT:-/kaggle/input/datasets/danh99/catalog-movielen-20m/catalog.parquet}"
BASELINE_INPUT="${BASELINE_INPUT:-}"
KAGGLE_INPUT_DIR="${KAGGLE_INPUT_DIR:-/kaggle/input}"

cd "$REPO_DIR"

# Validate both manual inputs before preparing data or downloading dependencies.
python3 - "$CATALOG_INPUT" "$BASELINE_INPUT" "$KAGGLE_INPUT_DIR" <<'PY'
import hashlib
from pathlib import Path
import shutil
import sys

inputs = (
    (("catalog.parquet", "CBF-LORA", "CBF-LORA.parquet"), sys.argv[1], "CATALOG_INPUT",
     "c3f0151964ab04310b62c96547fddc0a5603df76d741cbb4b30e2c444de931a4",
     Path("data/processed/movie_content/catalog.parquet")),
    (("best.pt",), sys.argv[2], "BASELINE_INPUT",
     "0732c4fbf21ed5259419dfbcb5cc4d56322e5b38fedb97f3f0ce99412440def2",
     Path("artifacts/multvae_ml20m/baseline_mps_seed42/best.pt")),
)
copies = []
for names, override, variable, expected, target in inputs:
    name = " / ".join(names)
    accepted = {value.casefold() for value in names}
    search_root = Path(sys.argv[3])
    candidates = [Path(override)] if override else sorted(
        path for path in search_root.rglob("*")
        if path.is_file() and path.name.casefold() in accepted
    )
    if len(candidates) != 1 or not candidates[0].is_file():
        raise SystemExit(f"Attach one {name}, or set {variable} to its exact path.")
    source = candidates[0]
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise SystemExit(f"FAIL: uploaded {name} differs from the local input.")
    copies.append((source, target))
for source, target in copies:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    print("PASS: input matches local data. Source:", source)
PY

python3 -m pip install --upgrade uv
python3 -m uv sync --python 3.11 --frozen --extra content
uv_python() {
    python3 -m uv run --python 3.11 --frozen --extra content python "$@"
}
uv_python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Enable a Kaggle GPU and verify CUDA-enabled PyTorch before continuing.")
torch.ones(1, device="cuda")
torch.cuda.synchronize()
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))
PY

mkdir -p data/raw
ARCHIVE="data/raw/ml-20m.zip"
if [[ ! -f "$ARCHIVE" ]]; then
    curl --fail --location --retry 3 \
        https://files.grouplens.org/datasets/movielens/ml-20m.zip \
        --output "$ARCHIVE.tmp"
    printf '%s  %s\n' "$ARCHIVE_SHA256" "$ARCHIVE.tmp" | sha256sum --check -
    mv "$ARCHIVE.tmp" "$ARCHIVE"
fi
printf '%s  %s\n' "$ARCHIVE_SHA256" "$ARCHIVE" | sha256sum --check -
unzip -o "$ARCHIVE" ml-20m/ratings.csv ml-20m/movies.csv ml-20m/links.csv -d data/raw

# Invoke the local pipeline unchanged, including seed 42 and the 10-core rules.
uv_python scripts/prepare_ml20m.py
uv_python - <<'PY'
import json
from pathlib import Path

expected = {
    "train": "b22cc091596505693478891561a1bae729ead19981a0806690f6e150c401d10a",
    "valid": "d57d156866bc89f138b1a2c0b7f8c0e6c619dde37bfb81f89582e26484a83d37",
    "test": "527630ea5c60f7ec715bd73c98eee25844f2e74aba68c1f2bd554d22df36d225",
}
output = Path("data/processed/ml20m_lightgcn")
manifest = json.loads((output / "manifest.json").read_text())
if manifest["canonical_sha256"] != expected:
    raise SystemExit("FAIL: splits differ from local data; do not start training.")
print("PASS: train/valid/test content matches the local splits.")
print("Output:", output.resolve())
PY

# Build token/context caches and frozen Qwen embeddings with the same local pipeline.
uv_python -m content_recsys prepare --device cuda \
    --data-dir data/processed/ml20m_lightgcn \
    --catalog data/processed/movie_content/catalog.parquet \
    --baseline artifacts/multvae_ml20m/baseline_mps_seed42/best.pt \
    --output-dir artifacts/content_multvae_ml20m
printf 'Prepared data ready: %s/artifacts/content_multvae_ml20m/prepared\n' "$REPO_DIR"
