# MovieLens 20M Recommendation Models

The content encoder + Mult-VAE benchmark compares BPR and InfoNCE across
serialized/multiview embeddings and projection/LoRA training (48 runs). Projection
runs on the Mac now (24 runs); LoRA is deferred to a future GPU stage (24 runs):

```bash
uv sync --extra content
uv run --extra content python -m content_recsys search --stage projection
```

See [the benchmark guide](docs/content_multvae.md) for resume, smoke verification,
validation selection, the deferred GPU specification and English reports. Test
evaluation waits for both stages. This benchmark does not run unit tests.

This project prepares implicit MovieLens 20M interactions and trains LightGCN and
Mult-VAE in PyTorch, following [`paper/lightGCN.pdf`](paper/lightGCN.pdf) and
[`paper/MultVAE.pdf`](paper/MultVAE.pdf).

```bash
uv sync
uv run python -m lightgcn.train --batch-size 524288 --max-epochs 30 --eval-every 2 --patience 5 --validation-users 20000
```

The prepared data must exist in `data/processed/ml20m_lightgcn/`. The current selected weights are in `artifacts/lightgcn_ml20m/best.pt`, with configuration, history, and final metrics beside them. The command above trains the original baseline settings and would replace that directory; pass `--output-dir` for a new run.

See [`docs/data_preparation.md`](docs/data_preparation.md) for the dataset split, [`docs/lightgcn.md`](docs/lightgcn.md) for the model and training protocol, and [`docs/lightgcn_search.md`](docs/lightgcn_search.md) for the completed architecture search and reproduction command.

To retrain the selected architecture on train + validation, compare scratch
training with warmup/fine-tuning, and report full-test metrics:

```bash
uv run python -m lightgcn.retrain --device mps
```

See [`docs/lightgcn_retraining.md`](docs/lightgcn_retraining.md) for the protocols
and [`artifacts/lightgcn_retrain/report.md`](artifacts/lightgcn_retrain/report.md)
for the measured comparison. Focused checks use the built-in test runner:
`uv run python -m unittest discover -s tests -v`.

## TMDB credentials

Store `TMDB_READ_ACCESS_TOKEN` and `TMDB_API_KEY` in the root `.env` file,
which is excluded from Git. `.env.example` contains the variable names without
credentials. Keep the read access token as the raw value, without a `Bearer ` prefix.

From the project root, export the environment-file path once in your terminal:

```bash
export UV_ENV_FILE="$PWD/.env"
uv run python
```

Subsequent `uv run` commands in that same terminal automatically load the file.
The export applies only to that shell session and its child processes. Repeat it
from the project root in each new terminal; an export performed by an assistant's
command process does not update an already-open user terminal.

Python code can read the credentials using
`os.environ["TMDB_READ_ACCESS_TOKEN"]` and `os.environ["TMDB_API_KEY"]`.
The movie metadata crawler also reads the root `.env` directly when the token is
not already exported. To collect English TMDB metadata and fill missing content
from Wikidata/Wikipedia:

```bash
uv run python scripts/crawl_movie_metadata.py --limit 30
uv run python scripts/crawl_movie_metadata.py
```

See [`docs/movie_content.md`](docs/movie_content.md) for cache/resume options,
output schemas, coverage checks, and source attribution. No additional dependency
is needed.

For a single command without exporting, pass the file explicitly:

```bash
uv run --env-file .env python
```

Python does not load `.env` automatically when run directly. The commands above
use uv's built-in environment-file support and require no additional dependency.

## Mult-VAE

The Mult-VAE PR baseline from `paper/MultVAE.pdf` uses the same prepared splits,
Apple MPS, user batches of 500, beta annealing capped at 0.2, and up to 200 epochs
with early stopping. It reports Recall/NDCG at 10, 20, 50 and 100 using this
repository's metric definitions. See [`docs/multvae.md`](docs/multvae.md) for the
architecture, evaluation input, stopping/resume protocol and English report.

```bash
uv run python -m multvae.train --device mps --max-epochs 200 --patience 10
```

Mult-VAE outputs go to `artifacts/multvae_ml20m/baseline_mps_seed42/`, including
checkpoints, exact configuration, epoch logs, final metrics and `report.md` with PNG
figures. A new run requires an empty output directory; use `--resume` to continue
the last completed epoch.
