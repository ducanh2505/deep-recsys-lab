# deep-recsys-lab

`deep-recsys-lab` is a local, reproducible implementation of Multi-VAE for
implicit MovieLens-20M collaborative filtering. It contains the data pipeline,
paper model, deterministic ranking evaluation, immutable ONNX model
registration, and a stateless ONNX Runtime/BentoML recommendation Service.

This repository does not contain MovieLens data, checkpoints, or a code
license. Downloaded and processed artifacts are ignored by Git. MovieLens has
separate usage restrictions: it is for research use, must be acknowledged, may
not be redistributed without permission, and may not be used commercially
without permission. Read the [official MovieLens-20M README](https://files.grouplens.org/datasets/movielens/ml-20m-README.html)
before downloading it.

## Install

Python 3.12 is required. Install one and only one accelerator extra:

```bash
uv sync --extra cpu --extra dev       # CPU, CI, and serving
uv sync --extra mps --extra dev       # Apple Silicon MPS training
uv sync --extra cu126 --extra dev     # CUDA 12.6 training
```

The mutually exclusive extras follow uv's explicit PyTorch index guidance.
`device=auto` selects CUDA, then MPS, then CPU; a requested unavailable device
fails with a useful error. ONNX, ONNX Script, and ONNX Runtime are core
dependencies because model registration exports and validates the production
artifact on every accelerator configuration.

## Train, evaluate, register, and serve

Training and evaluation are unchanged. Serving requires an explicit immutable
version; `latest` is intentionally rejected.

```bash
deep-recsys data download --root data/raw
deep-recsys data prepare --input-dir data/raw/ml-20m --output-dir data/processed
deep-recsys train --config-name config dataset=movielens20m
deep-recsys evaluate --checkpoint outputs/multvae/best.pt --data-dir data/processed

deep-recsys model register \
  --checkpoint outputs/multvae/best.pt \
  --data-dir data/processed \
  --model-version multvae-onnx-v1

bentoml serve src/service.py:RecommendationService \
  --arg model_tag=deep_recsys:multvae-onnx-v1 \
  --port 3000
```

Registration loads the PyTorch checkpoint on CPU, exports deterministic FP32
logits to ONNX opset 20, and verifies numerical and unseen-ranking parity before
committing the model. The Model Store artifact contains `model.onnx`, ordered
item IDs, movie metadata, configuration, metrics, and a checksum-validated
schema-v3 manifest. Registering the same version twice fails instead of
overwriting it. Service startup revalidates the checksums, ONNX metadata,
dynamic-batch I/O contract, CPU execution provider, and a finite probe inference.

Schema-v2 PyTorch-only model tags cannot be rebuilt with this service because
the production image no longer contains Torch. Re-register their original
checkpoints under a new immutable version; already-built schema-v2 containers
continue to run unchanged.

The public serving endpoints are:

- `POST /recommend` with `{ "movie_ids": [...], "top_k": 20 }`
- `POST /model_info`
- `GET /livez`, `GET /readyz`, and `GET /metrics`
- generated BentoML API documentation at `/`

Set `DEEP_RECSYS_API_KEY` to require `X-API-Key` on `/recommend` and
`/model_info`. Probes, metrics, and generated documentation remain
unauthenticated. Recommendation request bodies are capped at 64 KiB.
`/model_info` reports the artifact schema together with
`inference_backend=onnxruntime`, `execution_provider=CPUExecutionProvider`, and
`onnx_opset=20`.

## Run the backend and serving console with Docker Compose

Compose builds local source-serving images for the BentoML backend and the
standalone serving console. Register a schema-v3 ONNX model first; the default
configuration expects `deep_recsys:multvae-onnx-v1` in `$HOME/bentoml/models`.
Schema-v2 tags such as `deep_recsys:multvae-v1` are not compatible with the
current ONNX-only Service.

```bash
cp .env.example .env
docker compose up --build
```

The default local URLs are:

- serving console and same-origin API proxy: `http://127.0.0.1:8080`
- BentoML API documentation: `http://127.0.0.1:3000`
- backend readiness probe: `http://127.0.0.1:3000/readyz`

Edit `.env` to select another explicit `MODEL_TAG`, a custom
`BENTOML_HOST_HOME`, different ports, or an optional `DEEP_RECSYS_API_KEY`.
The host store's `models` directory is mounted read-only, and both published
ports bind to loopback by default. If authentication is enabled, enter the same
API key in the serving console. To inspect startup failures or stop the stack:

```bash
docker compose logs backend
docker compose down
```

The frontend waits for the backend readiness check. A missing or incompatible
model therefore makes the backend exit and prevents the console from starting;
the backend logs contain the model-loading error.

## Build and containerize for production

For a self-contained production backend image, BentoML owns the Python 3.12 CPU
image and OCI generation. The Compose Dockerfiles intentionally serve source
and mount a host model store; the production flow below embeds the immutable
model artifact into the generated image instead.

```bash
bentoml build src \
  --arg model_tag=deep_recsys:multvae-onnx-v1 \
  --version multvae-onnx-v1

bentoml containerize deep_recsys_service:multvae-onnx-v1 \
  -t deep-recsys-lab:multvae-onnx-v1

docker run --rm -p 3000:3000 deep-recsys-lab:multvae-onnx-v1
```

The public `deep_recsys_service` keeps authentication, request IDs, middleware,
`/recommend`, and `/model_info`. Its `deep_recsys_inference` dependency owns the
model and the single ONNX Runtime session. Both Services use one worker, metrics,
a concurrency limit of eight, and BentoML's 60-second timeout. The internal
`recommend_batch` API adaptively combines requests into batches of at most eight
with a 10 ms latency budget. Each valid batch uses one contiguous NumPy matrix
and one ONNX Runtime call; request-specific catalog errors remain isolated.
Request-vector construction, seen-item masking, and deterministic top-k
selection use NumPy, so the production image has no Torch dependency.

## Controlled inference benchmark

The model-forward benchmark compares the checkpoint and its registered ONNX
artifact in the same process. It uses repeated rounds and exits nonzero if ONNX
Runtime's median p50 is slower; use `--report-only` only for diagnostics.

```bash
python scripts/benchmark_inference.py \
  --checkpoint outputs/multvae-20260818-224228/best.pt \
  --data-dir data/processed \
  --model-tag deep_recsys:multvae-onnx-v1 \
  --output outputs/onnx-inference-benchmark.json
```

The verified local 20,108-item, batch-size-one run used 25 warm-ups and five
rounds of 200 calls. PyTorch recorded p50 `1.466750 ms`, p95 `1.628900 ms`, and
`666.021` inferences/second; ONNX Runtime recorded p50 `1.390500 ms`, p95
`1.454731 ms`, and `714.016` inferences/second. That is a `1.054836x` p50
speedup with maximum absolute score error `0.00001621`. These are same-host
model-forward measurements, not a portable production capacity claim. HTTP
latency and throughput remain non-gating end-to-end service measurements.

The HTTP benchmark sends synchronized waves of eight over warm persistent
connections. Fourteen warm-up requests cover BentoML 1.4's dispatcher
calibration sequence, and 80 logical requests are measured by default. Its
optional functional gate reads BentoML's adaptive-batch histogram and requires
the batch-size sum to exceed its count; p50, p95, and throughput remain
non-gating. The benchmark sends exactly the requested attempts and records HTTP
status counts; the separate OCI smoke request remains the correctness gate.

```bash
python scripts/benchmark_service.py \
  --base-url http://127.0.0.1:3000 \
  --requests 80 --concurrency 8 --warmup 14 \
  --require-adaptive-batching \
  --output outputs/bento-serving-benchmark.json
```

The verified local 20,108-item OCI run recorded successful-response p50
`29.704 ms`, p95 `49.034 ms`, and `207.411` attempted requests/second. BentoML
recorded a batch-size sum of `72` over `35` ONNX calls (mean `2.057`), so the
functional gate passed. Of 80 synchronized attempts, 58 returned 200 and 22
were shed with 503 under the selected hard 10 ms budget; status counts and HTTP
performance remain non-gating.

## Model serving UI

The standalone serving console calls BentoML through a narrow same-origin
proxy. With the Service running on port 3000, start the UI and open
`http://127.0.0.1:8080`:

```bash
python scripts/serve_model_ui.py \
  --backend-url http://127.0.0.1:3000 \
  --port 8080
```

## Reproduction and synthetic smoke

The full reproduction command is:

```bash
deep-recsys train --config-name config \
  dataset=movielens20m device=auto trainer.epochs=200 trainer.batch_size=500 \
  trainer.total_anneal_steps=200000
```

Tests use an in-memory synthetic dataset, so no external data is needed for
the core workflow. The bounded train → register → serve acceptance path is:

```bash
python -m deep_recsys_lab.smoke --output-dir /tmp/deep-recsys-smoke
```

This produces a versioned schema-v3 ONNX Bento model tag, not a resumable
training checkpoint. CI gates numerical/ranking parity, same-process p50
non-regression, and evidence of multi-request adaptive batches. Warm HTTP
p50/p95 latency and throughput remain non-gating artifacts because shared-runner
service measurements are noisy.

The clean analysis notebook in `notebooks/analysis.ipynb` only reads saved
metrics and model metadata. The original downloaded `mulvae-cf.ipynb` is
preserved unchanged and ignored by Git. The local validation record is in
[`docs/acceptance.md`](docs/acceptance.md).

## References

- Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara,
  [Variational Autoencoders for Collaborative Filtering](https://dawenl.github.io/publications/LiangKHJ18-vae_cf.pdf), WWW 2018.
- Dawen Liang et al., [official `vae_cf` notebook implementation](https://github.com/dawenl/vae_cf).
- F. Maxwell Harper and Joseph A. Konstan, [The MovieLens Datasets: History and Context](https://doi.org/10.1145/2827872).
- The prior thesis repository is acknowledged in `docs/references.md`.
