# Movie Recommender Lifecycle Showcase

A local-first portfolio project that demonstrates a time-respecting evaluation stage for an
implicit-feedback movie recommender: Kafka ingestion, a 50% Data Snapshot, a withheld 50–60%
Future Window, Popularity, sparse ItemKNN, Mult-VAE, LightGCN Candidate Pools, query-mode
Learned Hybrid Fusion, immutable Serving Artifacts, FastAPI recommendations, and a static HTML
report.

This orphan branch is intentionally independent from the repository's `main` history. Ticket
#34 implements the ingestion and Popularity tracer bullet; #35 adds the leakage-free
evaluation stage; #36 adds immutable activation; #37 adds the Mult-VAE vertical slice; #38
adds the LightGCN vertical slice; #39 adds query-mode Learned Hybrid Fusion; #40 adds the
rolling 50-to-100 showcase; and #41 completes the static portfolio report and serving-latency
evidence.

## Documents

- [Domain language](CONTEXT.md)
- [Product and engineering specification](docs/spec.md)
- [Implementation roadmap](docs/roadmap.md)
- [Architecture decisions](docs/adr/)
- [Serving Artifact contract](docs/artifacts.md)
- [Tracking issue #36](https://github.com/ducanh2505/deep-recsys-lab/issues/36)
- [Tracking issue #37](https://github.com/ducanh2505/deep-recsys-lab/issues/37)
- [Tracking issue #38](https://github.com/ducanh2505/deep-recsys-lab/issues/38)
- [Tracking issue #39](https://github.com/ducanh2505/deep-recsys-lab/issues/39)
- [Tracking issue #40](https://github.com/ducanh2505/deep-recsys-lab/issues/40)

## Scope at a glance

- A small deterministic MovieLens-shaped fixture becomes immutable Rating Events.
- Ratings of at least 4.0 become Positive Interactions; the original rating is retained.
- The events cross an official Apache Kafka broker in Docker Compose and are stored as
  append-only Parquet batches.
- Snapshot materialization deduplicates deterministic Event IDs after replay.
- Popularity, sparse binary ItemKNN, small seeded Mult-VAE, and snapshot-bounded LightGCN are
  evaluated through one Candidate Retriever contract.
- Mult-VAE uses one binary catalog-index profile for Known-User and History-Only Queries; it
  prefers MPS, records the actual device, and retries on CPU with a recorded fallback reason.
- LightGCN trains a normalized Subject–Movie graph from snapshot Positive Interactions, uses
  persisted Subject embeddings for Known-User Queries only, and records measured MPS/CPU
  device evidence with an explicit fallback reason.
- RRF is reported as a heuristic baseline and Oracle Union is reported only as a coverage ceiling.
- LHF trains separate LightGBM binary classifiers for Known-User and History-Only queries from
  an inner validation window wholly inside the 50% Data Snapshot. Training and serving share a
  versioned feature schema; missing evidence is presence=0, rank=0, score=0.0.
- FastAPI serves LHF for Known-User and History-Only requests and Popularity for Empty-History,
  with observed-Movie exclusion. The immutable artifact carries a minimal CPU ItemKNN payload;
  loading never fits ItemKNN or reads the Event Store.
- A self-contained HTML report separates the 50% Data Snapshot, 50–60% Future Window, retrieval
  coverage, final ranking quality, Popularity, ItemKNN, Mult-VAE, LightGCN, RRF, LHF, Oracle
  Union, realized headroom, and neural resource diagnostics by query mode.
- The rolling command advances cumulative 50%, 60%, 70%, 80%, 90%, and 100% Data Snapshots;
  each 50–90% Future Window is evaluated before its Kafka ingest, and only the smoke-tested
  100% artifact is activated.
- The rolling command benchmarks the loaded 100% CPU artifact after activation. It records
  warm-up/sample counts, p50/p95/p99, throughput, runtime hardware, and per-mode SLO status in
  `latency_benchmark.json`; the report keeps missing samples and failures visible.

Downloaded MovieLens data and generated artifacts are never committed. The checked-in fixture is
deliberately local and small; this repository does not yet provide a pipeline that runs the full
MovieLens 20M dataset, so fixture numbers are not 20M results.

## Reproduce the portfolio report from a clean checkout

Requirements: Python 3.12, `uv`, Docker Desktop, and a shell with `open` (or another local HTML
viewer). From a clean checkout of this revision, run the following from the repository root. The
command uses the official Apache Kafka Compose service and the deterministic fast fixture; it does
not download MovieLens 20M.

```bash
uv sync --dev
docker compose up -d kafka
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run movie-recsys rolling --output artifacts/rolling
open artifacts/rolling/report.html
```

`rolling` creates the six cumulative lifecycle stages, the smoke-tested active 100% artifact,
`latency_benchmark.json`, and the self-contained `report.html`. The HTML file can be opened
directly from disk: it has inline CSS and aggregate JSON evidence and does not need a web server,
CDN, frontend, or interactive dashboard. The report's headline quality is the 90% → 100% Future
Window evaluation; Stage 100 has no future-quality claim. The native-thread limits keep this small
fixture and LightGBM reruns reproducible; the report still records the runtime-confirmed CPU, OS,
Python version, and training/serving device evidence.

Check which artifact is active and inspect the API provenance:

```bash
cat artifacts/rolling/active.json
uv run movie-recsys serve --artifact artifacts/rolling
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/recommendations \
  -H 'content-type: application/json' \
  -d '{"subject_id": 1, "top_n": 10}'
```

The benchmark is an in-process ASGI request-path measurement at concurrency one. The active
artifact is loaded before warm-up; training, artifact loading, and server startup are excluded
from each sample. It is not network or loopback latency. Known-User and History-Only have a
200 ms p95 target; Empty-History has a 20 ms p95 target. An error, too-small sample, or SLO
failure still produces the report with its reason and any successful measurements.

## Run the fast evaluation lifecycle

Requirements: Python 3.12, `uv`, and Docker Desktop (the command starts the official Apache
Kafka Compose service automatically when `localhost:9092` is unavailable).

```bash
uv sync --dev
uv run movie-recsys fast --output artifacts/fast
```

The command creates an artifact store at `artifacts/fast/`, containing an immutable
`artifact-<id>/` directory, `active.json`, `report.html`, `data_snapshot.json`, and
`future_window.json`. It intentionally publishes the fixture twice to exercise at-least-once
replay; the full materialized snapshot remains unchanged by the duplicate batch. Fitting uses
only the first 50% of chronologically ordered events. The next 10% supplies Gold Candidates and
metrics, never model state, fusion labels, or features. The fast stage uses a prefix and an
inner-validation suffix inside the 50% snapshot for LHF supervision, then refits base retrievers
on the complete snapshot before evaluating the Future Window. A fixture without both label
classes is recorded as `untrained fallback`, never as a learned model.

The report compares Top-200 Popularity, ItemKNN, Mult-VAE, and LightGCN pools plus LHF on
Known-User and History-Only cohorts with `Coverage@200`, `ConditionalRecall@10`,
`EndToEndRecall@10`, `NDCG@10`, RRF, the validation-selected best single retriever, diagnostic
Oracle Union coverage, and realized Oracle headroom. LightGCN is available only for Known-User;
Empty-History uses Popularity without fusion.

To serve the loadable artifact:

```bash
uv run movie-recsys serve --artifact artifacts/fast
curl -s -X POST http://127.0.0.1:8000/recommendations \
  -H 'content-type: application/json' -d '{"subject_id": 1, "top_n": 5}'
```

`serve` accepts either an artifact store (which follows `active.json`) or one explicit immutable
artifact directory. The API loads persisted runtime payloads only; it does not fit a model,
replay Kafka, or read the Event Store. Restarting the API with the same store therefore keeps
the artifact identity and recommendation responses stable. Known-User and History-Only routes
return LHF order; Empty-History returns Popularity. Responses retain artifact/snapshot
provenance and the explicit fusion training status.

Run the deterministic tests without Docker:

```bash
uv run --group dev pytest
```

The real Kafka integration test is skipped unless the Compose broker is reachable. Start it
explicitly with `docker compose up -d kafka` when you want to run that test against the real
boundary. Mult-VAE and LightGCN training do not require Docker or an MPS device; unavailable,
incompatible, or slower MPS paths are recorded and retried on CPU. The fast lifecycle selects
CPU explicitly on hosts where the native MPS benchmark is unstable; the individual vertical
slice functions retain their MPS probe/fallback seams.

## Run the rolling 50-to-100 showcase

The rolling command uses the same deterministic fixture and local Kafka boundary as `fast`, but
ingests only the first 50% at the beginning of the run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run movie-recsys rolling --output artifacts/rolling
```

For each stage, the command materializes the cumulative prefix, fits all retrievers and the
query-mode LHF, exports and smoke-tests an immutable artifact, evaluates Known-User and
History-Only queries on the next ten-percent Future Window, and only then ingests that window.
The 100% stage trains and activates its artifact after smoke tests; it has no 100→110% quality
claim. The final headline quality is the 90→100% evaluation.

The output directory contains `rolling_checkpoint.json`, append-only `event_store/` batches,
per-stage `snapshots/`, `evaluations/`, `stages/`, immutable artifact directories, `active.json`,
`latency_benchmark.json`, and `report.html`. A rerun with the same source, seed,
configuration, and code revision validates and reuses completed stages without changing their
artifact bytes. A partial or corrupt stage is
quarantined and rebuilt; a source/configuration/code-revision mismatch fails instead of mixing
runs. Kafka replay remains at-least-once and deterministic Event IDs keep duplicate evidence out
of snapshots and training.

The rolling showcase deliberately remains local and small: it does not download MovieLens 20M,
add a database, distributed scheduler, multi-host lock, registry, Kubernetes deployment, public
hosting, frontend app, or interactive dashboard. `fast` remains available for the original
single-50% lifecycle smoke path; use `rolling` for the complete portfolio report and three-mode
latency evidence.
