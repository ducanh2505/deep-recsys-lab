# Serving Artifact contract

Issue #36 makes the Serving Artifact the hand-off between the lifecycle and the CPU serving
runtime. The local store is deliberately small:

```text
artifacts/fast/
├── <immutable-artifact-id>/
│   ├── manifest.json
│   └── model.json                 # CPU retrievers + query-mode LHF runtime payloads
└── active.json
```

The lifecycle writes payloads to a staging directory under `.staging/`. It validates the typed
manifest and every payload checksum, runs the same three query modes used by the API, and only
then atomically renames the completed directory to `<immutable-artifact-id>/`. A published
artifact is never overwritten.

`active.json` is an atomic pointer containing only the relative artifact ID:

```json
{"artifact_id": "artifact-..."}
```

It is not a copy of the artifact and does not contain an absolute path. Activation writes a
temporary pointer in the same store and uses `os.replace`. If export, validation, or smoke
validation fails, the old pointer is untouched. A missing, malformed, or missing-target pointer
causes runtime loading to fail closed; there is no fallback to another artifact.

## Manifest and runtime

The manifest records the schema/version, artifact identity, `fast` lifecycle stage and 50% data
cutoff, source and positive-interaction counts, implicit threshold, canonical dataset checksum,
snapshot fingerprint, query modes, configuration and configuration checksum, seed,
the CPU serving device, non-negative timings, evaluation metrics, source revision, payload
inventory, payload SHA-256 checksums, bundle integrity checksum, and manifest checksum. The
`multivae_training` and `lightgcn_training` provenance objects separately record requested and
actual neural devices, training duration, seed, fixed hyperparameters, benchmark evidence where
applicable, and fallback reasons. `fusion_training.known_user` and `.history_only` record the
inner-validation snapshot fingerprint, validation event IDs, row counts, fixed LightGBM
parameters, CPU training status, and an explicit `untrained fallback` reason when a valid
two-class fixture is unavailable. LightGCN's actual device is never rewritten to `mps` after a
CPU fallback.

The top-level `training_device` field remains the CPU artifact/runtime field retained from Issue
#36. For neural device claims, use `multivae_training.actual_device` or
`lightgcn_training.actual_device`; each is the observed device and is never rewritten to `mps`
after a CPU fallback.

Dataset and configuration checksums use canonical UTF-8 JSON with sorted keys and stable
separators. Dataset events are sorted by their canonical representation, so the checksum does
not depend on an absolute input path, event order, or dictionary insertion order.

The runtime loader accepts only a complete, compatible artifact. `model.json` payload schema 4
contains the Popularity state, minimal CPU ItemKNN incidence state, Mult-VAE weights, LightGCN
Subject/Movie mappings and CPU-serving embeddings, and separate Known-User/History-Only LHF
classifiers. Each LHF payload includes its exact query-mode retriever bank, feature
schema/version/order, missing-evidence representation, training metadata, and a CPU-loadable
LightGBM model string (or an explicit `untrained fallback` with no model). It rejects missing or
malformed fields, unsupported versions, invalid configuration, partial bundles, absolute or
traversing payload paths, missing payloads, corrupted payloads, inconsistent weight shapes,
incompatible feature banks, and checksum mismatches. The runtime uses CPU inference for every
retriever and fusion classifier; LightGCN requires a persisted Known-User Subject embedding and
cannot infer one from supplied history. It does not fit a model, fit ItemKNN, read Kafka, or read
the Event Store.

## Commands and deliberate limits

Run the complete local path with:

```bash
uv run movie-recsys fast --output artifacts/fast
uv run movie-recsys serve --artifact artifacts/fast
```

The fast profile exports loadable Mult-VAE, LightGCN, ItemKNN, and both LHF payloads and evaluates
Known-User and History-Only modes beside Popularity, RRF, and Oracle Union. Known-User and
History-Only API responses use their LHF order; Empty-History remains Popularity. Mult-VAE and
ItemKNN are smoke-tested for Known-User and History-Only serving, while LightGCN is smoke-tested
only for Known-User serving. History-Only and Empty-History never receive LightGCN Candidates.
This issue does not add a distributed registry, multi-host locking, object storage,
retention/garbage collection, latency benchmarking, an LLM reranker, or a downstream ranker.

## Rolling lifecycle artifacts

Issue #40 adds a resumable local orchestration command:

```bash
uv run movie-recsys rolling --output artifacts/rolling
```

The command uses one source-event ordering `(event_time, event_id)` and advances these exact
cumulative boundaries:

```text
50% snapshot → evaluate 50–60% → ingest 60%
60% snapshot → evaluate 60–70% → ingest 70%
70% snapshot → evaluate 70–80% → ingest 80%
80% snapshot → evaluate 80–90% → ingest 90%
90% snapshot → evaluate 90–100% → ingest 100%
100% snapshot → train/export/smoke/activate (no future window)
```

`artifacts/rolling/` contains a small `rolling_checkpoint.json` ledger plus append-only
`event_store/`, `snapshots/stage-*.json`, `evaluations/stage-*.json`, `stages/stage-*.json`,
immutable artifact directories, `active.json`, and `report.html`. A stage is reusable only when
its stage record, snapshot, evaluation record (for 50–90), artifact manifest/payload checksums,
source dataset checksum, configuration checksum, seed, and source revision all match. A directory
left behind without a complete record is not treated as complete. Invalid partial/corrupt state
is moved to `.rolling-quarantine/` and rebuilt without changing completed stage artifacts.

The stage 50% LHF is trained from inner validation wholly inside the 50% snapshot. Later LHF
stages can use persisted validation rows/outcomes from earlier completed stages; their feature
vectors and candidate pools are retained from the snapshot/model that produced those outcomes.
The current Future Window is never used to train or select the current stage model. Stages 50–90
are published but are not active serving artifacts. Only the 100% artifact can replace the active
pointer, and activation happens after its smoke tests; a failure leaves the previous pointer
unchanged. Stage 100 deliberately contains no future-quality metrics or 100→110% claim.

Rolling manifests additionally carry `rolling_stage`, `rolling_configuration_sha256`, and an
`evaluation_boundary` describing the cumulative cutoff and withheld window. The stage record
and evaluation record carry the per-mode baseline/LHF metrics and measured evaluation/ingestion
timings because the published bundle is immutable before its Future Window is evaluated.

This is a local checkpoint/resume mechanism only. It does not add a database, distributed
scheduler, multi-host lock, model registry, MovieLens 20M download, latency benchmark, or the
portfolio report redesign from Issue #41.
