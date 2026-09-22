# Serving Artifact contract

Issue #36 makes the Serving Artifact the hand-off between the lifecycle and the CPU serving
runtime. The local store is deliberately small:

```text
artifacts/fast/
├── <immutable-artifact-id>/
│   ├── manifest.json
│   └── model.json                 # Popularity + Mult-VAE + LightGCN runtime payloads
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
snapshot fingerprint, Popularity API query modes, configuration and configuration checksum, seed,
the CPU serving device, non-negative timings, evaluation metrics, source revision, payload
inventory, payload SHA-256 checksums, bundle integrity checksum, and manifest checksum. The
`multivae_training` and `lightgcn_training` provenance objects separately record requested and
actual neural devices, training duration, seed, fixed hyperparameters, benchmark evidence where
applicable, and fallback reasons. LightGCN's actual device is never rewritten to `mps` after a
CPU fallback.

The top-level `training_device` field remains the CPU artifact/runtime field retained from Issue
#36. For neural device claims, use `multivae_training.actual_device` or
`lightgcn_training.actual_device`; each is the observed device and is never rewritten to `mps`
after a CPU fallback.

Dataset and configuration checksums use canonical UTF-8 JSON with sorted keys and stable
separators. Dataset events are sorted by their canonical representation, so the checksum does
not depend on an absolute input path, event order, or dictionary insertion order.

The runtime loader accepts only a complete, compatible artifact. `model.json` is a checksummed
composite payload containing the Popularity state, Mult-VAE weights, and LightGCN Subject/Movie
mappings and CPU-serving embeddings, plus deduplicated Subject histories and inference
configuration. It rejects missing or malformed fields, unsupported versions, invalid
configuration, partial bundles, absolute or traversing payload paths, missing payloads,
corrupted payloads, inconsistent weight shapes, and checksum mismatches. The runtime uses CPU
inference for both neural retrievers; LightGCN requires a persisted Known-User Subject embedding
and cannot infer one from supplied history. It does not fit a model, read Kafka, or read the
Event Store.

## Commands and deliberate limits

Run the complete local path with:

```bash
uv run movie-recsys fast --output artifacts/fast
uv run movie-recsys serve --artifact artifacts/fast
```

The fast profile exports loadable Mult-VAE and LightGCN payloads and evaluates them beside
Popularity and ItemKNN. The final API response remains Popularity until Issue #39 adds Learned
Hybrid Fusion; Mult-VAE is smoke-tested for Known-User and History-Only serving, while LightGCN
is smoke-tested only for Known-User serving. History-Only and Empty-History never receive
LightGCN Candidates. This issue does not add a distributed registry, multi-host locking, object
storage, retention/garbage collection, continuous rollback, or learned fusion.
