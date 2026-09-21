# Serving Artifact contract

Issue #36 makes the Serving Artifact the hand-off between the lifecycle and the CPU serving
runtime. The local store is deliberately small:

```text
artifacts/fast/
├── <immutable-artifact-id>/
│   ├── manifest.json
│   └── model.json
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
snapshot fingerprint, Popularity query modes, configuration and configuration checksum, seed,
training/serving devices, non-negative timings, evaluation metrics, source revision, payload
inventory, payload SHA-256 checksums, bundle integrity checksum, and manifest checksum.

Dataset and configuration checksums use canonical UTF-8 JSON with sorted keys and stable
separators. Dataset events are sorted by their canonical representation, so the checksum does
not depend on an absolute input path, event order, or dictionary insertion order.

The runtime loader accepts only a complete, compatible Popularity artifact. It rejects missing or
malformed fields, unsupported versions, invalid configuration, partial bundles, absolute or
traversing payload paths, missing payloads, corrupted payloads, and checksum mismatches. The
runtime imports the serving representation only; fitting code, Kafka, and the Event Store remain
on the lifecycle side of the seam.

## Commands and deliberate limits

Run the complete local path with:

```bash
uv run movie-recsys fast --output artifacts/fast
uv run movie-recsys serve --artifact artifacts/fast
```

The fast profile currently exports and serves Popularity. ItemKNN is evaluated but remains an
evaluation-only retriever until a later issue adds a serving payload. This issue does not add a
distributed registry, multi-host locking, object storage, retention/garbage collection,
continuous rollback, or neural model payloads.
