# Architecture

The platform separates stable contracts from replaceable implementations.

## Boundaries

- `core` owns identifiers, registries, hashing, atomic I/O, workspace paths, and provenance.
- `conf` contains packaged typed Hydra defaults.
- `datasets` validates and materializes generic interactions. Dataset-specific readers live
  in `integrations/datasets`.
- `models` owns training plugins. Framework-specific base classes do not leak into the
  plugin protocol.
- `retrieval`, `fusion`, and `reranking` are independent extension points. `hybrid`
  composes them from configuration.
- `experiments` orchestrates train and evaluation stages and records run-local state.
- `artifacts` defines the immutable portable boundary between training and serving.
- `serving` contains the framework-neutral engine. BentoML is an optional adapter.

Dependencies point inward toward contracts. In particular, the serving engine does not
depend on experiment orchestration, and dataset integrations do not alter core schemas.

## State layout

```text
var/
├── data/
│   ├── raw/                  # user-managed local source files
│   └── prepared/<digest>/    # normalized events, mappings, sparse splits
├── runs/<run-id>/            # config, state, metrics, provenance, scratch space
├── models/<plugin>/<digest>/ # immutable served artifacts
└── cache/                    # disposable provider or feature caches
```

Artifact identity is derived from normalized configuration, the dataset digest, runtime
metadata, and payload checksums. Run IDs identify orchestration attempts and may contain
time; they are not model identities.

## Extension flow

A model or retriever receives `TrainingContext` and returns a `RuntimeSpec`. The runtime
spec declares a plugin name, safe payload arrays/files, a runtime kind, metadata, and
supported query modes. Artifact creation adds the dataset mappings, train interactions,
checksums, schema, and content digest. A runtime loader then reconstructs scoring without
loading Python checkpoints.
