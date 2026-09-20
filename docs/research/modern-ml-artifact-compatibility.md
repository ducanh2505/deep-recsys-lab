# Modern model-artifact and runtime compatibility

Status: research note, not an architectural decision
Reviewed: 2026-09-17

## Question

Should an old Approach Artifact always execute with the exact inference-code Git commit that
created it? If code changes from C1 to C2 while the manifest schema remains unchanged, must a
formal C2 execution create a new Artifact identity, even when it reuses the payload bytes?

## Conclusion

The underlying goals are current production practice: immutable content-addressed objects,
digest-pinned executable releases, exact provenance for decision-bearing evidence, and fresh
qualification when executable behavior may have changed. They are not obsolete rules.

The proposed boundary is nevertheless stricter than the usual modern design. Making a whole
repository Git commit part of every thin model Artifact identity conflates four independently
versioned things: model payload, runtime implementation, deployment release, and evidence. It
also invalidates identity for unrelated documentation or tooling changes. Modern systems use
both of these legitimate packaging styles:

1. A **bundled executable artifact** contains model data, inference code, and dependencies.
   Changing code creates a new bundle digest.
2. A **versioned model format plus a separately pinned runtime** lets a newer compatible
   runtime load the same immutable model payload. Changing code creates a new runtime or
   serving-release identity, not necessarily a new model-payload identity.

This repository's implementation already follows the second style. The recommendation is to
retain immutable Approach Artifacts but reverse the whole-Git-commit rule for thin Artifacts.
Record and pin C2 as a new Code Snapshot or immutable runtime/container version and create new
execution or serving qualification evidence for `(artifact A1, runtime R2)`. Do not retrain,
re-export, or mint an `A2` merely because the loader implementation changed, unless the
Artifact actually contains executable code or its payload/declared inference format changed.
The Lab does not need a separate Runtime Bundle domain object until a concrete packaging or
deployment workflow needs to own one.

## Separate the identities

| Concern | What its identity should cover | Effect of C1 to C2 |
| --- | --- | --- |
| Model payload / Approach Artifact | Immutable learned state, executable graph if present, catalog and semantic metadata, exact payload checksums, model-format version | Unchanged if those bytes and semantics are unchanged |
| Runtime implementation | Inference source Code Snapshot or immutable wheel/OCI-image version, required libraries, and entrypoint | New identity R2 |
| Serving release | Exact Artifact, runtime identity, serving adapter/configuration, and resolved semantic dependencies | New release D2 |
| Schema compatibility | A declared relation between one reader/runtime and one Artifact format version | Determines whether R2 may load A1; it is not an identity |
| Behavioral compatibility | Evidence from contract, regression, parity, and operational tests under declared tolerances | Must be established for `(A1, R2)`; schema compatibility alone is insufficient |
| Reproducibility / provenance | Source revision, build process, dependencies, environment, and all identities above | Records how A1/R2/D2 were made; need not be hashed into A1 |

A useful representation is:

```text
A1 = digest(model-format manifest + immutable payloads)
R1 = digest(C1 runtime bundle + locked runtime dependencies)
D1 = digest(A1 + R1 + serving configuration + contract)
Q1 = qualification evidence for (A1, R1, D1, environment)

C2 produces R2
A1 remains A1
D2 and Q2 are new
```

If the chosen export is a SavedModel, ONNX graph, compiled program, Bento, or container that
already contains the computation/code, those executable bytes belong to the bundled object's
identity. The split above applies when the Artifact is deliberately a thin payload consumed by
a stable, versioned runtime contract.

## Evidence from current primary sources

### Immutable executable releases are modern practice

OCI separates a manifest, configuration, and content-addressed layers. Its image manifest is
explicitly designed for content-addressable images, and the configuration and layers are
referenced by digest. A runnable manifest should contain enough data to have a unique image
identity ([OCI Image Manifest Specification](https://github.com/opencontainers/image-spec/blob/main/manifest.md)).
Kubernetes likewise states that image digests are immutable and recommends a digest when the
same code must run every time
([Kubernetes Images](https://kubernetes.io/docs/concepts/containers/images/)). Thus a code
change in a bundled runtime normally creates a new image digest and a new deployment revision.
Kubernetes creates a new Deployment revision when the Pod template, including a container
image, changes
([Kubernetes Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)).

This does not require copying unchanged large blobs. An OCI manifest can reference existing
content-addressed blobs, and the Distribution Specification defines mounting an already
existing blob instead of uploading it again
([OCI Distribution Specification](https://github.com/opencontainers/distribution-spec/blob/main/spec.md#mounting-a-blob-from-another-repository)).
Therefore the proposed zero-copy operation is technically sound. However, `artifact rebind`
is project-specific terminology. In conventional tooling it is more naturally a new image,
bundle, or release manifest referencing the same model blob.

SLSA also models output identity separately from source provenance. Build outputs are the
attestation `subject`, while an exact Git commit used by the build is a
`resolvedDependencies` entry
([SLSA Build Provenance](https://slsa.dev/spec/v1.2/build-provenance)). The in-toto statement
binds attestations to immutable subjects by digest
([in-toto Statement specification](https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md)).
This supports retaining a clean Git commit as strong provenance without automatically making
that commit part of the model-payload digest.

### Current ML tooling supports both packaging styles

MLflow and BentoML demonstrate the bundled style. An MLflow Model packages the model with
dependencies and metadata; it can include inferred or explicitly declared source files in the
model's `code` directory
([MLflow dependency management](https://mlflow.org/docs/latest/ml/model/dependencies)). A
Bento build separately names model versions and packages selected source files and locked
Python packages into a deployable Bento
([Bento build options](https://docs.bentoml.com/en/latest/reference/bentoml/bento-build-options.html)).
For these bundles, rebuilding with C2 creates a new deployable bundle even if the referenced
model weights are unchanged.

KServe demonstrates the split style. A `ServingRuntime` declares its container image and the
model-format versions it supports. An `InferenceService` separately supplies `storageUri`,
`modelFormat`, and a chosen runtime; KServe recommends explicitly setting `runtimeVersion` in
production
([KServe ServingRuntime](https://kserve.github.io/website/docs/concepts/resources/servingruntime)).
This is a direct modern precedent for serving the same model payload with C2: retain the model
URI/identity, pin a new runtime image/version, and create a new service revision.

ML frameworks make the same boundary explicit:

- TensorFlow checkpoints contain parameter values but no computation, so source code is
  required to use them. SavedModel includes both the computation and parameters and is
  independent of the source code that created it
  ([TensorFlow checkpoints](https://www.tensorflow.org/guide/checkpoint)).
- PyTorch recommends saving a `state_dict`; loading constructs `TheModelClass` from available
  code and then loads the old state. Saving an entire pickled model is instead tied to its
  classes and directory structure
  ([PyTorch saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)).
- scikit-learn distinguishes interoperable ONNX from Python-object formats. Pickle/joblib/
  cloudpickle require a compatible Python environment, and cross-version loading is explicitly
  unsupported; its guidance is to preserve source and dependency versions and often freeze the
  environment in a container
  ([scikit-learn model persistence](https://scikit-learn.org/dev/model_persistence.html)).
- ONNX versions the IR, operator specifications, and individual model independently. Models
  declare the operator sets they require; semantic changes to an operator require a new
  operator/opset version
  ([ONNX Versioning](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)).

There is consequently no universal rule that an old model must check out its creation commit.
Whether it needs old source, bundles executable semantics, or intentionally accepts a new
runtime is a property of the serialization format and runtime contract.

### Schema compatibility is not behavioral compatibility

Protocol Buffers permits additions such as new fields as binary wire-safe evolution, but its
official guide warns that a wire-safe change may still break application code
([Proto3 updating a message type](https://protobuf.dev/programming-guides/proto3/#updating)).
Its best practices also assume clients and servers are not upgraded simultaneously and may be
rolled back
([Protocol Buffers best practices](https://protobuf.dev/best-practices/dos-donts/)). A readable
manifest therefore proves only that C2 can parse A1, not that C2 produces equivalent rankings.

Likewise, TensorFlow guarantees substantial SavedModel/GraphDef backward compatibility while
explicitly excluding exact floating-point values, random-number sequences, and some bug-fix
behavior from bit-for-bit stability
([TensorFlow version compatibility](https://www.tensorflow.org/guide/versions)). Exact output
parity is therefore not a universal compatibility rule. It is reasonable in this repository
for the declared deterministic Top-N contract, but other Problems should use their own
domain-visible equality, tolerance, or invariant policy.

## Assessment of the proposed rule

| Proposed element | Assessment |
| --- | --- |
| Immutable, content-addressed Artifact payloads | Mainstream and worth keeping |
| Exact recoverable code/environment for final evidence | Mainstream and worth keeping; bind evidence to it |
| Bundled Artifact identity includes bundled code | Mainstream and correct by content-addressing |
| Thin Artifact can only run its creation commit | Defensible reproducibility mode, but not a universal compatibility policy |
| Whole clean repository commit participates in every thin Artifact identity | Overly broad and nonstandard; unrelated files churn identity, and source is more naturally execution provenance or runtime-package input |
| C2 requires new formal evidence | Correct; at minimum create a Run/Trial and qualification bound to a new Code Snapshot or immutable runtime version—not a new model payload |
| Zero-copy A1 to A2 rebind | Safe under a monolithic identity model and analogous to a new OCI manifest referencing old blobs, but unnecessary after identities are separated |
| Development override | Useful, but call it an unqualified development execution and always record commit/dirty state; it need not pretend to override Artifact identity |
| Old evaluation evidence automatically follows C2 | Incorrect; evidence remains bound to R1. C2 needs new evidence proportional to the changed code and claim |

The rule is not “too old”; rather, it mixes a modern immutable-release principle with a coarse
source-checkout mechanism. Checking out C1 is useful for forensic reproduction. Modern serving
normally builds and pins a C1 wheel/container once, instead of performing a Git checkout when
each Artifact is loaded.

## Repository-specific finding

The current code and the recent domain documentation describe different architectures:

- [`docs/architecture.md`](../architecture.md) says Artifact identity is derived from normalized
  configuration, dataset digest, runtime metadata, and payload checksums, and that a runtime
  loader reconstructs scoring.
- [`src/recsys/artifacts/store.py`](../../src/recsys/artifacts/store.py) implements exactly that
  identity basis. No Git commit or loader-source digest participates.
- [`src/recsys/artifacts/runtimes.py`](../../src/recsys/artifacts/runtimes.py) resolves the
  manifest's stable runtime name through the registry in the **currently running code**.
- [`src/recsys/core/provenance.py`](../../src/recsys/core/provenance.py) already records the Git
  commit and dirty state separately as Run provenance.

Under the implemented v1 format, C2 already loads A1 whenever C2 accepts manifest schema v1 and
the named runtime understands the payload. The system does not reconstruct C1, which agrees
with the subsequently accepted separation between Artifact and runtime Code Snapshot. The v1
format still lacks an explicit runtime-contract major, a single canonical identity document,
strict per-runtime metadata schemas, and semantic-reference/evidence gates. The accepted v2
design addresses those gaps; v1 is treated as an experimental legacy format without a new
compatibility reader, migration, or fixture.

## Recommended production-grade learning design

1. **Keep `artifact_id` as the immutable Approach Artifact/payload identity.** Derive it from
   one closed canonical `identity` document containing exact payload descriptors, the Artifact
   format name/version, semantic configuration, Dataset/Binding references, and any
   computation graph actually stored in the payload. Keep `artifact_id` outside that document
   to avoid self-reference, and do not maintain a second field allowlist. Store provenance in
   a separate sidecar or evidence record. For the current local-only scope, reference small
   semantic documents such as the Approach Variant and Problem–Dataset Binding by content ID;
   retain their canonical values in Run/Experiment/Artifact Provenance records instead of
   duplicating them in every manifest. Full self-describing closure is deferred until
   distribution or workspace-independent audit requires it.
2. **Version the loader contract explicitly.** Use a global integer artifact-format major for
   the shared manifest/container and an integer runtime-contract major scoped by a stable
   logical runtime name. Compatibility is keyed by `(artifact_format_major, runtime,
   runtime_contract_major)`; each runtime implementation declares the keys it can read. A
   shared breaking representation change increments the format major, while an
   embedding-specific payload change does not churn unrelated ONNX contracts. Minor and patch
   components are deferred until a concrete compatibility need justifies them. The logical
   runtime name is a persisted canonical ID: class/module renames do not change it, loaders do
   not silently rewrite it, and a genuinely new runtime family uses a new ID and new Artifact
   identity. An incompatible encoding change within the same representation/execution family
   bumps that runtime's contract major; a separately selectable family such as TensorRT beside
   ONNX receives a new runtime ID. A change to Problem-observable behaviour additionally
   belongs to the Approach Specification or Variant rather than being hidden as versioning.
   Both the shared manifest and each logical runtime's inference-affecting metadata use closed,
   typed schemas; unknown semantic fields fail instead of silently selecting defaults. Every
   semantic manifest value is identity-covered or deterministically validated from covered
   content. Creation time, source revision, notes, and build details remain external
   provenance, and a generic extension namespace is deferred until a concrete integration
   needs one.
3. **Give the runtime an immutable execution identity.** Initially use the clean Code Snapshot
   already present in the domain model; when a concrete packaging workflow exists, prefer a
   built-wheel or OCI-image digest and retain its source commit and locked dependencies as
   provenance. This identifies R1/R2, not A1/A2, without adding a speculative Runtime Bundle
   object now.
4. **Bind all decision-bearing evidence to both identities.** An assessment Run, Inference
   Receipt, Serving Trial, and Serving Qualification should record `artifact_id` and the
   runtime Code Snapshot or immutable version, plus contract, environment, dependency
   snapshots, and actual inference mode. Existing evidence for `(A1, R1)` never silently
   becomes evidence for `(A1, R2)`.
5. **Make C2 compatibility an explicit gate.** Check manifest/payload structure, runtime-format
   support, contract fixtures, deterministic ordered Top-N parity where promised, and relevant
   resource/security tests. Use a domain-specific tolerance or invariants where exact bits are
   not a contract. Keep small immutable fixtures for every supported artifact-format,
   logical-runtime, and runtime-contract key in the artifact/runtime test boundary; add them
   with support, retain them while support is claimed, and retire them only when support is
   deliberately withdrawn. These are test assets, not Experiment evidence. A runtime declares
   the exact compatibility keys it currently supports rather than promising to read every
   historical format indefinitely; dropping a key is an explicit breaking runtime release,
   while the original clean Code Snapshot or immutable runtime version remains the forensic
   reproduction path.
   A production rollout can then use a new serving revision and canary; KServe, for example,
   retains previous and latest revisions and splits traffic during rollout
   ([KServe canary rollout](https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary-example)).
6. **Keep two execution policies.** Development may use a dirty/current checkout but produces
   development evidence with explicit provenance. Assessment and serving qualification require
   a clean Code Snapshot or immutable packaged-runtime version.
7. **Retain the bundled option.** If an export includes inference source, compiled objects, or an
   OCI image, those bytes correctly participate in that bundled Artifact's identity. Do not
   force thin and bundled forms into the same identity semantics.

For C2 specifically, the desired formal workflow becomes:

```text
verify that runtime R2 supports A1's format
build and digest R2 from C2
run compatibility and behavioral qualification for (A1, R2)
record a new Run / Serving Trial / Serving Qualification (and deployment revision if served)
reuse A1 unchanged
```

No retraining or payload copy is needed. Re-export A1 only if C2 migrates or transforms the
payload, changes embedded computation, or changes Artifact-declared semantics. If behavior
changes intentionally, follow the repository's Variant/Specification rules and obtain new
evaluation evidence; a schema-version match cannot waive that requirement.

Loading itself remains read-only and never performs an implicit migration. An unsupported
compatibility key fails before payload interpretation, leaving use of the original runtime or
an explicit offline conversion as separate choices. A future converter would produce a new
content-addressed Artifact, preserve lineage to the source, and bind its code, environment,
and parity evidence. A dedicated Artifact Conversion Build record is deferred until such a
workflow actually exists.

## Decision recommendation

Supersede only the exact-implementation clauses of ADR 0004: remove the requirement that a thin
Artifact's identity include the whole clean Git commit and loader entrypoint. Keep its decisions
on immutability, content addressing, integrity, deduplication, and recoverable provenance.
Retain ADR 0009's clean Code Snapshot requirement for assessment evidence, but bind that Snapshot
to the Run or Serving Trial rather than folding it into `artifact_id`.

This layered model is both closer to modern production systems and more educational for an
engineering portfolio: it demonstrates content-addressable storage, versioned formats,
reproducible builds, supply-chain provenance, compatibility testing, evidence scoping, and safe
rollouts without inventing identity churn for unrelated source changes.
