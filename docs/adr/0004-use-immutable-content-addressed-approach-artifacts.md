# Use immutable content-addressed approach artifacts

Committed Approach Artifacts are immutable and identified from their declared inference
semantics, Dataset identity, Problem–Dataset Binding, and content rather than full creation
lineage, a Run ID, or a human-assigned version. Assigned versions and lineage-complete
hashes would be easier to read or make every creation history self-contained, but would
give identical executable objects different identities; content addressing provides
stronger integrity, deduplication, reproducibility, and rollback. Human-readable labels and
Artifact Provenance remain external, and multiple Runs may retain provenance links to the
same Artifact.

Artifact identity covers the exact Approach Variant and declared inference semantics,
Dataset and Problem–Dataset Binding, artifact-format and inference-runtime contract versions,
logical runtime selection, External Dependency Specifications, immutable payloads, and any
embedded computation graph. A bundled Artifact includes Approach-specific inference code,
compiled objects, or an immutable container in its hashed payload. A thin Artifact does not
include the non-bundled runtime implementation's Git commit or source digest in its identity.
That implementation is bound separately as a clean Code Snapshot, or immutable external
runtime/container version, by every decision-bearing Run and Serving Trial.

A Composite Recommendation Approach is assembled from already committed constituent
Approach Artifacts, and a Reranked Recommendation Approach is assembled from one already
committed base Approach Artifact; neither assembly refits its inputs. Composite identity
binds each stable constituent slot to the exact constituent Artifact identity and composite
Variant, while Reranked identity binds the exact base Artifact, reranking Variant, and
inference-phase External Dependency Specifications. Each resulting Artifact packages all
base or constituent inference material plus its owned inference-context payload, so normal
inference has no child-artifact-store lookup and sees one atomic Artifact. A provider model,
URI, or local cache remains an External Dependency unless all required immutable bytes are
deliberately included in the identity-bearing payload.

Source Artifacts remain immutable and independently usable, but their assessment or
qualification evidence does not transfer to the derived Artifact. This duplicates payload
bytes, but holds exact base executables fixed when comparing fusion or reranking methods and
avoids runtime artifact dependencies.

The shared Artifact manifest is a closed schema, and every logical runtime owns a closed,
typed schema for all inference-affecting metadata under its contract major. Unknown fields
fail before inference rather than being ignored or defaulted. A shared structural change
increments the artifact-format major; an incompatible runtime-specific semantic field change
increments only that logical runtime's contract major. The design has no generic extension
namespace until a concrete integration supplies its semantics and ownership.

Every semantic manifest value is either covered by Artifact Identity or deterministically
derived and validated from identity-covered content. Descriptive labels, creation timestamps,
source commits, build duration, notes, and other lineage stay in Artifact Provenance outside
the manifest. This prevents harmless provenance differences from creating duplicate Artifact
identities and prevents a semantic contract field from being changed without an integrity
failure.

The manifest therefore contains the resulting `artifact_id` plus one closed canonical
`identity` document holding every semantic field and immutable payload descriptor. The ID is
the digest of that entire normalized document; it is not produced from a second hand-written
allowlist, and the ID itself is outside the digest to avoid recursion. Adding a semantic field
necessarily changes the canonical document and ID, while provenance is stored in a separate
sidecar or evidence record. This structure removes the failure mode in which a field appears
in the manifest but is accidentally omitted from identity verification.

The identity document references small semantic records—including the Approach Variant,
Problem–Dataset Binding, and inference-phase External Dependency Specifications—by their
content IDs rather than embedding their canonical documents. The governing Run, Experiment,
and Artifact Provenance retain those documents, while the manifest carries only the strict
fields needed for integrity and inference. Dataset content and other large payloads are
likewise referenced by digest. This keeps the current local Lab simple and avoids a new
registry or duplicated schemas; a self-describing export can be introduced only when
cross-workspace distribution or independent archival audit becomes a real use case.

Artifact executability and evidence eligibility are separate checks. When payload integrity
and the strict runtime fields are valid, a missing or hash-mismatched referenced semantic
document does not prevent local development inference. It does prevent assessment, final
Experiment Results, Serving Trials, and Serving Qualifications from using that execution as
decision-bearing evidence. Those workflows resolve and verify every referenced document in a
preflight gate and fail before creating a Run or Trial; the condition is not an
Approach-attributed Inference Failure.

Canonical semantic documents are small and immutable, so the Lab retains every materialized
document indefinitely and provides no automatic garbage collection, reference counting, or
delete operation for them. This simple policy prevents dangling references without creating a
retention subsystem. It applies only to semantic documents, not to Dataset bytes, Artifact
payloads, Training Checkpoints, source bundles, containers, or other potentially large
content, whose retention remains governed separately.

This separation lets a later runtime implementation execute an existing thin Artifact
without retraining, re-exporting, copying its payload, or minting a nominally new Artifact.
It is allowed only when the runtime declares support for the Artifact's format and runtime
contract and passes structural, payload-integrity, and Problem-appropriate behavioural
compatibility gates. Sharing a manifest schema proves only that bytes can be parsed; it does
not prove equivalent inference behaviour. Evidence for `(Artifact A1, runtime C1)` therefore
does not transfer to `(Artifact A1, runtime C2)`: C2 requires new Run or Serving Trial and
Qualification evidence. Resolving C1 remains necessary for forensic reproduction of the
original evidence, not for every future use of A1.

Runtime compatibility is guarded in CI by a small checked-in suite of immutable Artifact
fixtures for every supported compatibility key: artifact-format major, stable logical runtime
name, and that runtime's contract major. The artifact/runtime test boundary owns the suite;
fixtures use synthetic or redistributable data and are test assets rather than domain objects,
Experiments, or decision-bearing evidence. Every runtime change must load all keys it still
declares supported, validate structure and payload integrity, and satisfy the owning Problem's
behavioural comparison rule—exact ordered Top-N output for the current Problem, or an
explicitly declared tolerance or invariant for a future Problem. A fixture is added when
support for a key is introduced, retained while that support is claimed, and removed only when
support is deliberately withdrawn.

Fixtures for Artifacts declaring internal full-Catalog scoring for composition additionally
verify exact score-vector equality under canonical Candidate mapping, as required by the
owning Approach Specification. This applies when a later runtime claims compatibility with
the same thin Artifact: preserving standalone Candidate order cannot justify changing scores
that affect fusion. Composite fixtures still verify the composite's exact final order. A
runtime failing internal score parity cannot bypass the failure through public output parity
or silently remove the Artifact's declared capability. This accepts a stricter numerical
compatibility boundary, including rejection of changes that preserve standalone ranking, in
exchange for protecting composition semantics. The checks remain verification evidence over
declared fixtures, not proof of equivalence for every possible query.

Backward compatibility is explicit rather than indefinite. Each runtime implementation
declares the exact compatibility keys that it supports, and the current runtime rejects every
undeclared key before interpreting payload content. Withdrawing a key is an intentional
breaking runtime release recorded in release notes; it neither mutates nor creates an Approach
Artifact. An older Artifact remains forensically reproducible through its original recoverable
Code Snapshot or immutable runtime/container version, while using a newer runtime requires
that runtime to keep the key declared and its fixtures passing.

The initial scheme uses independent monotonically increasing integer majors, without minor or
patch components. A breaking change to the shared manifest or container increments the
artifact-format major. A breaking change to one logical runtime's payload interpretation
increments only that runtime's contract major; other runtimes do not churn. Retraining under
the same representation changes Artifact content but not either major, while a code-only
refactor or semantics-preserving optimization changes only the execution Code Snapshot and
evidence.

The logical runtime name stored in a manifest is a permanent canonical machine identity. It
does not follow Python class, module, package, or human-facing label renames, and a loader may
not silently canonicalize or rewrite it. If a distinct runtime family needs a new name, new
Artifacts use that identity and therefore receive new content identities; old Artifacts keep
their original name, which remains an independently declared and fixture-tested compatibility
key for as long as current code supports it. A cosmetic rename uses documentation or a display
label rather than changing persisted identity.

A contract-major increment and a new logical runtime name represent different boundaries. An
incompatible payload-layout or interpretation change stays within the same runtime family and
increments that family's contract major when its inference mechanism and public capabilities
remain conceptually the same. A representation or execution family intended to coexist as a
separately selectable implementation receives a new canonical runtime name—for example an
immutable TensorRT engine rather than an ONNX graph. Moving from exact dense scoring to an
approximate index also requires a new runtime name and, because it changes observable
recommendation behaviour, an appropriate Approach Specification or Variant. A refactor or
performance optimization that preserves representation and semantics changes neither field.

Artifact loading is read-only. A loader must reject an unsupported compatibility key rather
than migrating payloads in memory, rewriting the manifest, or producing an unrecorded derived
representation. The caller may instead use a still-supported older runtime. If a future
offline conversion workflow is justified, it creates a new content-addressed Artifact with
lineage to the immutable source and records the converter Code Snapshot, environment, and
parity verification; it never overwrites the source. A first-class Artifact Conversion Build
is deliberately deferred until an actual converter requires an owned execution lifecycle, so
the Lab does not maintain a speculative domain object now.

CLI and serving configuration cannot override the Artifact's logical runtime name or contract
major. They may select an Artifact and an immutable runtime implementation or Code Snapshot
that declares support for its exact compatibility key, but changing the key would change the
identity being interpreted. Low-level tests or diagnostics may invoke another loader
explicitly; those outputs are not attributed to the Artifact, receive no Inference Receipt,
and cannot contribute evaluation or qualification evidence.

For now, one Code Snapshot provides exactly one implementation for each compatibility key.
The runtime does not auto-select among CPU, accelerator, or other backends and does not fall
back according to detected hardware. Supporting concurrent implementations would require a
stable selector owned by the Run or Serving Qualification, separate parity checks, and
separately scoped evidence; that mechanism remains deferred until a concrete comparison or
deployment need exists and would not change Artifact Identity.

The repository's existing flat `schema_version = 1` Artifact format predates this decision and
is treated as an experimental legacy format. The first format implementing the canonical
identity document, explicit runtime contract major, strict runtime metadata, and external
provenance is `artifact_format_major = 2`. The Lab will not add a v1 compatibility reader,
migration, or fixture merely to preserve locally generated artifacts; their bytes are not
deleted, but new code may reject them and they must be regenerated if needed. Dataset manifest
versioning is an independent format and is unaffected.

A semantics-preserving runtime bug fix can keep A1. An intentional change to
Problem-observable inference behaviour must be represented by a new Approach Specification
or Variant and Artifact; a payload or embedded-graph transformation also creates a new
content identity. Supporting both thin and bundled forms keeps lightweight research exports
possible while allowing self-contained deployment packages. Runtime libraries and hardware
remain Execution Environment conditions unless their bytes are deliberately included in the
Artifact, and semantic external services or resources remain declared External
Dependencies. This accepts separate compatibility and qualification evidence in exchange for
avoiding whole-repository commit churn in model identity and for keeping model, runtime, and
release concerns independently auditable.
