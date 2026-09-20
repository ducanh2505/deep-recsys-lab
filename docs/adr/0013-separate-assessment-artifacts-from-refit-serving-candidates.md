# Separate assessment artifacts from refit serving candidates

An Approach Artifact used to obtain assessment evidence remains immutable and distinct from
any Artifact later built for serving from a broader data scope. After its source Experiment
completes, a declared Refit Procedure may reuse the selected Approach Variant and admit data
that was previously protected for assessment, but it creates a new content-addressed
Artifact through a Refit Build with explicit lineage. A Run remains an execution belonging
to exactly one Experiment; a Refit Build is a separate domain execution and cannot add
Measurements to the source Experiment, even though both may share internal orchestration and
provenance infrastructure. Serving the assessment Artifact directly would preserve an exact
evidence-to-binary link but leave available data unused; overwriting or relabelling it after
refitting would destroy that link. Prior assessment supports selection of the Variant and
Refit Procedure, not the exact quality of the new Artifact, so the refit Artifact must pass
declared parity, integrity, contract, and operational checks before receiving the Serving
Candidate role. That role is never a global Artifact flag: an immutable Serving
Qualification relates the exact Artifact to one Serving Contract, Workload, Environment
Specification, inference runtime Code Snapshot or immutable runtime version, Serving Adapter
implementation, and set of acceptance criteria, so changed targets require new evidence
rather than relabelling the Artifact. A compatible runtime change may reuse the immutable
Artifact bytes but cannot reuse the prior Qualification. The Lab accepts another build and
validation boundary and scoped qualification records in exchange for preserving unbiased
evidence while supporting realistic serving preparation without universal production-ready
claims.
