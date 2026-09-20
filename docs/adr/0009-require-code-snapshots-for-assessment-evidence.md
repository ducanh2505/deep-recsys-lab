# Require code snapshots for assessment evidence

Every Run contributing to assessment or a final Experiment Result binds an immutable,
recoverable Code Snapshot; the default is a clean Git commit reachable through declared
repository provenance. Allowing an arbitrary dirty worktree would keep iteration fast, but
a commit plus a dirty flag cannot reconstruct the code that actually ran. Automatically
archiving the whole worktree would preserve more state but risks capturing datasets,
credentials, and unrelated files. Exploratory dirty-worktree Runs therefore remain allowed
as development evidence, while they become assessment-eligible only if a future mechanism
safely materializes their relevant source as a content-addressed Snapshot. This gate adds
commit or snapshot discipline in exchange for making published evidence recoverable rather
than merely attributable to an approximate code revision.

For a thin Approach Artifact, the inference runtime's Code Snapshot is bound to execution
evidence rather than included in Artifact Identity. A later clean Snapshot may execute the
same Artifact only after satisfying its declared artifact-format and inference-runtime
contract plus structural, integrity, and behavioural compatibility checks. It produces new
Run evidence; it does not inherit the conclusions obtained under the earlier Snapshot. The
whole clean repository commit is the initial deliberately coarse Snapshot, so an unrelated
commit changes execution provenance without forcing a duplicate Artifact. Exact reproduction
still resolves the original Snapshot, while ordinary compatible use may run under a newer
one.

Assessment preflight also resolves every semantic document referenced by the Artifact and
verifies its content ID. An Artifact whose strict runtime fields and payload remain valid may
still support local development inference when a reference is missing or mismatched, but it
cannot start a decision-bearing Run or contribute to an Experiment Result. This is an
evidence-integrity failure before inference, not an Inference Failure attributed to the
Approach.
