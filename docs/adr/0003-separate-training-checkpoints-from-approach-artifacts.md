# Separate training checkpoints from approach artifacts

Training Checkpoints are Run-local resumability state, while Approach Artifacts are the
verified inference boundary used by evaluation and serving; serving never loads a Training
Checkpoint as its source of truth. Serving checkpoints directly would avoid an export step
and duplicated payloads, but would couple serving to training frameworks and serialization,
so the Lab accepts export and validation costs in exchange for portability and integrity.

A checkpoint may also be a stopping-point candidate considered by a Selection Procedure.
The Run retains an immutable trace of every considered stopping point, checkpoint digest,
Measurement reference, and final selection. Only the selected Checkpoint must be exported and
verified as an Approach Artifact. Non-selected Checkpoint bytes may be discarded after the
Run unless a Research Study declares a retention requirement; their decision trace remains
even when the training-framework payload does not.

Because checkpoint Measurements influence the stopping decision, each considered checkpoint
retains its own minimal Evaluation Outcome Set keyed by checkpoint digest. This preserves the
final Problem-defined per-case outputs needed to recompute a corrected metric without keeping
the framework checkpoint bytes or internal scores. Storage grows with the number of evaluated
stopping points, but otherwise deleting a non-selected checkpoint would make the selection
decision impossible to audit or correct.

The selected checkpoint is not assessment-ready merely because selection succeeded. The
exported Artifact must pass the Variant's predeclared output-parity rule. Personalized Top-N
Ranking requires exact ordered Candidate outputs; a numeric tolerance exists only when numeric
values are themselves part of another Problem's output contract. Evidence binds the
checkpoint digest, Artifact identity, verification inputs, comparison logic Code Snapshot,
and outcome. Verification uses predeclared contract fixtures or Fit/Selection cases bound by
the Experiment or Run, never Assessment inputs. Assessment always invokes the verified
Artifact through the inference runtime Code Snapshot bound by that Run. A parity failure
rejects the Artifact and blocks Assessment rather than silently evaluating or serving the
framework checkpoint.

An Artifact declaring internal full-Catalog scoring for composition also passes a separate
Approach-Specification-owned parity check: exact score-vector equality under canonical
Candidate mapping on the same predeclared verification cases. Public Candidate ordering alone
cannot establish this capability's preservation, because changing score magnitudes while
preserving standalone order can change a z-score composite's final output. The composite also
checks its own exact final Candidate order. A failed internal check rejects export rather
than silently removing the declared capability or accepting public parity as sufficient.
This stricter baseline may reject numerical differences that preserve standalone ranking,
but protects composition semantics without introducing score tolerances or public score fields.

The current workflow supports only behaviour-preserving export from the selected checkpoint.
Any change to ordered Top-N output fails parity. An intentionally output-changing export such
as ranking-altering quantization must later be modeled as a distinct Approach Variant and be
selected and evaluated using its exported representation; it cannot inherit the full-precision
checkpoint's evidence. That workflow is deferred until a concrete research need justifies it.
