# Preserve experiment results as immutable evidence

An Experiment Result is an immutable evidence snapshot: a correction creates a linked
superseding Result, while the invalidated Result remains available for audit and stable
citation. Overwriting would reduce stored versions and lifecycle bookkeeping, but could
silently change conclusions already used by a study or showcase; retaining them accepts that
cost in exchange for traceability. Run Measurements follow the same rule: a
semantics-preserving evaluator bug fix produces a new Measurement, bound to the evaluator
Code Snapshot, that supersedes the invalid one within the same Experiment. Changing metric
semantics instead changes the Evaluation Metric Specification and requires a linked
Experiment under its frozen specification.
