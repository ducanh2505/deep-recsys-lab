# Treat exposed assessment data as development evidence

An Evaluation Partition provides independent assessment evidence for an Approach only while
its observations have not influenced the fitting, design, or selection decisions under
assessment. Once results from that Partition inform later work, reuse remains valid for
benchmark comparison but cannot support an unbiased final generalization claim for that work;
such a claim requires a fresh, unexposed Assessment Partition.

Treating every immutable or publicly named test split as perpetually independent would make
benchmarking simpler and preserve scarce held-out data, but repeated feedback creates adaptive
overfitting that the split identity alone cannot reveal. The Lab accepts additional provenance
discipline and the possible cost of reserving fresh data in exchange for claims whose evidence
boundary matches the actual design history. Exposure cannot be undone after observations have
influenced a decision, which makes this evidence rule important to establish before results
accumulate.

The Experiment Specification records an `independent`, `exposed`, or `unknown` declaration
for each Assessment Partition relative to its full candidate set and Selection Procedure,
with concise provenance, before the first Run. `Exposed` and `unknown` are limited to benchmark
evidence. The declaration is intentionally comparison-wide: if one candidate was influenced,
the shared comparison is not presented as independent final evidence.

Exposure is based on influence, not merely raw-data access. Aggregate or slice metrics,
failed-case lists, individual outputs, and pass/fail responses all count when a human or
automated agent uses them to alter design or selection. Knowing only the Problem schema and
Evaluation Protocol before a Partition is materialized does not by itself expose that
Partition.

Predeclared repetitions and stopping rules remain independent when executed after an earlier
result is viewed, because the frozen plan prevents that observation from changing their
selection. Repetitions added, removed, or stopped in response to Assessment evidence are
adaptive; they require a new Experiment and cannot turn reuse of that Partition back into
independent evidence.
