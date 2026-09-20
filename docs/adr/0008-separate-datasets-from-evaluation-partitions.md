# Separate datasets from evaluation partitions

A Dataset identifies adapted observations independently of experimental partitioning, while
an Evaluation Protocol materializes immutable Evaluation Partitions with independently
verifiable identities. An Experiment binds those Partitions, and all Approach Variants in a
Direct Comparison use the same concrete assignments or aligned named folds. Embedding
train, validation, and test assignments in a prepared Dataset and its digest would keep the
pipeline and cache model simpler, but it would fragment one observation collection into
multiple Dataset identities and make misaligned comparisons or data leakage harder to
detect. Approaches may transform the fitting data supplied to them but may not repartition
observations across protocol-defined access boundaries. Those boundaries use purpose-based
Fit, Selection, and Assessment roles rather than relying on ambiguous train, validation,
and test labels: fitting may alter learned state, selection may choose a Variant or stopping
point, and assessment remains inaccessible until the relevant choices are fixed. A Protocol
may omit unused roles, but evidence without an Assessment role is identified as development
evidence rather than an unbiased generalization claim. The Lab accepts an additional
materialized object, stricter data access, and reduced usable data in exchange for reusable
Dataset identity, comparable evidence, and explicit leakage controls.

An early-stopping rule that reads only Fit Partition signals is part of Approach training.
Once a held-out metric or checkpoint comparison influences stopping, the observations must
come from the Selection Partition under the frozen Selection Procedure. The Assessment
Partition is unavailable to both paths.
