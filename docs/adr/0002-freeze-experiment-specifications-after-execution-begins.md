# Freeze experiment specifications after execution begins

An Experiment Specification becomes immutable when its first Run starts; semantic changes
create a new linked Experiment, while non-semantic display metadata may still be edited.
Mutable specifications would make iteration cheaper and produce fewer records, but could
silently change the meaning of historical results, so invalid Experiments are marked as
such rather than rewritten. One Experiment may contain both selection and assessment when
its candidate Variants, metrics, decision rule, tie-breakers, and assessment transition are
declared before execution. The selected Variant is then an outcome of the frozen procedure,
not a mutation of the Specification; adding candidates or changing the procedure after
observing selection evidence requires a new linked Experiment.

The frozen plan also includes the decision-bearing repetition count, Run dimensions, and any
sequential stopping or completion rule. Later repetitions that were already planned remain
valid after an earlier Assessment result is viewed because that result cannot change their
role. Adding or suppressing repetitions, or stopping based on observed Assessment evidence,
cannot alter the original Experiment Result; such adaptive execution requires a new linked
Experiment and treats the reused Assessment Partition as exposed.

The Specification also freezes a technical replacement policy. An eligible Evaluation
Integrity Failure may be superseded by a new full Run with the same planned dimensions; the
invalid attempt and supersession lineage remain retained, while only the eligible replacement
contributes to the Result. A valid completed Run is never replaceable because its observed
metric is unfavorable. This permits recovery from infrastructure or harness failures without
allowing retries to become result selection.
