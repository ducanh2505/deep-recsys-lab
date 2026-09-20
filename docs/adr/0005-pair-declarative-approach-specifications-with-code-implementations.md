# Pair declarative approach specifications with code implementations

Approach Specifications are declarative, serializable data, while code implementations
satisfy the target Recommendation Problem's Approach interface at the external seam. A
purely declarative system would expose too much algorithmic structure and produce shallow
interfaces, while code-only plugins would hide identity and reproducibility inputs; the
hybrid keeps Experiment orchestration independent of implementation details and leaves
component seams internal unless a Specification deliberately exposes configurable slots.
These interfaces are internal extension seams for the Lab, not a public third-party plugin
compatibility promise; that promise can be introduced only when real external consumers
justify stabilizing it.

Approach-owned stochastic training policy is part of this declarative identity. For surrogate
negative sampling, the distribution, exclusion rule, samples per positive, and refresh
schedule are Variant bindings because they change the fitted Approach; only the Random Seed
selecting a realization remains a Run dimension. Evaluation distractor sampling stays outside
the Approach under its Evaluation Protocol.

Optimizer semantics, learning-rate policy, batch size, fixed epoch or step budget, and any
early-stopping rule, patience, and observation metric are also Variant bindings because they
change how fitted state is produced. The realized stopping point or chosen checkpoint is a
Run and Artifact provenance outcome of the frozen procedure, not another Variant. Device
placement remains an Execution Environment dimension; content-addressed Artifact identity and
environment evidence capture any concrete numerical difference it produces.

Stopping based only on Fit Partition signals, such as training loss, is wholly an
Approach-owned training procedure. Stopping that observes a held-out metric or selects among
checkpoints uses a Selection Partition and is additionally governed by the Experiment's
frozen Selection Procedure. Neither training nor selection may inspect Assessment evidence.
