# Retain evaluation outcomes for decision-bearing runs

Each Run used by Selection or Assessment retains an immutable Evaluation Outcome Set for
each Evaluation Slice and exact evaluated Artifact or Training Checkpoint digest before
judgments and metric aggregation, with one explicit output or inference failure for every
included case. This permits audit and metric correction without
repeating potentially expensive or externally dependent inference, and prevents failed cases
from disappearing silently; exploratory Runs may omit the set to avoid universal storage
cost, but cannot later supply decision-bearing evidence without rerunning and capturing it.
The retained successful outcomes contain only final Problem-defined outputs and necessary
case and receipt correlation; intermediate candidates, scores, and component contributions
remain opt-in Diagnostic Observations. A metric requiring more output than the Slice
requested therefore requires new inference rather than retroactively treating internal state
as official output. A case-level Inference Failure remains an outcome attributable to the
evaluated system, even when the Protocol stops early; an Evaluation Integrity Failure instead
invalidates the Run because its evidence cannot be trusted. Conflating these categories
would either hide Approach reliability failures or treat evaluator defects as model quality.
Each Evaluation Metric Specification therefore declares how Inference Failures affect that
measure, and the governing Protocol always reports attempted, successful, and failed counts.
There is no Lab-wide mapping: bounded ranking measures may use their meaningful worst value,
while another Problem may require explicit exclusion with coverage or Slice invalidation;
Direct Comparisons require the same declared treatment.
Decision-bearing Runs continue across isolated Inference Failures until the full Evaluation
Cohort is attempted. A Protocol may predeclare resource-protection conditions for early
termination, but a triggered stop leaves the Run Incomplete: its partial outcomes may support
development or operational analysis, never Selection or Assessment quality evidence, and
unattempted cases are reported rather than extrapolated or mislabeled as Approach failures.
An Incomplete Run may resume its unattempted cases under the same identity-bearing
Reproducibility Envelope, preserving recorded outcomes append-only. If a Code, environment,
External Dependency, Artifact, Dataset, Partition, Protocol, or other identity-bearing binding
changes—or a recorded outcome is to be replaced—execution proceeds as a new Run so one Run
never mixes evidence from distinct systems.

Resource exhaustion under the declared execution conditions, including an Approach-caused
out-of-memory condition, is attributable to the evaluated system and is not converted into a
replaceable technical attempt. Evaluation Integrity Failure is reserved for external
infrastructure, data, or harness defects that prevent faithful execution of the frozen
Specification. This distinction keeps capacity and reliability limitations visible while
still allowing the predeclared replacement policy to recover from invalid evaluation evidence.

When the available evidence cannot distinguish these causes, the Run records an unresolved
classification. The attempt remains retained and is neither automatically replaced nor
charged to the Approach. Diagnostic executions may investigate it, but until attribution is
resolved the affected repetition and any Result claim that requires it are inconclusive.

Evaluation Cohort membership is fixed before inference and shared across directly compared
Variants. A runtime failure on a case that the Variant declared it could handle remains an
explicit Inference Failure; it cannot be turned into a Variant-specific cohort exclusion
after outputs are observed.

Metric aggregation may not perform a second, silent cohort filter. Once materialization has
excluded every case without a qualifying Judgment, an empty Judgment reaching a metric is an
evaluator contract defect and therefore an Evaluation Integrity Failure that invalidates the
Run. Skipping it would make reported case counts diverge from the frozen Cohort while leaving
apparently valid evidence.

Every Training Checkpoint whose Measurement influences early stopping or checkpoint selection
therefore retains a separate minimal Outcome Set. Non-selected checkpoint bytes may be
discarded, but their final Problem-defined outputs remain sufficient to correct a
semantics-preserving metric implementation bug and replay the frozen selection rule.
