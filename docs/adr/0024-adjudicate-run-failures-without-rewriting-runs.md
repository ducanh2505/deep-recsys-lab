# Adjudicate run failures without rewriting runs

An ambiguous or disputed Run failure is classified through a separate immutable Run Failure
Adjudication rather than by mutating the Run. The Adjudication identifies one Run, records an
Approach-attributed failure, Evaluation Integrity Failure, or unresolved decision, and retains
the rationale, diagnostic-evidence references, adjudicator identity, and decision time. A
correction creates a new Adjudication that explicitly supersedes the old one; Experiment
Results bind the latest non-superseded decision in their evidence lineage.

This object has a concrete operational role: after a failure such as an unexplained CUDA
process exit, later diagnostics may establish whether the attempt is evidence about the
Approach or invalid evaluation evidence eligible for technical replacement. Storing that
decision directly on the Run would be simpler, but would either rewrite history or leave no
auditable path for correction. The Lab accepts a small additional schema and supersession
lifecycle to keep failure attribution reviewable. An integrity Adjudication does not itself
authorize replacement; the frozen Experiment replacement policy must also permit it, and
diagnostic executions do not become decision-bearing Runs merely by being cited.

The Approach implementation may emit exceptions, resource readings, and other diagnostic
facts, but it cannot label its own failure as an evaluation-integrity defect. The evaluation
boundary owns that classification so an Approach cannot erase its reliability or capacity
failures through self-reporting. Organizational separation is not required for this personal
Lab: the same person may occupy both author and adjudicator roles, provided the adjudicator
identity, rationale, and cited evidence remain explicit.

Automated Adjudications bind the Code Snapshot of their classification logic and their input
evidence. Manual Adjudications instead bind the adjudicator identity, rationale, and cited
evidence. Both are immutable and corrected only through supersession; a separate policy
object is deferred because these bindings already make the concrete decision reproducible
without adding another lifecycle to maintain.
