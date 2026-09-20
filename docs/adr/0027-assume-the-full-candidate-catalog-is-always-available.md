# Assume the full candidate catalog is always available

Every Candidate in the exact Catalog bound by a Problem–Dataset Binding is assumed available
for every serving invocation and evaluation case. The Lab does not model inventory, active
windows, business-rule eligibility, or caller-supplied candidate subsets. Approach inference
operates against that complete Catalog, although an Approach remains free to retrieve or rank
only an internal subset; misses caused by that choice remain part of its measured behaviour.

A Candidate does not become ineligible or irrelevant merely because the query evidence
contains an earlier interaction with it. A later held-out interaction may still construct a
valid Evaluation Judgment for that Candidate. A novelty-only objective must be declared by a
different Problem Specification rather than introduced by filtering inference output or
evaluation targets.

The current implicit-feedback Evaluation Protocol constructs binary Candidate relevance.
Repeated qualifying held-out events for the same Subject and Candidate form one relevant
Candidate; event frequency remains available to Approach Signal Preparation but affects
evaluation utility only under another explicitly declared frequency- or value-graded
Protocol.

For that binary implicit Protocol, an individual held-out event qualifies only when its
canonical value is greater than zero, and the predicate is applied before repeated events
collapse to Candidate relevance. Zero-valued and negative events remain Dataset observations
but do not create positive Judgments. A different predicate or threshold is a different
Evaluation Protocol identity rather than an evaluator implementation detail.

A Subject with no qualifying Candidate after that predicate is not an evaluable case for the
binary ranking metrics. The Protocol excludes it while materializing the shared Evaluation
Cohort before any Approach inference, records `no_qualifying_judgment` as the reason, and
reports the excluded count. Assigning zero would pretend an undefined target was a model miss;
treating it as an Inference Failure would attribute an evaluator precondition to the Approach.

If correct materialization leaves no cases in the Evaluation Cohort, the Dataset–Protocol
combination cannot produce evidence. Experiment preflight is rejected, the exclusion summary
is retained for diagnosis, and no Run or Measurement is created. This is not an Evaluation
Integrity Failure because the evaluator honored its contract and execution never began; zero
metrics would falsely resemble measured model quality.

Request-specific eligibility would make the serving simulation closer to many real products,
but it would add business-domain inputs and policies that this research repository does not
need. The simpler full-Catalog assumption improves parity and comparability between evaluation
and serving at the cost of making production-applicability claims explicitly inapplicable to
inventory or policy filtering. Evaluation-only candidate sampling, if supported, must be
named as measurement methodology rather than item availability.

The Evaluation Protocol may measure against the full Catalog or use an explicitly sampled
candidate methodology needed by an academic benchmark. A sampled Protocol freezes sampler
semantics, sample size, randomness inputs, and exact materialized per-case samples, and every
Variant receives the same samples. Results are labeled as sampled and are not directly
comparable with full-Catalog evaluation or a different sampling scheme. Unsampled items remain
Catalog members; sampling never becomes serving availability policy.

Evaluation samples are called Sampled Distractor Candidates rather than negatives by default.
In an implicit-feedback Problem, absence of an observed event does not establish negative
preference; the distractor only supplies bounded rank competition for the case's target. The
Protocol excludes relevant targets, declares how other interactions are treated, and shares
the exact materialized distractors across Variants. Approach-owned surrogate negative sampling
for training remains a separate Signal Preparation choice.

Point-in-time distractor construction uses only information available at the query time;
future interactions cannot be consulted merely to make the sampled pool look more negative.
A Reported-result Reproduction may deliberately preserve a paper's all-time interaction
exclusion for fidelity, but its result is scoped as benchmark evidence and cannot support a
point-in-time-correctness claim.
