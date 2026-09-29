# Paper validation screening

Issue #48 screens Known-User and History-Only independently. The initial plan fixes
15 two-valued Known-User axes (at most 31 settings including baseline) and 11
History-Only axes (at most 23 settings). Budget/checkpoint axes appear first. Each
factor registration names one completed same-mode reference; both alternatives
for an axis must use that same reference. Additional combinations require a new
registration and rationale before their validation run. An unrun axis needs a
recorded measured-loss skip reason before freeze.

The command sequence is:

```text
movie-recsys paper-screen-plan --root SCREEN --source-sha256 DATASET_SHA256
movie-recsys paper-screen-baseline --root SCREEN --mode known_user
movie-recsys paper-screen-run --root SCREEN --run-id known_user-baseline --reference-report KNOWN_47_REPORT
movie-recsys paper-screen-baseline --root SCREEN --mode history_only
movie-recsys paper-screen-run --root SCREEN --run-id history_only-baseline --reference-report HISTORY_47_REPORT
movie-recsys paper-screen-variant --root SCREEN --mode known_user --axis multivae_dropout --alternative 0 --reference known_user-baseline --run-id known-dropout-0
movie-recsys paper-screen-run --root SCREEN --run-id known-dropout-0
movie-recsys paper-screen-freeze --root SCREEN
```

Repeat the variant registration and run for planned alternatives in each mode;
use `paper-screen-skip` with `--mode`, `--axis`, and `--reason` for a defensible
skip. The plan records exact numeric values, quota/order rules, and reasons.
Handlers for later #49–53 factors can be added without changing that plan.
The runner rejects an unimplemented factor instead of logging its requested
configuration as if it had been evaluated.

The first full-data baseline must exactly replay the archived #47 validation
cohort, per-Subject final metrics, source and union diagnostics, Gold loss,
and final macro metrics. This check completes before the cohort seal or a run
result is committed. Archived reports did not store raw Candidate Pools, so the
first replay populates the new cache and pays that scoring cost. Every later
run checks the same training/validation membership, inner fit identity, catalog,
eligible validation Gold Sets, and opaque sealed-test membership commitment.
No test scores enter the screening log.

`SCREEN/fit-cache` stores checksum-verified train-only source fits under exact
snapshot, source configuration, code, seed, and device keys, up to 4 GiB.
`SCREEN/pool-cache` stores full-precision per-source inner and validation pools
in bounded shards under fitted-state, Query, catalog, depth, and code keys, up
to 12 GiB. Both caches reserve at least 8 GiB free. Unchanged sources reuse
their fits and pools while each configuration trains its own inner-fold fusion
classifier and re-evaluates the final validation ranking. History-Only pool
Queries carry an opaque ordinal and fold-in history, never a Subject ID or Gold
label. Run records include source/union/final metrics, per-Subject evidence
checksum, wall time, peak RSS, code fingerprint, and cache hit keys.

The selector uses paired per-Subject Recall@100 intervals within each mode,
NDCG@20 for a Recall tie, and retains the baseline without a positive Recall
point gain. Freeze is write-once and binds both selected and baseline run and
Subject artifacts. Library test entry points require this persisted two-mode
freeze plus matching mode, source, configuration, cohort, and sealed-test
membership. Only explicit checksum-free fixtures can use a fixture receipt.

Screening code changes invalidate source caches and prevent mixing runs with
the previous baseline. After #49–53 handlers change the code, create a fresh
screen directory and replay both #47-verified baselines from the final code
before accepting variant outcomes. The final paper test gate remains closed
until every axis is completed or skipped and both modes are frozen.
