# Distinguish occurrence time from availability time

Temporal Dataset Adaptation preserves Occurrence Time—the time an event or value happened—and
Availability Time—the earliest time it was knowable to the simulated system—as distinct
semantics. An Evaluation Protocol making point-in-time or production-applicability claims
admits observations and features using Availability Time. When the source supplies only
Occurrence Time, the Protocol must explicitly assume both times are equal and scope its claim
accordingly. Non-temporal benchmarks are not required to supply either field.

That equality assumption applies to a whole observation family consumed by the Protocol, not
selectively to records whose Availability Time happens to be missing. A point-in-time
Protocol requires complete Availability Time for that family or the family-wide equality
assumption; partial population is rejected until Dataset Adaptation resolves the source
semantics. This avoids interpreting unknown provenance as immediate availability while
leaving the Dataset usable for non-temporal claims.

For an observed Interaction Event, Availability Time must be equal to or later than
Occurrence Time; an earlier value is internally inconsistent and fails Dataset Adaptation.
This is not a universal comparison against dates carried by metadata: a future business date,
such as a scheduled release, can legitimately be known in advance and is not the Occurrence
Time of an already observed interaction. Effective-dated metadata semantics remain deferred
until a concrete Problem requires them.

Naive temporal values default to UTC to keep the generic ingestion path simple. A
source-specific Dataset Adaptation may explicitly declare another source timezone, and all
canonical temporal values are normalized to UTC. The chosen interpretation affects the
canonical instants and therefore Dataset identity; the Lab does not treat the UTC default as
an unknown-timezone error. The current generic parser's UTC interpretation is consistent
with this decision, while source-timezone overrides may be added when a concrete adapter
requires one.

A single timestamp is easier to ingest and is sufficient for many static academic datasets,
but it cannot represent delayed ingestion, late-arriving events, or features computed after
the prediction point. Once that information is discarded, leakage cannot be diagnosed or
repaired from the prepared Dataset. The Lab accepts an optional second temporal semantic and
extra validation in exchange for point-in-time-correct evidence, while retaining the explicit
equality assumption as the simple path for sources that lack availability data.

For an ordered behavioral-history query, these fields have sequential rather than competing
roles: Availability Time filters what the simulated system could know at the cutoff, then
Occurrence Time and any valid source sequence order the admitted events by source-domain
behavior. A late event can therefore be absent at an earlier cutoff and inserted into its
behavioral position after becoming available. An arrival-ordered stream has different input
semantics and must be a separately declared Problem query form and Evaluation Slice; the
evaluator cannot substitute it implicitly.

When timestamps tie at the source's available resolution, row order, identifiers, or a
deterministic implementation tie-breaker cannot be presented as observed chronology. A
source-provided sequence may order the events only when its contract gives that sequence
temporal meaning; otherwise the events are one simultaneous group that a temporal Protocol
cannot split across an access boundary. This may reduce usable sequential cases, but avoids
turning reproducible arbitrary ordering into false causal evidence.

A source-provided sequence with declared chronological meaning may supply chronology when
Occurrence Time is absent or refine events tied at the available time resolution. When both
are present, the sequence cannot reverse the order of distinct Occurrence Times; such a
conflict fails Dataset Adaptation instead of silently privileging either representation. This
distinguishes a legitimate temporal sequence from ingestion order or another counter whose
semantics only happen to be monotonic.

For a per-case terminal holdout, a cutoff that would divide an unordered tied group moves
before the whole group, leaving every tied event held out. When the target Problem cannot
interpret that group as a valid multi-target Judgment, or the remaining history is shorter
than the Protocol's declared minimum, the case is excluded under a predeclared Evaluation
Cohort rule with its reason and count reported. The evaluator never chooses one tied event
merely to satisfy a desired split size.

An ordered-history query or sequential Approach also requires chronology preserved by the
Dataset, independently of whether a split is temporal. Problem–Dataset Binding rejects that
query form when neither complete Occurrence Time nor a source-provided sequence with declared
temporal meaning can order its evidence; deterministic file positions or identifiers cannot
repair the missing semantics. The same Dataset may remain compatible with an unordered
interaction-set query.

Even when most of a Dataset has usable chronology, an individual ordered-history evaluation
case can contain a tied event group whose order the source does not define. The ordered Slice
excludes that case before inference under `ambiguous_required_query_order` and reports the
reason and count; it does not manufacture an order from identifiers, row positions, or a
deterministic implementation tie-breaker. The case remains eligible for a separately declared
unordered Slice whose query semantics accept the simultaneous evidence. A richer query form
that represents simultaneous groups is deferred until a concrete Problem requires it.

The same constraint applies before fitting, not only when constructing evaluation queries.
When an Approach Variant's Signal Preparation requires totally ordered sequences, an
unresolved simultaneous group in its Fit Partition makes that Experiment fail preflight
before a Run or Approach Artifact is created. This does not invalidate the Dataset for
unordered Approaches. The Lab rejects arbitrary tie-breaking and implicit removal because the
former fabricates transitions while the latter silently changes the training population. A
future Variant may instead declare identity-bearing simultaneous-group handling or exclusion
semantics, together with affected counts, when a concrete use case justifies them.

The current implementation conflicts with this decision: `ordered_train_events` uses the
stored training-row position to break timestamp ties, and both the Markov and SASRec fitting
paths consume that manufactured total order. This must be corrected early by adding preflight
validation and removing row-order fallback from order-dependent fitting rather than treating
the existing behavior as intended domain semantics.
