# SASRec semantic boundary and current implementation

Status: research note, not an architectural decision
Reviewed: 2026-09-18

## Question

What must remain true for the repository's `sasrec` built-in to mean Kang and McAuley's
**Self-Attentive Sequential Recommendation (SASRec)**, which deviations are legitimate
Variants, and where does the current implementation conflict with repository domain rules?

This note uses only the original paper and the authors' TensorFlow implementation, pinned to
commit [`e373896`](https://github.com/kang205/SASRec/tree/e3738967fddab206d6eeb4fda433e7a7034dd8b1).

## Conclusion

Keep `sasrec` as the stable Approach key and define it as the **Self-Attentive Sequential
Recommendation Approach**: a causal, position-aware self-attention encoder maps an ordered
suffix of Candidate events to a next-Candidate ranking. It does not learn a Subject identity;
both a stored known-Subject sequence and a request-supplied sequence are native inputs to the
same encoder.

The durable boundary is:

```text
meaningfully ordered Candidate-event sequence
  -> retain the most recent context-window events and left-pad
  -> Candidate embedding + position representation
  -> one or more causal self-attention/point-wise-FFN blocks
  -> final non-padding position state
  -> Candidate relevance scores for next-event ranking
```

For the current implicit-feedback Problem, use an event-preserving Signal Preparation—not
the existing pair-collapsing `binary_positive` binding:

```text
positive_occurrence_sequence:
  keep each event whose value > 0
  preserve repeated qualifying Subject-Candidate events as distinct occurrences
  order them only by domain-valid chronology
```

This is a named binding inside the existing Variant identity, not a new domain object.
Collapsing repeats would erase potentially predictive recurrence and transition evidence.

The current model is still a credible SASRec Variant, but it is not a paper reproduction. It
uses one right-aligned example per historical prefix, predicts only that example's final
position, and applies exact full-Catalog softmax cross-entropy. The paper/reference instead
packs shifted next-event targets into every non-padding position and uses binary logistic loss
with one sampled unobserved Candidate per position. Both train a causal self-attentive
next-Candidate model; they are separate, explicitly named training-objective/example-building
Variants.

## Identity-bearing SASRec semantics

The paper defines the task as predicting the next item from the preceding ordered actions and
uses a shifted version of the action sequence as training targets
([paper, Methodology](https://arxiv.org/html/1808.09781#S3)). It retains the most recent `n`
actions and left-pads shorter sequences
([paper, Embedding Layer](https://arxiv.org/html/1808.09781#S3.SS1)). The following therefore
belong to the Approach boundary:

- **Ordered next-event evidence.** Event order can change the score. An unordered interaction
  set or bag is not a valid substitute.
- **Bounded recent context.** Inference consumes the most recent suffix up to a declared
  context-window length. The window length is a Variant binding, while silently selecting a
  different part of the history changes its meaning.
- **Position-aware causal self-attention.** Each prediction state may attend only to its
  current and previous non-padding positions; future access is target leakage. The paper uses
  learned positional embeddings and explicitly masks links to future positions
  ([paper, Sections III-A and III-B](https://arxiv.org/html/1808.09781#S3.SS2)).
- **Point-wise nonlinear transformation and residual/normalization structure.** SASRec stacks
  attention with position-wise feed-forward transformations rather than reducing history to
  a mean or last-item transition
  ([paper, Sections III-B and III-C](https://arxiv.org/html/1808.09781#S3.SS3)).
- **Next-Candidate ranking from sequence state.** The paper scores a Candidate by the dot
  product between the sequence state and a Candidate embedding, and its default shares the
  input/output Candidate embedding table
  ([paper, Prediction Layer](https://arxiv.org/html/1808.09781#S3.SS4)).
- **No required learned Subject identity.** The paper notes that the history already represents
  the Subject and reports no gain from adding a Subject embedding. The reference model accepts
  a user field but never uses it in scoring
  ([paper, prediction discussion](https://arxiv.org/html/1808.09781#S3.SS4),
  [reference model](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/model.py#L4-L15)).

A bidirectional encoder that can inspect the masked target or future events is a BERT-style
sequential recommender, not SASRec. An encoder whose normal inference semantics discard order
is an attentive set/profile recommender. Either would make the `sasrec` name misleading even
if implemented with Transformer layers.

## Event preparation and chronology

The paper treats the presence of every review or rating as an implicit action and orders
actions by timestamp; it uses the last two actions as validation and test targets
([paper, dataset preparation](https://arxiv.org/html/1808.09781#S4.SS1)). This supports an
event sequence, not one row per distinct Subject-Candidate pair. Repeated events are not
discussed away by the paper and should remain distinct unless a Variant explicitly chooses
deduplication.

The primary sources do not define a rule for equal-resolution timestamp ties. Repository
[ADR-0026](../adr/0026-distinguish-occurrence-time-from-availability-time.md) therefore remains
authoritative: a sequence-dependent Fit requires complete Occurrence Time or a source sequence
with declared temporal meaning, and an unresolved simultaneous group fails preflight. Stable
row order and identifier sorting are reproducible but do not constitute observed chronology.

For next-event evaluation, the query must be a temporal prefix of its Judgment. The paper's
terminal validation/test split satisfies that condition. A random holdout can put later events
in Fit while judging an earlier event; it cannot support a chronological next-event claim even
though the model can mechanically train on the remaining rows. Partition and Evaluation
Protocol remain separate from SASRec identity, but their compatibility must be validated.

## Training construction and objective

The paper constructs a right-aligned fixed-length input and a one-position-shifted output, so
one sampled Subject sequence provides targets at all non-padding positions. Its objective is
binary cross-entropy for positive next events and Candidates outside the Subject's sequence;
the paper samples one negative per time step per epoch
([paper, Network Training](https://arxiv.org/html/1808.09781#S3.SS5)). The authors' sampler
selects a Subject with at least two Fit events, builds the latest shifted window, and uniformly
draws each negative outside that Subject's entire Fit sequence
([reference sampler](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/sampler.py#L12-L41)).

The current repository instead generates every transition once as an independent
`prefix -> next event` example. It retains only the most recent `max_length` events in each
prefix, always reads the last encoder position, and uses exact full-Catalog categorical
cross-entropy ([`sasrec.py`](../../src/recsys/models/sasrec.py)). Consequences include:

- an epoch is an exhaustive shuffled pass over materialized transitions, unlike the reference
  sampler's with-replacement Subject budget;
- long Subject histories contribute early as well as recent rolling-window transitions,
  whereas one reference sample contains only its latest window;
- every non-target Candidate competes in the softmax, including Candidates previously observed
  by the Subject; the reference sampled loss never chooses any Fit-observed Candidate as its
  negative;
- there is no negative-sampling seed or distribution because the objective is exact over the
  Fit-time Catalog.

These are substantial Variant semantics, not grounds to rename the architecture. Exact
full-softmax next-event likelihood is coherent and simpler to reason about, while sampled
binary loss scales differently and is required for paper reproduction. The repository should
name the initial binding, for example `all_prefix_full_catalog_cross_entropy`, and never claim
that its measurements reproduce the paper's sampled objective.

## Architecture: reference versus current code

The authors' default uses two blocks, learned positions, a shared Candidate embedding, Adam
with learning rate `0.001`, batch size `128`, and one attention head by default. Dataset-specific
dropout and context lengths are research settings rather than universal identity
([paper, implementation details](https://arxiv.org/html/1808.09781#S4.SS3),
[reference CLI](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/main.py#L12-L27)).

| Concern | Paper/reference | Current repository | Boundary assessment |
| --- | --- | --- | --- |
| Attention | Scaled dot-product, causal and padding masks; configurable heads, default one. | One causal head with Q/K/V and an additional output projection. | Same SASRec family. Head count and output projection are architecture bindings. |
| Positions | Learned absolute positions over a left-padded fixed window. | Same high-level rule. | Matches. |
| Candidate embeddings | Input embedding scaled by `sqrt(d)`; input/output table shared. | No embedding scaling; table shared for output dot products. | Scaling is a Variant detail; sharing matches the default. |
| Normalization | Paper specifies pre-normalization around each sublayer and final normalization; reference implements that structure. | LayerNorm follows each residual branch (post-norm), with no separate final norm. | Legitimate Transformer/SASRec architecture Variant, not paper-compatible. |
| Feed-forward | Point-wise `d -> d -> d` with ReLU in the paper/reference. | Point-wise `d -> 2d -> d` with GELU. | Width and activation are Variant bindings. |
| Dropout | Input embeddings, attention weights, and FFN paths in the reference. | Residual-branch attention output and whole FFN output only. | Still SASRec, but placement/rate must be explicit and is not a reproduction. |
| Defaults | Paper default has two blocks and tuned dimensions; context is 50 except MovieLens-1M at 200. | `dimension=32`, two blocks, context 50, dropout 0.1. | Compact baseline settings, not canonical paper settings. |

The exact reference structure is visible in its embedding/block construction
([reference model](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/model.py#L15-L68))
and attention/FFN implementation
([reference modules](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/modules.py#L133-L264)).

Changing head count, normalization placement, FFN width/activation, embedding scaling,
dropout placement, initialization, dimensions, or number of blocks remains inside the SASRec
family when declared. Removing causality, supplying an unordered profile as if it were an
ordered sequence, or replacing sequence-conditioned next-event inference with Subject-ID-only
scoring crosses the useful naming boundary.

## Inference, capabilities, and seen Candidates

The authors' implementation scores requested Candidates from the final sequence position
([reference model](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/model.py#L70-L84)).
The current [`SequentialOnnxRuntime`](../../src/recsys/artifacts/runtimes.py) follows the same
rule, scoring every Artifact Catalog Candidate from the last position of either a stored
known-Subject Fit sequence or a supplied history.

Therefore the initial capability contract should be:

- support both `known_subject` and `supplied_history` through the same sequence encoder;
- store the immutable prepared, ordered, last-`context_window` Fit sequence for each known
  Subject in the Approach Artifact; do not reconstruct it from the generic summed train row;
- require at least one prepared event for either query form; an empty sequence fails as
  `missing_required_query_evidence` rather than returning all tied zero scores;
- after preparation, two identical ordered suffixes must produce identical scores regardless
  of Subject ID, earlier discarded events, input values, or request representation;
- score the full Fit-time Candidate Catalog with raw dot-product logits. Do not softmax,
  normalize, clip, or present them as calibrated or cross-query-comparable probabilities;
- retain Candidates already present in the query sequence. SASRec itself produces relevance
  scores; serving-time seen-item filtering is not part of its identity.

The paper's reported evaluation ranks a target against 100 sampled negatives. The reference
chooses those negatives outside the Subject's training history
([reference evaluation](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/util.py#L40-L85)).
That is an Evaluation Protocol binding, not evidence that production inference must remove
seen Candidates. Full-Catalog evaluation/serving without `exclude_seen` remains SASRec; it is
simply not directly comparable to the paper's sampled-candidate numbers.

The ONNX model plus prepared known-Subject sequences is sufficient serving state. Optimizer
state belongs to a Training Checkpoint rather than the Approach Artifact. A Candidate outside
the Artifact Catalog is unsupported. A Catalog Candidate with no positive Fit occurrence may
still receive negative/full-softmax learning signal and remains scoreable; SASRec does not
provide content-based cold-item generalization.

## Comparison with current repository behavior

| Concern | Current behavior | Assessment |
| --- | --- | --- |
| Signal preparation | [`ordered_train_item_indices`](../../src/recsys/datasets/sequences.py) converts every Fit row to a token, ignoring `value`. | **Conflict for the initial implicit Variant.** Zero/negative events become positive actions. Filter events before sequence construction; do not collapse qualifying repeats. |
| Missing chronology | With incomplete timestamps, row position orders all events. | **Domain conflict.** Deterministic ingestion order cannot satisfy an ordered-history contract. |
| Timestamp ties | Row position silently breaks equal timestamps. | **ADR-0026 conflict.** Unresolved simultaneous Fit groups require preflight failure or an explicit group-handling Variant. |
| Partition compatibility | Default `user_holdout` is random; temporal splitting also breaks timestamp ties by row. | A mechanical run is possible, but a next-event claim requires a declared prefix-safe Protocol and valid tie semantics. |
| Fit examples/objective | All prefixes, final-position target, exhaustive epochs, full-Catalog CE. | Coherent SASRec Variant; explicitly distinguish it from paper/reference sampled BCE. |
| Query values | Runtime sequence uses item IDs/order and silently ignores values, including non-positive ones. | **Preparation conflict.** Fit and both query adapters need the same event predicate. |
| Query order | Serving sorts by timestamp only when all events have one; otherwise it trusts list position, and input position silently breaks timestamp ties. | Must be bound as caller-declared sequence order or rejected when chronology is insufficient/ambiguous. |
| Empty query evidence | Public history requests must be nonempty, but a known Subject can have an all-padding stored sequence and runtime returns tied zeros. | Add prepared-evidence validation and fail explicitly. |
| Context window | Runtime consistently retains the last configured `max_length`; generic serving/evaluation also impose a separate 5,000-event cap. | Model truncation is sound. The generic cap conflicts with the current no-transport-limit domain rule and can alter semantics if `max_length > 5000`. |
| Artifact | ONNX scorer and `known_sequences` are stored; generic raw/summed `train` is also bundled. | Runtime state shape is sound, but known sequences must be Approach-prepared immutable state rather than raw-event state. |
| Seen Candidates | Full Catalog is scored and no serving exclusion is applied. | Correct for this repository's contract; paper's candidate sampling belongs to reproduction evaluation. |
| Tests | Tests check train-row containment, one simple temporal ordering case, runtime type/capability, and round-trip serving. | Gaps: value filtering, repeats, prefixes/targets, ambiguous ties, empty evidence, context truncation equivalence, causal leakage, and known/history equivalence. |

## Recommended initial repository contract

1. Keep `sasrec`; use the canonical term **Self-Attentive Sequential Recommendation
   Approach**. Reserve claims of paper reproduction for a dedicated Variant and Protocol.
2. Bind the initial implicit Variant to `positive_occurrence_sequence`: filter event-level
   `value > 0`, retain every qualifying occurrence including repeats, and require domain-valid
   total order. This reuses Signal Preparation and creates no additional object.
3. Reject Fit before Run creation when no valid transition exists or when any required Fit
   sequence contains unresolved simultaneous events. Do not use row order as chronology.
4. Treat the current all-prefix, exhaustive-epoch, exact full-Catalog CE construction as a
   named Variant. Keep the reference shifted-window/sampled-BCE construction as a separate
   reproduction Variant; never switch between them as an implementation optimization.
5. Treat the current single-head learned-position causal encoder with shared Candidate
   embeddings as SASRec. Record post-norm, `2d` GELU FFN, dropout placement, dimension, layers,
   initialization, optimizer, and budget as explicit Variant bindings.
6. Support both query forms. A known Subject supplies the Artifact's prepared Fit suffix;
   supplied history goes through identical event filtering, chronology validation, and suffix
   truncation. Empty prepared evidence fails.
7. Return raw full-Catalog logits and retain seen Candidates. Evaluation candidate sampling or
   masking belongs to an Evaluation Protocol and may not change serving semantics.
8. Add contract tests around preparation/order and equivalence before tuning architecture.
   Current round-trip tests show executability, not that the named semantics are preserved.
