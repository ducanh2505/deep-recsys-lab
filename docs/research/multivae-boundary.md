# Mult-VAE semantic boundary and current implementation

Status: research note, not an architectural decision
Reviewed: 2026-09-18

## Question

What must remain true for the repository's `multivae` built-in to mean Liang et al.'s
**Mult-VAE** rather than merely "some variational autoencoder," and which differences from
the paper are ordinary Variant bindings?

This note uses only the authors' paper and their accompanying reference notebook. The latter
is pinned to commit
[`ad103c5`](https://github.com/dawenl/vae_cf/tree/ad103c506c8897a853968e1b98c30f31cf5e30f6).

## Conclusion

Keep `multivae` as the stable Approach key, but expand the term in domain documentation as
**Multinomial VAE for collaborative filtering (Mult-VAE)**. It is not a generic VAE and does
not mean a multimodal VAE.

The durable Approach boundary should be:

```text
non-negative unordered Subject-Candidate profile over the Fit-time Catalog
  -> L2 normalization and Fit-only input dropout
  -> diagonal-Gaussian amortized encoder
  -> reparameterized latent sample during Fit
  -> full-Catalog decoder logits
  -> multinomial negative log-likelihood + beta-weighted KL to N(0, I)

inference = decoder(encoder_mean(prepared_profile))
```

For the current implicit-feedback Problem, the simplest initial Variant is
`binary_positive`: filter event-level observations by `value > 0`, then collapse each
Subject-Candidate pair to one, for both Fit and query preparation. A non-negative count
Variant is also within the paper's boundary, but signed values or untransformed explicit
ratings are not a generic Mult-VAE input contract.

`supplied_history` is a native capability, not a heuristic fold-in. The encoder was designed
to infer a latent representation from an unseen Subject's supplied interaction profile.
`known_subject` can be supported as a convenience adapter that supplies the same prepared
Fit profile; it must not introduce a learned per-Subject identity. Equivalent prepared
profiles should therefore produce identical scores.

## What is identity-bearing

The paper defines each Subject's input as a bag-of-clicks vector over the item set, models it
with a multinomial conditional likelihood, and uses a diagonal-Gaussian variational posterior
with a beta-weighted KL term ([paper, Sections 2.1-2.2](https://arxiv.org/html/1802.05814v1#S2)).
The authors emphasize that multinomial likelihood and adjusted VAE regularization are the two
essential changes for collaborative filtering
([paper, Introduction](https://arxiv.org/html/1802.05814v1#S1)).

The following are therefore part of the Approach boundary:

- **Unordered whole-profile evidence.** The input is a fixed-Catalog bag, not a sequence;
  timestamps and event order do not affect scoring.
- **Non-negative evidence.** The paper defines `x` as click counts in the natural numbers and
  binarizes it for the reported experiments. Its likelihood is `sum_i x_i log pi_i`, where
  `pi` is one softmax distribution over the entire Catalog
  ([paper, Section 2.1](https://arxiv.org/html/1802.05814v1#S2.SS1)).
- **A variational bottleneck.** The encoder produces both mean and diagonal variance, Fit uses
  reparameterized samples, and the objective includes KL divergence to a standard Gaussian
  ([paper, Section 2.2.1](https://arxiv.org/html/1802.05814v1#S2.SS2.SSS1)).
- **Multinomial reconstruction.** Full-Catalog `log_softmax` reconstruction is not
  interchangeable with Gaussian squared error, independent logistic loss, or pairwise BPR.
- **Posterior-mean inference.** At prediction time the latent value is the encoder mean and
  Candidates are ranked by decoder logits; sampling at serving time would define a different
  inference rule
  ([paper, Section 2.4](https://arxiv.org/html/1802.05814v1#S2.SS4)).
- **Amortized history fold-in.** The model can score a Subject absent from Fit by encoding its
  supplied profile without optimizing a new Subject embedding. The paper evaluates exactly
  this strong-generalization setting
  ([paper, Sections 2.4 and 4.3](https://arxiv.org/html/1802.05814v1#S4.SS3)).

Removing the variance/sample/KL path produces a deterministic autoencoder such as Mult-DAE,
not Mult-VAE. Replacing the multinomial reconstruction objective likewise leaves this named
Approach. Keeping beta at zero for the entire Fit makes the claimed variational
regularization vacuous and should not be the default meaning of `multivae`.

## What can vary without making the name misleading

These choices materially change a Variant or Experiment but remain inside the Mult-VAE
family when declared explicitly:

- binary presence versus non-negative event counts;
- encoder/decoder depth and hidden/latent dimensions;
- input-dropout probability and parameter initialization;
- optimizer, learning rate, batch size, and fixed training budget;
- the beta cap, warm-up duration, schedule shape, and update unit;
- fixed-final-state versus validation-selected checkpoint export;
- dense exact softmax versus a declared approximation of its normalization;
- whether an Evaluation Protocol masks supplied evidence from its metric ranking.

The final point is deliberately outside model identity. The authors' reference evaluation
sets fold-in items to negative infinity before computing metrics
([reference notebook, evaluation loop](https://github.com/dawenl/vae_cf/blob/ad103c506c8897a853968e1b98c30f31cf5e30f6/VAE_ML20M_WWW2018.ipynb#L1232-L1256)).
That does not require repository serving to implement `exclude_seen`; full-Catalog serving
without such filtering remains Mult-VAE. It only means a reproduction Experiment must state
the paper's evaluation policy.

## Comparison with the current code

| Concern | Paper/reference semantics | Current repository | Assessment |
| --- | --- | --- | --- |
| Fit input | Whole Subject rows; paper permits counts and reports binary implicit input. Reference creates CSR values as ones. | [`multivae.py`](../../src/recsys/models/multivae.py) densifies the shared summed `PreparedDataset.train`; [`prepare.py`](../../src/recsys/datasets/prepare.py) preserves arbitrary finite event values and sums duplicates. | **Semantic conflict.** Negative values invalidate the multinomial interpretation; raw ratings/counts may also express unintended semantics. Bind Approach-owned Signal Preparation. |
| Encoder preprocessing | L2-normalize each row, then apply input dropout only during Fit. Reference uses keep probability `0.5`. | L2 normalization followed by Fit-mode dropout with probability `0.2`. | Mechanism matches; dropout rate is a harmless capacity/regularization binding. |
| Architecture | Symmetric MLP encoder/decoder; diagonal Gaussian heads; reported one-hidden example is `[I, 600, 200, 600, I]` with tanh. | One hidden layer each side, separate mean/log-variance heads, defaults `[I, 64, 32, 64, I]`, tanh. | Same architecture family; dimensions and depth are Variant bindings. |
| Fit latent | Reparameterized sample from the approximate posterior. | Samples `mean + epsilon * std` in training mode. | Matches. |
| Objective | Mean full-Catalog multinomial negative log-likelihood plus beta-weighted analytic KL to `N(0, I)`; no VAE weight decay in the reported setup. | Same `log_softmax` reconstruction and analytic KL; Adam without weight decay. | Matches, conditional on valid non-negative prepared input. |
| KL annealing | Paper starts beta at zero and increases by optimizer update; reference uses `min(0.2, update_count / 200000)`. | Beta is constant inside each epoch and rises from `0.2 / epochs` to `0.2`. | Still a partially regularized Mult-VAE Variant, but not paper-compatible. Bind start, cap, schedule unit, and warm-up updates explicitly. |
| Epoch sampling | Reference shuffles users once, then covers each row once per epoch. | A fresh full permutation is generated for every batch before taking one slice. | Implementation conflict: one named epoch can repeat and omit Subjects. It does not change Approach identity, but its budget semantics are not a data pass. |
| Inference | Disable latent sampling; decode posterior mean; rank raw decoder logits, which are order-equivalent to softmax probabilities. | ONNX export calls `sample=False` and returns decoder logits. | Matches. Scores are uncalibrated ranking evidence for one query, not cross-query confidence. |
| Query capability | The encoder natively folds in an unseen Subject from supplied history. A known Subject is not required by the model. | Declares both known-user and history modes. Known-user supplies its stored train row; history sums request values. The generic evaluator preferentially selects known-user mode. | Capability set is sound, but both adapters currently inherit the raw-value preparation conflict. The current evaluator therefore does not demonstrate the paper's unseen-Subject fold-in capability; that is an Evaluation Protocol coverage gap, not a model-name conflict. |
| Model selection | Reference keeps the checkpoint with best validation NDCG. | Exports the state after the final configured epoch. | Experiment/Selection binding, not an Approach-name conflict. It must be declared rather than conflated with paper reproduction. |

The architecture and objective correspondence can be checked directly in the authors'
reference notebook: it L2-normalizes and drops out the input, splits encoder output into mean
and log variance, samples only while training, and optimizes multinomial NLL plus annealed KL
([reference implementation](https://github.com/dawenl/vae_cf/blob/ad103c506c8897a853968e1b98c30f31cf5e30f6/VAE_ML20M_WWW2018.ipynb#L732-L809)).
The reported hyperparameters—batch size 500, 200,000-update warm-up, beta cap 0.2, and a
`[I, 600, 200, 600, I]` network—are reproduction settings, not the universal model definition
([reference training setup](https://github.com/dawenl/vae_cf/blob/ad103c506c8897a853968e1b98c30f31cf5e30f6/VAE_ML20M_WWW2018.ipynb#L957-L984)).

## Recommended initial repository contract

1. Keep stable key `multivae`; define its display/domain name as **Multinomial VAE
   Approach** and cite Liang et al.
2. Bind the initial implicit Variant to `binary_positive` at event level for both Fit and
   query. Do not consume the generic summed train matrix as its semantic input.
3. Support both `supplied_history` and `known_subject`. Implement `known_subject` by loading
   the Artifact's immutable prepared profile and running the same encoder path; do not store
   or learn Subject embeddings.
4. Reject an empty prepared query profile as `missing_required_query_evidence`; an encoder
   bias producing scores from an all-zero vector is not a documented cold-start method.
5. Score every Fit-time Catalog Candidate with deterministic posterior-mean decoder logits.
   Do not normalize, clip, sample, or silently filter previously observed Candidates during
   serving.
6. Give beta scheduling identity-bearing Variant fields at least `start`, `cap`,
   `warmup_updates`, and schedule shape. Count optimizer updates, not epochs, because beta is
   applied per update.
7. Keep architecture size, dropout, optimizer profile, budget, and checkpoint-selection
   policy as explicit Variant/Experiment bindings. Do not claim paper reproduction merely
   because the Approach key is `multivae`.

The primary sources do not settle how this repository should treat Fit-time Catalog
Candidates with zero positive Fit evidence. The paper's experiment constructs its item set
from training Subjects, so that question should be decided explicitly for this Lab rather
than inferred from the reference dataset filter.
