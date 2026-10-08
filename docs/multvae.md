# Mult-VAE on the existing MovieLens 20M splits

## Baseline and adaptations

This implementation follows Liang et al., *Variational Autoencoders for Collaborative
Filtering* (WWW 2018), in `paper/MultVAE.pdf`, and the author's
[MovieLens notebook](https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb).
It uses Mult-VAE PR, with a multinomial likelihood and partial KL regularization.
It does not implement Mult-DAE or a hyperparameter search.

The encoder is `I -> 600 -> (mean[200], log_variance[200])`; the decoder is
`200 -> 600 -> I`. Hidden layers use tanh; posterior outputs and decoder logits
have no activation. Inputs are binary histories, L2-normalized before dropout
with probability 0.5. Weights use Xavier uniform initialization; biases use
truncated normal initialization with standard deviation 0.001 and bounds
[-0.002, 0.002], matching the TensorFlow notebook. Adam uses learning rate 0.001,
betas (0.9, 0.999), epsilon 1e-8, and zero weight decay. The train batch size is
500 users, including the final partial batch.

The loss for each user is `-sum_i x_ui * log_softmax(logits)_i + beta * KL`, where
`KL = 0.5 * sum_j (mean_j^2 + exp(log_variance_j) - 1 - log_variance_j)`.
Both components are averaged by users, not interactions. Training samples latent
codes through reparameterization; evaluation disables dropout and decodes the
posterior mean. No negative sampling is used.

The notebook schedule is retained exactly:
`beta = min(0.2, global_step / 200000)`, with the completed update count used
before each optimizer step. This reaches 0.2 after 40,000 completed updates,
not after 200,000 updates. The current data produces 260 batches per epoch,
so the cap is first used in epoch 154. The seed is 42 (the notebook uses 98765).

The existing per-user splits and 10-core filtering are retained; see
[data_preparation.md](data_preparation.md). The paper instead separates training,
validation and test users. Consequently, these results are not directly comparable
with the paper's published MovieLens table. Checkpoint selection and early stopping
also follow the repository's Recall@20 objective rather than the paper's NDCG@100.

## Data and metrics

`multvae.data` reads `data/processed/ml20m_lightgcn/`, verifies each canonical
SHA-256 against the manifest, checks schema/counts/duplicates/disjointness, and
builds contiguous ID mappings from train. Evaluation labels can be inspected for
these structural integrity checks but never enter training losses or selection
through test metrics. Data preparation is not rerun or modified. User offsets and
item indices stay on CPU; only one dense float32 user batch is transferred to MPS.

Validation evaluates all 124,967 users with validation targets, using train-only
encoder input and excluding train items from ranking. The 4,790 users with no
validation targets are omitted. Final test uses the selected weights and, by
default, train plus validation histories as encoder input. Train and validation
items are excluded from test recommendations. Test labels are not inputs or
optimization targets. An explicit `--test-input train` can instead use train-only
input while still masking both train and validation; the requested baseline uses
`--test-input train-valid`.

Recall@10/20/50/100 divides hits by the user's total number of target interactions,
exactly as the existing LightGCN evaluator. It does not divide by `min(K, targets)`.
NDCG at the same cutoffs uses binary hits, discount `1/log2(rank+1)`, and ideal
discounted gain for `min(K, targets)` items. One sorted top-100 produces all eight
metrics. All 11,508 catalog items are candidates before known-item masking.
Means are macro averages over eligible users. Reported standard errors describe
per-user variation, not variation between training seeds.

## Train and resume on Apple MPS

```bash
uv sync
uv run python -m multvae.train \
  --device mps --batch-size 500 --eval-batch-size 500 \
  --max-epochs 200 --patience 10 --early-stop-start anneal-cap \
  --seed 42 --test-input train-valid \
  --output-dir artifacts/multvae_ml20m/baseline_mps_seed42
```

The requested default selects the strict highest **full-validation Recall@20**
over all completed epochs. Equal scores retain the earlier checkpoint. Validation
is performed every epoch. Early-stopping patience starts at zero at the first epoch
whose final training update uses beta 0.2; after that it increments for checks that
do not beat the global best Recall@20 and resets on improvement. Training stops
after 10 such consecutive checks, or at 200 epochs. The selected checkpoint may
come from before the cap. `--early-stop-start immediate` is available for explicitly
requested runs that may end before annealing finishes.

MPS is required when selected; unsupported operations are not silently sent to CPU.
`PYTORCH_ENABLE_MPS_FALLBACK=1` is rejected for training. CPU mode is available for
small fixture checks. No mixed precision, gradient clipping, or automatic changes
to batch size, annealing, or learning rate are made. Non-finite loss, gradients or
ranking logits stop the run. Timing synchronizes MPS at phase boundaries.

The output directory must be empty for a new run. Resume from the last completed
epoch using the same model, optimizer, data and stopping settings:

```bash
uv run python -m multvae.train \
  --device mps --max-epochs 200 --patience 10 --early-stop-start anneal-cap \
  --seed 42 --test-input train-valid \
  --output-dir artifacts/multvae_ml20m/baseline_mps_seed42 \
  --resume artifacts/multvae_ml20m/baseline_mps_seed42/last.pt
```

The checkpoint restores optimizer state, update count, NumPy/CPU/MPS RNG states,
history, selection state and patience. An interrupted epoch is rerun from its last
completed checkpoint. The embedded best checkpoint in `last.pt` reconciles interrupted
multi-file saves. A run already stopped by early stopping resumes final evaluation
rather than training further. Fixed seeds cannot guarantee identical results across
different MPS, PyTorch or macOS versions.

## Artifacts and English report

Each run saves `config.json`, `data_manifest.json`, `data_audit.json`, `best.pt`,
`last.pt`, `history.json`, `run.log`, and final `metrics.json`. Checkpoints use only
weights-only-compatible values and can be loaded using
`torch.load(path, map_location="cpu", weights_only=True)`.

After training, `multvae.report` generates `report.md`, `history.csv` and PNG figures
for training NLL/objective/KL/beta, all eight per-epoch validation metrics, compute
time, and selected-checkpoint validation/test metrics. Report text and figure labels
are in English. The report distinguishes selected and last epochs, records the
stop reason, counts, environment, fingerprints, measured runtime and observed MPS
memory. Existing LightGCN @20 metrics are included for context only when dataset
fingerprints match; Mult-VAE's additional validation history at test inference is
called out explicitly. No missing LightGCN cutoffs are fabricated.

```bash
uv run python -m multvae.report \
  --run-dir artifacts/multvae_ml20m/baseline_mps_seed42

uv run python -m multvae.evaluate_checkpoint \
  --checkpoint artifacts/multvae_ml20m/baseline_mps_seed42/best.pt \
  --device mps \
  --output artifacts/multvae_ml20m/baseline_mps_seed42/rechecked_metrics.json
```

A report generated before final `metrics.json` exists is clearly labeled in progress.
Final test metrics are unavailable in that report. `--skip-test` is intended for
smoke checks or validation-only experiments.

No unit-test suite or test-runner dependency was added, as requested. Implementation
verification uses syntax checks and small, temporary CPU/MPS fixtures, including
metric arithmetic, deterministic inference, finite gradients, checkpoint reload,
resume state and report rendering:

```bash
uv run python -m compileall -q multvae lightgcn scripts main.py
```
