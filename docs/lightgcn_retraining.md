# Selected LightGCN retraining on train + validation

Run both agreed protocols, verify the dataset, and write the comparison report:

```bash
uv run python -m lightgcn.retrain --device mps
```

The runner requires an empty output directory. To repeat the experiment without
replacing results, pass `--output-dir artifacts/lightgcn_retrain_repeat`.
Paths passed to the LightGCN commands are anchored to the repository root.

The selected architecture remains K=2, embedding dimension 32, fixed layer
weights `(0.975, 0.0125, 0.0125)`, Adam learning rate 0.001, sampled L2 coefficient
0.001, batch size 524,288, and seed 42. The existing train-only checkpoint at
`artifacts/lightgcn_ml20m/best.pt` was selected at epoch 6.

## Data and evaluation

The loader keeps the original train-derived user and item mappings and merges
train plus validation in memory: 7,980,614 positive interactions, 129,757 users,
and 11,508 items. It rebuilds the normalized adjacency, offsets, and negative
sampling bitset from that union. A negative cannot be an item observed by the
user in either source split. Test interactions never enter the graph or positive
training examples; they are not consulted by early stopping.

The loader rejects duplicated interactions, overlap between any two source
splits, unknown IDs, mismatched manifest counts, and users without an unobserved
item to sample. The experiment runner additionally recomputes each split's
canonical SHA-256 using the original four-column preparation format and verifies
the hashes against the manifest before either run.

Each retained checkpoint is evaluated once on all 129,757 test users using
Recall@20 and NDCG@20 over the complete 11,508-item catalog, excluding train plus
validation items. There is no evaluation on the former validation split after
it becomes training data. The evaluator reconstructs the graph from the
checkpoint's `config.train_split`; missing metadata in older checkpoints means
the original train-only protocol.

## Training protocols

**Scratch:** Xavier initialization, a new Adam optimizer, exactly 6 epochs on
train + validation, and the last checkpoint retained.

**Fine-tune:** load the selected epoch-6 train-only weights, verify architecture
and ID mapping, and initialize a new Adam optimizer. The initialization option
loads weights only, even when the source checkpoint contains optimizer state.
Run 3 warmup epochs followed by at most 10 main epochs. With this dataset and
batch size there are 16 steps per epoch and 48 warmup steps. The learning rate
used before each update is `0.001 * min(t / 48, 1)`, where `t` is the one-based
optimizer step. It starts at approximately 0.0000208333 and reaches 0.001 at the
last warmup step. Both embeddings and Adam moments update during warmup; the
same optimizer continues into the main phase.

Warmup does not save candidate checkpoints or advance early-stopping counters.
The first main epoch initializes train-loss selection. Patience is 3 main
epochs; min-delta is 0.0001. The patience reference changes only when the loss
decreases by at least min-delta from the previous significant improvement.
Separately, every new absolute minimum saves a checkpoint, including smaller
improvements that do not reset patience. The selected loss is the epoch's
interaction-weighted mean sampled BPR plus L2 loss, measured during learning,
not a separate end-of-epoch loss pass. Non-finite losses abort the run before
that batch's update.

Train-loss early stopping measures convergence of the training objective and
does not directly select ranking quality. Fine-tuning also inherits 6 prior
train-only epochs and has a different total budget from the scratch run.
The report's single-seed test differences are descriptive comparisons.

## CLI and artifacts

`lightgcn.train` adds these options while retaining train-only validation
selection as its default:

| Option | Behavior |
| --- | --- |
| `--train-split train-valid` | Use the union for graph, positives, and negative rejection. Default: `train`. |
| `--init-checkpoint PATH` | Load compatible initial weights into a fresh Adam optimizer. |
| `--warmup-epochs N` | Linear per-step learning-rate warmup; default 0. |
| `--max-epochs N` | Main epochs, excluding warmup. |
| `--early-stopping validation` | Existing validation Recall@20 selection; forbidden with `train-valid`. |
| `--early-stopping train-loss` | Lowest main-phase training loss; `--patience` controls stopping. |
| `--early-stopping none` | Run all main epochs and retain the last. |
| `--min-delta VALUE` | Significant absolute train-loss decrease; default 0.0001. |

The runner records the exact commands in `artifacts/lightgcn_retrain/summary.json`.
Each of `scratch/` and `finetune/` contains:

- `best.pt`: selected weights, original ID maps, configuration, CPU Adam state,
  and selected training record. The filename also applies to fixed-epoch runs.
- `config.json`: architecture, initialization checkpoint, source epoch, split
  counts, source manifest hashes, stopping rules, warmup steps, and device.
- `history.json`: epoch loss, training time, phase, main epoch, optimizer steps,
  first/last learning rates, and checkpoint-selection events.
- `metrics.json`: test results, epoch counts, selected loss, stopping reason,
  and training/evaluation/total run times.
- `train.log`: complete training subprocess output.

`epoch` counts from the start of the new run; `main_epoch` excludes warmup.
`source_epoch` identifies the epoch of the initialization checkpoint. Saving
Adam state in the original run does not include sampling RNG states. The
continuation command below restores Adam and recovers those random draws for
older checkpoints; continuation checkpoints save the sampling RNG state directly.

The root output directory contains `data_audit.json`, `summary.json`, and the
[measured comparison report](../artifacts/lightgcn_retrain/report.md).
The original baseline checkpoint, metrics, manifest, and split files are preserved.

## Continue fine-tuning with Adam state

Continue from the selected fine-tune checkpoint for at most 10 additional epochs:

```bash
uv run python -m lightgcn.continue_finetune \
  --device mps --max-epochs 10 --patience 3 --min-delta 0.0001
```

This command defaults to `artifacts/lightgcn_retrain/finetune/best.pt` and writes
a separate run under `artifacts/lightgcn_retrain/finetune_continue_10/`. It restores
both learned weights and Adam moments/step counters, keeps the source learning
rate and all architecture/training hyperparameters, and does not warm up again.
It carries the training-loss minimum and significant-improvement reference from
the source checkpoint/history. Patience and min-delta must match the source run.

For an older checkpoint without sampling RNG state, the command replays its
shuffle and negative-sampling random draws through the selected epoch, using the
source seed and batch size. This recovery does not compute gradients or update
weights. New continuation checkpoints save NumPy RNG state and early-stopping
state for direct subsequent continuation. The CPU fixture verifies identical
updates to uninterrupted training after restoring Adam and random sampling.
MPS can still vary slightly because of device/kernel nondeterminism.

If none of the added epochs improves the source loss, the source weights remain
the selected checkpoint. Test is evaluated once after selection, with the same
full-catalog protocol. Output includes `best.pt`, configuration, epoch history,
training log, metrics, data audit, and a
[continuation report](../artifacts/lightgcn_retrain/finetune_continue_10/report.md)
comparing test results with the prior fine-tune and train-only baseline.

`epoch` and `main_epoch` are cumulative across the resumed fine-tuning chain;
`continued_epoch` counts only newly added epochs. Thus the first continuation
starts at total epoch 14 / main epoch 11 after the initial 3+10 epoch run.
Use `--checkpoint PATH --output-dir NEW_DIRECTORY` for another continuation;
the output directory must be empty, and a source with missing Adam state is rejected.

## Verification

The tests use small temporary CPU fixtures and Python's built-in `unittest`;
no additional test dependency is required:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall lightgcn scripts tests main.py
```

Coverage includes graph normalization and split separation, negative sampling,
warmup learning rates and Adam moment continuity, patience and accumulated
min-delta improvements, warmup exclusion from selection, the 3+10 epoch limit,
restoration of the selected weights, non-finite losses, checkpoint compatibility,
ranking masks, and Recall/NDCG denominators.
