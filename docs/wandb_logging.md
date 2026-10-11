# W&B training logging

## Current scope

The three standalone training commands enable W&B **online by default**:

- `python -m content_recsys train`: projection or LoRA, BPR or InfoNCE.
- `python -m multvae.train`: the Mult-VAE baseline.
- `python -m lightgcn.train`: fresh training or a resume from its new `last.pt`.

All runs use W&B Cloud, team `ducanh250599`, project `RecSys`, unless overridden.
The project is at <https://wandb.ai/ducanh250599/RecSys>. The display name defaults
to the output directory name. A family tag (`content_recsys`, `multvae`, or
`lightgcn`) and device tag help filter runs; seed and model settings are in config.
Each standalone experiment has its own group, shared by its offline sessions.

Logging includes validation and final test metrics **already computed by the
trainer**. It does not add evaluation passes or change checkpoint selection.
Content search, LightGCN search/retrain/continuation, standalone evaluation,
smoke and preparation commands do not enable W&B through this integration.
`lightgcn.retrain` explicitly disables W&B in its child training commands, and
programmatic content callers omit tracking options. The separate Mult-VAE
objective experiments (`search`, `extend_search`, `compare_scratch`) already
require online W&B; see [their guide](multvae_objectives.md) for that workflow.
The broader orchestration and evaluation ideas below remain proposals.

## Setup on the Mac

From the repository root, install the locked dependencies:

```bash
uv sync --locked --extra content
```

The `content` extra is needed for the content encoder, but not for LightGCN or
Mult-VAE. Use `uv sync --locked` when only running those two models.

Keep `WANDB_API_KEY` in the root `.env`, or export it in the shell that starts
training. This integration requires that environment variable for online mode;
it does not run an interactive login or read `.env` itself. The key must have
permission to write to `ducanh250599/RecSys`. No additional login is needed when
the environment variable contains a valid key.

Always include `--env-file .env` in the examples below. Python does not load
`.env` automatically. Alternatively, in your own terminal:

```bash
export UV_ENV_FILE="$PWD/.env"
```

That export affects the current terminal and its child processes only. W&B
credentials are not placed in experiment config, checkpoints or uploaded files.
The real `.env` remains ignored by Git. Supply credentials separately on Vast.ai.

## Training examples

Use a **new output directory** for a new experiment. Existing baseline outputs
must not be overwritten. Commands below are examples to run manually; adding the
integration does not launch them.

### LightGCN

```bash
uv run --env-file .env python -m lightgcn.train \
  --device mps --batch-size 524288 --max-epochs 30 \
  --eval-every 2 --patience 5 --validation-users 20000 \
  --output-dir artifacts/lightgcn_ml20m/wandb_seed42
```

This enables online logging without any W&B flags. The existing LightGCN
evaluator reports Recall@20 and NDCG@20. Intermediate validation can use a subset
of users; final `full_validation` uses all eligible users. Do not compare the two
as if they were measured on the same population.

### Mult-VAE

```bash
uv run --env-file .env python -m multvae.train \
  --device mps --max-epochs 200 --patience 10 \
  --output-dir artifacts/multvae_ml20m/wandb_seed42
```

This logs the objective, NLL, KL, weighted KL and beta annealing, plus the
trainer's Recall/NDCG at 10, 20, 50 and 100. `report.md` and PNG figures continue
to be generated locally by the existing trainer. They are **not uploaded**.

### Content projection

```bash
uv run --env-file .env --extra content python -m content_recsys train \
  --device mps --layout single --mode projection --loss bpr \
  --prepared artifacts/content_multvae_ml20m/prepared \
  --output-dir artifacts/content_multvae_ml20m/wandb_single_projection_bpr_seed42
```

Prepared data and frozen features must already exist. Training validates the
content-only binary profile; it does not run the separate hybrid/fusion evaluation
or perform test selection. Content final metrics live in `training.json` rather
than `metrics.json`.

## Modes and flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--wandb-mode` | `WANDB_MODE`, otherwise `online` | `online`, `offline` or `disabled` |
| `--wandb-entity` | `WANDB_ENTITY`, otherwise `ducanh250599` | Team or account |
| `--wandb-project` | `WANDB_PROJECT`, otherwise `RecSys` | Project |
| `--wandb-name` | Output directory name | Display name, not the stable run ID |
| `--wandb-log-every` | `100` | Optimizer-step interval for batch logging |

Explicit CLI flags take precedence over environment variables. For example,
`--wandb-mode disabled` disables W&B even if `.env` contains `WANDB_MODE=online`.

```bash
# Record W&B files locally, without contacting the Cloud.
uv run --env-file .env python -m multvae.train \
  --output-dir artifacts/multvae_ml20m/offline_seed42 --wandb-mode offline

# Disable W&B; ordinary local checkpoints/history/metrics still work.
uv run --env-file .env python -m lightgcn.train \
  --output-dir artifacts/lightgcn_ml20m/local_seed42 --wandb-mode disabled
```

After an offline run, upload it manually. Replace the path below with the actual
`offline-run-*` directory printed by W&B or found under the experiment directory:

```bash
uv run --env-file .env wandb sync \
  artifacts/multvae_ml20m/offline_seed42/wandb/offline-run-YYYYMMDD_HHMMSS-SESSION_ID
```

No automatic sync command is executed by the repository.

## Metrics and uploaded artifacts

Batch metrics use cumulative `global_step` (optimizer updates); epoch metrics use
`epoch`. W&B's internal history row counter is separate from both. First and last
batches of each epoch are logged in addition to every 100 updates, so short epochs
still have batch observations. `--wandb-log-every` changes that interval.

| Namespace | Examples |
| --- | --- |
| `batch/*` | `batch/loss`, `batch/learning_rate`, memory; content gradient norm; VAE NLL/KL/beta; LightGCN BPR/regularization |
| `train/*` | Epoch loss, learning rate, VAE objective components and beta |
| `valid/*` | Recall/NDCG and evaluated user count; VAE standard errors |
| `timing/*` | Training/validation time where measured, content epoch time |
| `memory/*` | Process peak RSS, MPS allocation/driver memory or CUDA allocator memory where available |
| `selection/*` | Selected epoch, improvements, patience counters, phase where applicable |
| `final/*` | Existing final validation/test metrics, selected epoch, stop reason and timing |

Final values are also stored in run summary for comparison in the W&B runs table.
Not all models have the same cutoffs or timing fields; absent metrics are not
fabricated. Apple MPS memory comes from PyTorch; it is not NVIDIA GPU telemetry.
Content LoRA logs learning rates for both projection and adapter parameter groups.

On completion, a `training-metadata` artifact contains
only existing files from this allowlist:

- `config.json`
- `history.json`
- `metrics.json`
- `training.json`
- `progress.json`

The integration does not upload report Markdown, PNG figures, checkpoints,
embeddings, raw/prepared datasets or per-user predictions. On interruption before
completion, the local files remain; the final artifact upload might not occur.

## Resume for newly created checkpoints

### Training state and W&B identity

`last.pt` stores training recovery state and a small `tracking` dictionary. The
output directory also has `wandb_state.json` containing run identity and the last
logged batch/epoch cursors. Neither contains credentials. Keep the whole output
directory when moving or backing up a run.

In online mode, resume reconnects to the same run ID in the same team/project.
W&B settings are separate from model settings, so changing mode, display name or
logging frequency does not invalidate the training checkpoint. Changing the
team/project during resume disables W&B with a warning rather than appending to
an unrelated Cloud run. Start a new experiment to use a different team/project.

If a process stops after logging steps beyond its last saved checkpoint, training
replays those updates. The sidecar and resumed Cloud summary retain high-water
log cursors, so already logged batch/epoch positions are skipped. SDK-queued
records can still be lost if the machine is killed before writing/syncing them;
keeping checkpoints does not guarantee that every metric reached the Cloud.
There is no automatic import/backfill of old local history.

W&B ignores its native `resume` option in offline mode. Each offline invocation
therefore creates a **new session ID**, with the same logical experiment group
and cumulative step/epoch axes. Syncing sessions yields separate grouped runs,
not a promise to merge them into one Cloud history. Training recovery still works.
Switching from offline/disabled to online does not upload the earlier history.

### LightGCN example

Repeat the model/training settings and add `--resume`:

```bash
uv run --env-file .env python -m lightgcn.train \
  --device mps --batch-size 524288 --max-epochs 30 \
  --eval-every 2 --patience 5 --validation-users 20000 \
  --output-dir artifacts/lightgcn_ml20m/wandb_seed42 \
  --resume artifacts/lightgcn_ml20m/wandb_seed42/last.pt
```

LightGCN saves `last.pt` after each completed epoch. Resume restores current
weights, Adam state, NumPy sampling RNG, torch CPU/accelerator RNG, history,
warmup progress, checkpoint selection and early-stopping counters. An unfinished
epoch is rerun from the last completed epoch. The selected best checkpoint is
embedded in `last.pt` to recover interrupted writes to multiple files. This makes
`last.pt` larger than a weights-only checkpoint.

`--max-epochs` is the total main-epoch budget, excluding warmup; it can be increased
on resume. Other training settings and the resolved device must match. A run
already stopped by early stopping only repeats final evaluation; resume does not
reset patience. `--init-checkpoint best.pt` starts a **new** optimizer/experiment
and is mutually exclusive with `--resume`. Old LightGCN checkpoints do not have
the new recovery state and cannot be used as `--resume` inputs.

### Mult-VAE example

```bash
uv run --env-file .env python -m multvae.train \
  --device mps --max-epochs 200 --patience 10 \
  --output-dir artifacts/multvae_ml20m/wandb_seed42 \
  --resume artifacts/multvae_ml20m/wandb_seed42/last.pt
```

Recovery uses the existing completed-epoch protocol: Adam, RNG, annealing update
count, best checkpoint, history and stopping state. Repeat any nondefault settings
from the original command. Increasing the total epoch budget is supported; an
early-stopped run remains stopped.

### Content example

Run the **same training command** with the same output directory. The trainer
automatically loads its `last.pt`. Content checkpoints are saved during an epoch
as well as at epoch boundaries. Preserve `epoch_bank.npy` alongside the checkpoint
to resume within an epoch after interruption. An unfinished batch interval is
replayed from the last checkpoint.

Content still checks exact training config, prepared/frozen fingerprints, code
hash and dependency lock. Do not change source/lock or move absolute paths while
resuming. This integration does not migrate checkpoints created before it was
installed. A directory with `training.json` is already completed: rerunning the
command returns cached results without starting another W&B session.

## Vast.ai setup and operation

Update the repository checkout on the GPU instance, including `training_logging/`,
`pyproject.toml` and `uv.lock`. Transfer the prepared inputs and frozen feature
cache separately. An older uploaded copy does not gain these changes automatically.
With a compatible CUDA environment already provisioned, install the locked
dependencies manually from that checkout:

```bash
uv sync --locked --extra content
```

Supply `WANDB_API_KEY` on the GPU instance separately,
for example by entering it in a private credentials file with permissions `600`:

```text
# /workspace/wandb.env on the GPU instance
WANDB_API_KEY=YOUR_KEY
WANDB_ENTITY=ducanh250599
WANDB_PROJECT=RecSys
```

Load that file explicitly for the training command from the repository root:

```bash
uv run --env-file /workspace/wandb.env --extra content python -m content_recsys train \
  --device cuda --layout single --mode lora --loss bpr \
  --prepared artifacts/content_multvae_ml20m/prepared \
  --output-dir artifacts/content_multvae_cuda/wandb_single_lora_bpr_seed42
```

This command enables W&B online by default. Add `--wandb-mode offline` or
`--wandb-mode disabled` when needed, and use a **new** `--output-dir` for a new
run. The prepared model's pinned revision must be available in the Hugging Face
cache or downloadable on the instance. Setting Hugging Face to offline for cached
model loading does not disable W&B network logging. These are manual setup
examples; this integration does not provision the instance or launch training.

Online mode requires outbound access to W&B Cloud. Save the entire experiment
directory, including `wandb/`, `wandb_state.json`, `last.pt` and `epoch_bank.npy`,
before deleting the instance. Cloud metrics do not replace a checkpoint backup.

## Failure behavior

SDK import/initialization, metric logging, artifact upload and shutdown errors are
reported to stderr. Training and its normal local outputs continue. Initialization
has a 15-second timeout; individual Cloud requests use a 10-second timeout and
at most two retries on the pinned SDK. Shutdown drains the SDK queue and has no
hard overall deadline. No interactive authentication or automatic offline retry is performed.
After a logging error, further W&B logging in that invocation is disabled; resume
can reconnect later. Missing credentials or wrong permissions can therefore leave
a successful training run with no Cloud metrics. Check for `wandb_start` and the
run URL in training output when online logging is expected.

If W&B was initialized before a network interruption, its own local buffering can
retain records. That differs from an initialization failure, which records no W&B
history. Existing JSON history is still available in either case, but this change
does not automatically upload it later. Actual training/data errors continue to
raise normally and are not suppressed by the logging wrapper.

## Future design: search, retrain, continuation and evaluation

**This section proposes a common policy across model families.** No W&B Sweep,
orchestrator logger or standalone evaluation logger has been added. The separate
Mult-VAE objective experiment commands already implement some per-trial training
logging and continuation behavior; their guide describes the available commands.

### Search: one run per trial, one group per search

Suppose a content projection search compares learning rates `0.0003` and `0.001`
at temperature `0.1`, with BPR and seed 42. Use a group such as
`content-projection-search-2026-10-09` and two training trial runs:

| Run name | Config | Selection metric |
| --- | --- | --- |
| `single_projection_bpr_lr0.0003_tau0.1_seed42` | `learning_rate=0.0003`, `temperature=0.1`, `seed=42` | Best content-only validation Recall@20 |
| `single_projection_bpr_lr0.001_tau0.1_seed42` | `learning_rate=0.001`, `temperature=0.1`, `seed=42` | Same validation protocol |

An optional orchestration run would record completed/failed trial counts and the
selected trial ID. Seed-43/44 replications would carry the winning configuration's
ID plus their own seed. Sort validation summary columns to compare trials; do not
use test metrics to select the winner. The repository's existing fixed search can
stay fixed: using native W&B Sweeps would be a separate change.

### Retrain: distinguish scratch and fine-tune

For the existing LightGCN `train+valid` comparison, use two child runs in one group:

- `lightgcn-retrain-scratch`: fresh weights/Adam, six main epochs, last checkpoint.
- `lightgcn-retrain-finetune`: selected train-only checkpoint as initialization,
  fresh Adam, three warmup epochs, up to ten main epochs, training-loss stopping.

Both configs would record `train_split=train-valid`, their selection criterion,
initialization type and source checkpoint hash/run ID where applicable. The parent
comparison would show final test Recall/NDCG after selection. No validation score
would be logged for selecting a model trained on validation interactions. Different
selection protocols must remain visible in the comparison table.

### Continuation: resume versus a branch from selected weights

Example A: a run stops during epoch 8; `last.pt` is from epoch 7. Reconnect to the
same online run, restore Adam/RNG/patience, replay epoch 8, and skip metric positions
already logged. This is the current standalone resume behavior.

Example B: training reached epoch 10, but `best.pt` is from epoch 6. Continuing from
that selected checkpoint follows a different trajectory. Create a new child run,
record `parent_run_id`, `source_checkpoint_sha256`, `source_epoch=6`, and the
restored optimizer/selection protocol. Keep cumulative epoch/step axes and a
separate `continued_epoch` starting at 1. The existing `continue_finetune` command
would need this explicit lineage tracking before it is covered by W&B.

### Standalone evaluation: one checkpoint, explicit protocol

Suppose a selected content checkpoint is evaluated with a content-only scorer and
a hybrid scorer. Create an evaluation run linked to the source training run and
embedding/checkpoint hash, with `job_type=evaluate`. Record the prepared-data
fingerprint, split, user population, profile (`binary` or `rating`), cutoffs and
masking/input policy. Example summary column names:

```text
evaluation/content/binary/recall@20
evaluation/hybrid/binary/recall@20
evaluation/hybrid/binary/alpha
evaluation/multvae/binary/recall@20
```

On validation, log the alpha grid and selected alpha. On test, use the immutable
validation selection lock and record its hash; do not tune alpha on test. Repeating
an evaluation with a different protocol should create a new evaluation run so it
does not overwrite the training run's original final metrics. The separate
LightGCN and Mult-VAE evaluation commands would follow the same source-lineage
pattern with their own explicit input/masking conventions.

## Verification

Implementation checks used temporary CPU fixtures (8 users, 24 items), never the
full MovieLens training pipeline. Uninterrupted and resumed LightGCN/Mult-VAE
runs matched weights, Adam state and loss history; content projection BPR/InfoNCE
matched after a pause inside an epoch. The actual W&B offline SDK recorded metrics
and only allowlisted JSON artifacts. Syntax, CLI flags, replay cursors and nonfatal
missing-key/SDK-failure behavior were checked without running unit tests. Live
Cloud authentication, CUDA/LoRA training and accelerator resume remain unverified.

## References

- [W&B Python SDK](https://docs.wandb.ai/models/ref/python/functions)
- [W&B modes](https://docs.wandb.ai/ref/cli/wandb-init)
- [Offline logging and manual sync](https://docs.wandb.ai/ja/support/models/articles/can-i-run-wandb-offline)
- [SDK resume/offline behavior](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_init.py)
