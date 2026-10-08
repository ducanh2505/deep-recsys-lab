# Content encoder + Mult-VAE benchmark

This experiment compares BPR and InfoNCE for personalized Top-20 recommendations.
It retains the audited MovieLens splits and the existing Mult-VAE checkpoint.
All experiment reports, tables and figure labels are in English. Focused device
tests use Python's built-in unittest; operational smoke runs precede the full benchmark.

## Run

Run commands from the repository root:

```bash
uv sync --extra content
uv run --extra content python -m content_recsys prepare
uv run --extra content python -m content_recsys smoke --mode projection
uv run --extra content python -m content_recsys search --stage projection
uv run --extra content python -m content_recsys report
```

`search` defaults to the projection stage. It also prepares inputs and runs smoke
verification when needed. Repeating the same command resumes compatible caches
and checkpoints. The default smoke checks projection only; `--mode all` explicitly
checks both training modes. Projection runs on the Mac; LoRA is deferred to a future GPU
stage at the user's request. The complete design still contains 32 search
configurations at seed 42 plus 16 replications at seeds 43/44 (48 training runs).
Each stage contains 16 search runs and eight replications.

The root `benchmark.json` tracks the active phase and run. Projection completes
at `waiting_for_lora_gpu` with `projection_completed=true`; this is successful
completion of the local stage, not completion of the full benchmark. Its
three-seed validation results, frozen controls, tables and figures are saved to
`projection_report.md` and `projection_summary.json`. Projection hyperparameters,
embedding hashes and fusion alphas are frozen in `projection_selection_lock.json`.
Learned-model test evaluation remains deferred until both stages finish and the
global selection lock is created. `report.md` remains explicitly partial.

## Deferred LoRA GPU stage

`deferred_lora_gpu.json` records the 16 LoRA search specifications, the eight-run
replication rule, fixed model revision, source hashes and unchanged training
budgets. It is a scheduling specification, **not an executable GPU launcher**.
No GPU resources are provisioned or jobs submitted. Partial Mac LoRA artifacts
are retained and do not count as a finished training run.

The executable CLI supports MPS/CPU/CUDA, including CUDA RNG checkpoint recovery
and allocator memory reporting. On the GPU host, map source/prepared/baseline/output
paths and run real-data CUDA smoke and resume checks before launching LoRA.
Start GPU runs in separate directories; do not resume an MPS checkpoint with a
different device/configuration. Copy and verify source data, the fixed baseline
checkpoint, and frozen feature caches. Preserve seed-paired sampling, float32,
batch/negative counts, epoch caps and validation-only selection. Record the GPU
environment and hashes separately; compare losses within the same hardware mode.

The explicit `search --stage all` option retains the original single-device
48-run orchestration for reproducibility. It can start LoRA on the selected
device, so it is not used for the deferred GPU workflow. Real-device CUDA execution
and combining the two stages still require verification on a GPU host.

Example CUDA LoRA smoke and single run (requires CUDA-enabled PyTorch):

```bash
uv run --extra content python -m content_recsys smoke --device cuda --mode lora \
  --output-dir artifacts/content_multvae_cuda
uv run --extra content python -m content_recsys train --device cuda \
  --layout single --mode lora --loss bpr --temperature 0.1 \
  --learning-rate 0.0001 --output-dir artifacts/content_multvae_cuda/lora_single_bpr
```

Run focused device tests with `uv run --extra content python -m unittest discover
-s tests -p 'test_content_cuda.py'`. CUDA hardware checks skip when no GPU is available.

Example single run:

```bash
uv run --extra content python -m content_recsys train \
  --layout single --mode projection --loss bpr --temperature 0.1 \
  --learning-rate 0.001 --output-dir artifacts/content_multvae_ml20m/manual_bpr
```

For custom output roots, pass `--prepared` to standalone `train`, `evaluate` and
`smoke`. A run directory resumes only with identical inputs/configuration/code.
No existing processed splits or baseline checkpoints are overwritten.

## Embeddings and objectives

Qwen3-Embedding-0.6B is pinned to revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, using float32, SDPA, left padding and
last-token pooling. Movie documents do not use a query instruction.

The single layout serializes metadata before plot text, capped at 512 tokens.
The multiview layout encodes plot/topic/people at 384/192/192 tokens, then
concatenates normalized blocks and structured context using square-root weights
0.50/0.25/0.15/0.10. Missing text views are zero; context includes missing numeric
indicators. Unverified TMDB mappings use MovieLens title/genres only.

Both learned modes project into normalized 512-dimensional vectors. LoRA trains
rank-8 q/v adapters with gradient checkpointing; the base weights stay frozen.
Binary histories train both objectives. Rating-minus-three profiles are evaluated
on the same learned vectors without additional training.

Each user supplies one positive per epoch, excluded from their profile. The
shared candidate pool contains batch positives and uniformly sampled items;
all known train positives are masked as negatives. Held-out labels never filter
training negatives. Both losses use cosine divided by temperature (0.05/0.10).
BPR averages softplus score differences per user; InfoNCE uses the positive plus
admitted negatives in a joint log-sum-exp denominator.

The full-history user profile is detached and uses an epoch-start memory bank.
Only current candidate embeddings receive gradients. The exact epoch-start bank
is retained for mid-epoch recovery; checkpoints are saved every 200 batches.

## Evaluation and selection

Checkpoint selection uses full-validation binary content Recall@20, then
NDCG@20, with earlier epochs retained on ties. Hyperparameter ties additionally
prefer the smaller learning rate and temperature. Fusion z-scores only unseen
items; alpha is searched from 0 to 1 in increments of 0.05. Raw endpoint scores
preserve exact content/Mult-VAE rankings.

Validation histories contain train; test histories contain train + validation.
The entire catalog is ranked, excluding seen items. The existing repository's
macro Recall/NDCG convention is reused. A selection lock records all embeddings,
fusion weights and the validation-selected deployment branch before any test
evaluation. Standalone test evaluation requires this lock:

```bash
uv run --extra content python -m content_recsys evaluate \
  --run-dir artifacts/content_multvae_ml20m/manual_bpr --split valid
```

The report includes seed standard deviations, paired 2,000-draw user bootstrap
intervals, frozen controls, baseline metrics, short/long-history groups, runtime,
memory and English figures. User bootstrap uncertainty is distinguished from
training-seed variability. No improvement over Mult-VAE is assumed.

References: [BPR](https://arxiv.org/abs/1205.2618),
[InfoNCE](https://arxiv.org/abs/1807.03748),
[SIGIR 2023 content objectives](https://arxiv.org/abs/2304.03112),
[Qwen3 Embedding](https://arxiv.org/abs/2506.05176).
