# Decoder-logit BPR with Mult-VAE

The model retains the Mult-VAE architecture and full binary train histories.
Inputs are L2-normalized and use the existing dropout during training. The
reconstruction target remains the complete train history, including the sampled
BPR positive. No selected positive is intentionally removed from the input.

For each user and epoch, preselect one uniformly sampled observed train item and
16 uniformly sampled items absent from that user's train history. Negatives are
sampled with replacement; duplicates are allowed. Validation/test labels do not
filter these samples. Separate NumPy RNG streams keep traversal and ranking
pairs matched between objective configurations with the same seed/epoch.

The total objective is:

```text
L = nll_weight * mean_users(-sum_items(history * log_softmax(logits)))
  + bpr_weight * mean_users(mean_negatives(softplus(logit_neg - logit_pos)))
  + beta(step) * mean_users(KL(q(z | history) || N(0, I)))
beta(step) = min(anneal_cap, completed_optimizer_steps / anneal_steps)
```

Decoder logits are relevance scores suitable for BPR; no additional sigmoid or
softmax is needed. Log-softmax differences equal logit differences because the
per-user normalizer cancels. The auxiliary BPR term backpropagates through the
sampled latent into both decoder and encoder. This modifies the original VAE
likelihood objective. Full-history reconstruction and BPR share their observed
positives, so BPR may be redundant; quality is measured on unseen held-out
targets with all known items masked, rather than inferred from training loss.
Unobserved items are not confirmed dislikes.

## Short objective grid on MPS

```bash
uv run --env-file .env python -m multvae.search \
  --output-dir artifacts/multvae_objectives_mps_seed42 \
  --max-epochs 10 --patience 3 --anneal-epochs 5 --wandb-mode online
```

The ten runs are the Cartesian product of NLL weights `{0, 1}`, BPR weights
`{0, 1, 10}`, and KL caps `{0, 0.2}`, excluding both zero-reconstruction
configurations. This provides no-KL controls, NLL-only controls, BPR-only
controls, and combined-objective runs. It does not optimize KL alone. Every run
starts from scratch at seed 42, using the same architecture, batch size 500,
Adam learning rate 0.001 and maximum epoch budget. Runs execute sequentially on
MPS. All eight Recall/NDCG metrics are logged to W&B online, along with raw and
weighted NLL/BPR/KL and beta. Logging must initialize successfully for each run.

At the default dataset size there are 260 batches/epoch. The search uses 6,500
denominator steps, reaching beta 0.2 after 1,300 completed updates, at the start
of epoch six. This faster schedule allows KL to be observed within ten epochs;
the existing historical baseline used 200,000 denominator steps and 177 epochs.
Historical baseline metrics therefore do not form a controlled comparison.
Standalone training keeps the original annealing defaults.

Early stopping starts immediately, with patience three, and checkpoint
selection maximizes full-validation Recall@20. Equal scores retain the earlier
checkpoint. Search breaks exact inter-run Recall ties using NDCG@20. Test is
evaluated only for the global validation winner, after saving the selection
lock; train+validation input and masking match the existing evaluator.

The root `report.md`, `results.csv`, `summary.json` and `objective_grid.png`
summarize ranking quality, raw/weighted terms and matched objective ablations.
`plan.json` records source hashes, input hashes and budgets. Each run retains
`best.pt`, `last.pt`, logs and its English report. Repeating the same search
continues interrupted runs from the last completed epoch, preserving optimizer,
sampler, RNG and stopping state, and skips completed runs. Changed source or
budgets require a new search directory. An interruption before the first
completed epoch requires inspecting that unfinished run before retrying.

A single combined-objective run can use:

```bash
uv run --env-file .env python -m multvae.train --device mps \
  --nll-weight 1 --bpr-weight 10 --negative-count 16 \
  --anneal-cap 0.2 --anneal-steps 6500 \
  --max-epochs 10 --patience 3 --early-stop-start immediate \
  --skip-test --wandb-mode online \
  --output-dir artifacts/multvae_ml20m/nll_bpr_kl_mps
```

NLL scales with history length whereas BPR averages pairs. Coefficients are
explicit experiment settings, not automatic scale matching. One positive per
user gives sparse ranking supervision. Ten epochs and one seed provide an
initial feasibility study, not evidence of a converged optimum or significance
across training seeds. Further negative counts, BPR coefficients, KL caps and
training budgets can be explored after assessing this controlled grid.

Verification uses Python syntax checks, CPU/MPS operational smoke training,
finite-gradient checks, exact checkpoint continuation on a small synthetic
dataset, manifest audits and actual held-out ranking. No unit-test suite is
added or run, following AGENTS.md.

References: [BPR](https://arxiv.org/abs/1205.2618),
[Mult-VAE](https://arxiv.org/abs/1802.05814).

## Promote four configurations to 50 epochs, then one to 200

```bash
uv run --env-file .env python -m multvae.extend_search \
  --source-dir artifacts/multvae_objectives_mps_seed42 \
  --output-dir artifacts/multvae_objectives_extended_mps_seed42 \
  --patience200 10 --wandb-mode online
```

The top four ten-epoch configurations are ranked by full-validation Recall@20,
using NDCG@20 for exact inter-run ties. Each continues from `last.pt` to at most
50 **total** epochs, retaining Adam, RNG, history and the global best checkpoint.
The winning configuration then continues to at most 200 total epochs. Promoted
checkpoints and fresh W&B identities are stored separately; source grid files
and original W&B runs are preserved.

At both stage transitions, KL annealing restarts from beta zero. Global steps
continue to increase; a saved `kl_anneal_start_step` offset defines
`beta = min(cap, (global_step - kl_anneal_start_step) / 6500)`. Cap 0.2 is reached
after 1,300 new updates (five new epochs). Resuming an interrupted stage keeps
this offset and does not restart annealing again. Sampling still uses the total
epoch number, so neither user traversal nor sampled pairs repeat from epoch one.

Each stage also starts a new patience window: three checks for stage 50 and ten
checks for stage 200. For nonzero KL, early stopping begins at the annealing cap;
the zero-KL control begins immediately. All epochs, including warmup, can provide
the best validation checkpoint. Early stopping can complete a stage before its
maximum budget. Promotion and checkpoint selection do not consult test scores;
the final stage-200 selected checkpoint is evaluated on test after selection.

For manual continuation, the trainer provides `--restart-kl-annealing` and
`--restart-early-stopping` with `--resume`. Use these only at an intentional stage
transition; an ordinary interrupted-run resume should omit both flags. Changing
patience or `--early-stop-start` on resume requires the explicit stopping restart.

## Fresh hybrid versus paper baseline, up to 200 epochs

```bash
uv run --env-file .env python -m multvae.compare_scratch --wandb-mode online
```

This retrains two independent models with fresh weights, Adam and seed 42:
the selected hybrid (`NLL=1`, `BPR=1`, KL cap 0.2, anneal denominator 6,500)
and the paper Mult-VAE objective (`NLL=1`, `BPR=0`, KL cap 0.2, denominator
200,000). Only hyperparameters are read from the earlier configuration files;
earlier checkpoints are never loaded. Both use the same full-history input,
architecture, optimizer, batches and existing audited splits on MPS. The shared
trainer logs diagnostic sampled BPR for the baseline with zero loss weight.

Each run has a 200-epoch maximum, patience ten after reaching the KL cap, and
validation Recall@20 checkpoint selection. Annealing starts at zero at epoch one
and runs continuously, without the prior promotion-stage resets. Test is
evaluated independently for each selected checkpoint after its validation
selection, using train+validation input and masking. Repeating the command only
resumes an interrupted run within this new experiment, preserving its original
zero annealing offset and online W&B identity.

Results, all eight Recall/NDCG metrics, differences, curves, verification and
both checkpoints are saved under
`artifacts/multvae_scratch_comparison_mps_seed42/`. The baseline follows the
paper architecture/objective/schedule on the repository's per-user splits; it
does not reproduce the paper's separate held-out-user protocol. Since BPR and
KL schedule differ together, this comparison cannot isolate BPR's effect. A
matched-annealing ablation and additional seeds would be needed for that claim.

## Fresh BPR-only objective with annealed KL

The following trains with NLL weight zero, BPR weight one, 16 negatives, KL cap
0.2 and a 100,000-update annealing denominator. All train history remains in
the encoder input. NLL is logged as a diagnostic with zero loss contribution.

```bash
uv run --env-file .env python -m multvae.train \
  --device mps --nll-weight 0 --bpr-weight 1 --negative-count 16 \
  --anneal-cap 0.2 --anneal-steps 100000 \
  --max-epochs 200 --patience 10 --early-stop-start anneal-cap \
  --seed 42 --test-input train-valid --wandb-mode online \
  --output-dir artifacts/multvae_bpr_kl_anneal100000_scratch_mps_seed42
```

Use an empty output directory to start from fresh weights and Adam. For the
earlier fast-annealing configuration, use `--anneal-steps 6500` and a separate
output directory. The schedule is `beta = min(0.2, step / anneal_steps)`, so
100,000 denominator steps reach cap after 20,000 updates, in epoch 77 with
260 batches per epoch. Early stopping starts after that cap; checkpoint
selection includes warmup epochs.

Measured seed-42 results on the existing audited MovieLens splits:

| Objective | Anneal steps | Epochs completed | Best epoch | Test Recall@20 | Test NDCG@20 | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NLL + BPR + KL | 6,500 | 85 | 75 | 0.348364 | 0.326088 | [Run](https://wandb.ai/ducanh250599/RecSys/runs/012b89845f73) |
| NLL + KL baseline | 200,000 | 177 | 167 | 0.352817 | 0.335151 | [Run](https://wandb.ai/ducanh250599/RecSys/runs/a8cb85c8c9e5) |
| BPR + KL | 6,500 | 35 | 25 | 0.148492 | 0.140236 | [Run](https://wandb.ai/ducanh250599/RecSys/runs/e1c1886c1c17) |
| BPR + KL | 100,000 | 87 | 4 | 0.170111 | 0.157059 | [Run](https://wandb.ai/ducanh250599/RecSys/runs/838c65ae92c8) |

All four runs start from scratch, have a 200-epoch maximum and patience ten
after the KL cap, and select checkpoints by validation Recall@20. The two
BPR+KL runs differ only in the annealing denominator. Slower annealing improves
their selected-checkpoint test Recall@20 by 14.56% and NDCG@20 by 12.00% in this
seed. The 100,000-step run selects epoch four at beta 0.01039; later validation
declines as beta increases. These results do not establish variation across
seeds or isolate BPR's effect in the NLL+KL baseline comparison.

Datasets and generated checkpoints/reports remain local under ignored `data/`
and `artifacts/`. The staged and comparison orchestration commands require the
earlier run configurations/checkpoints at their documented source paths; the
standalone command above does not require an earlier model checkpoint.
