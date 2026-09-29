"""Paired, validation-only selection for the two paper benchmark modes.

Call this module separately for Known-User and History-Only. Their Recall formulas
are different, so this module never combines Subject rows across modes. Callers
must verify the frozen cohort, catalog, and Gold Set fingerprints before passing
per-Subject metrics here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, TypedDict

import numpy as np

DEFAULT_BOOTSTRAP_DRAWS = 10_000
_MAX_BOOTSTRAP_CELLS = 1_048_576
_CUTOFF_LOWER = 0.025
_CUTOFF_UPPER = 0.975


class PairedRecallInterval(TypedDict):
    subject_count: int
    baseline_recall_at_100: float
    variant_recall_at_100: float
    absolute_delta: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    draws: int
    seed: int
    resampling_unit: str
    bootstrap_method: str
    quantile_method: str
    reliable_gain: bool
    conclusion: Literal["reliable_gain", "reliable_decrease", "inconclusive"]


def _validate_bootstrap_options(seed: int, draws: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 2:
        raise ValueError("bootstrap draws must be an integer of at least two")


def _subject_ids(
    baseline: Mapping[str, Mapping[str, float]],
    variant: Mapping[str, Mapping[str, float]],
) -> tuple[str, ...]:
    baseline_ids = set(baseline)
    if not baseline_ids:
        raise ValueError("paired comparison requires at least one eligible Subject")
    if baseline_ids != set(variant):
        raise ValueError("paired comparison requires identical Subject IDs")
    if any(not isinstance(subject_id, str) or not subject_id for subject_id in baseline_ids):
        raise ValueError("Subject IDs must be non-empty strings")
    return tuple(sorted(baseline_ids))


def _metric(row: Mapping[str, float], name: str, subject_id: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Subject {subject_id} is missing numeric {name}")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"Subject {subject_id} has invalid {name}")
    return resolved


def _run_average(
    rows: Mapping[str, Mapping[str, float]], subject_ids: tuple[str, ...], name: str
) -> float:
    total = math.fsum(_metric(rows[subject_id], name, subject_id) for subject_id in subject_ids)
    return total / len(subject_ids)


def _bootstrap_bounds(differences: np.ndarray, *, seed: int, draws: int) -> tuple[float, float]:
    """Bootstrap paired deltas without allocating a draws-by-Subjects matrix.

    Each draw resamples Subject indices with replacement. A bounded block of
    indices and sampled deltas is held at once, even for a very large cohort.
    Memory grows with the Subject vector and the one-dimensional draw results.
    """

    if np.all(differences == 0.0):
        return 0.0, 0.0
    subject_count = len(differences)
    generator = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    batch_size = min(draws, 64, max(1, _MAX_BOOTSTRAP_CELLS // subject_count))
    for first_draw in range(0, draws, batch_size):
        count = min(batch_size, draws - first_draw)
        totals = np.zeros(count, dtype=np.float64)
        remaining = subject_count
        while remaining:
            width = min(remaining, _MAX_BOOTSTRAP_CELLS // count)
            indices = generator.integers(0, subject_count, size=(count, width))
            totals += differences[indices].sum(axis=1)
            remaining -= width
        means[first_draw : first_draw + count] = totals / subject_count
    lower, upper = np.quantile(means, (_CUTOFF_LOWER, _CUTOFF_UPPER), method="linear")
    return float(lower), float(upper)


def paired_recall_interval(
    baseline: Mapping[str, Mapping[str, float]],
    variant: Mapping[str, Mapping[str, float]],
    *,
    seed: int = 42,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> PairedRecallInterval:
    """Return a paired Subject-bootstrap interval for absolute Recall@100 change.

    Inputs must be from the same protocol mode and frozen eligible cohort. Only
    Subject IDs are checked here; the caller must also check Gold Set and catalog
    fingerprints. A lower bound of exactly zero is not a reliable gain.
    """

    _validate_bootstrap_options(seed, draws)
    subject_ids = _subject_ids(baseline, variant)
    baseline_scores = np.fromiter(
        (_metric(baseline[subject_id], "Recall@100", subject_id) for subject_id in subject_ids),
        dtype=np.float64,
        count=len(subject_ids),
    )
    variant_scores = np.fromiter(
        (_metric(variant[subject_id], "Recall@100", subject_id) for subject_id in subject_ids),
        dtype=np.float64,
        count=len(subject_ids),
    )
    differences = variant_scores - baseline_scores
    lower, upper = _bootstrap_bounds(differences, seed=seed, draws=draws)
    reliable_gain = lower > 0.0
    conclusion: Literal["reliable_gain", "reliable_decrease", "inconclusive"]
    if reliable_gain:
        conclusion = "reliable_gain"
    elif upper < 0.0:
        conclusion = "reliable_decrease"
    else:
        conclusion = "inconclusive"
    return {
        "subject_count": len(subject_ids),
        "baseline_recall_at_100": float(baseline_scores.mean()),
        "variant_recall_at_100": float(variant_scores.mean()),
        "absolute_delta": math.fsum(float(value) for value in differences) / len(subject_ids),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence_level": 0.95,
        "draws": draws,
        "seed": seed,
        "resampling_unit": "Subject",
        "bootstrap_method": "paired_subject_with_replacement",
        "quantile_method": "linear",
        "reliable_gain": reliable_gain,
        "conclusion": conclusion,
    }


def select_validation_runs(
    baseline_id: str,
    runs: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    seed: int = 42,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
) -> dict[str, object]:
    """Select one mode's configuration using validation final-LHF metrics only.

    The largest point Recall@100 is the reference. Contenders whose paired
    Recall interval against it contains zero are tied; their NDCG@20 decides.
    A variant must have a positive point Recall gain over the baseline to be
    eligible, even if it has higher NDCG. Exact remaining ties prefer the
    baseline, then the lexicographically smallest run ID.
    """

    _validate_bootstrap_options(seed, draws)
    if baseline_id not in runs:
        raise ValueError("baseline run ID is missing")
    if not baseline_id:
        raise ValueError("baseline run ID must be non-empty")
    if any(not isinstance(run_id, str) or not run_id for run_id in runs):
        raise ValueError("run IDs must be non-empty strings")
    baseline = runs[baseline_id]
    subject_ids = _subject_ids(baseline, baseline)
    summaries: dict[str, dict[str, float]] = {}
    comparisons_to_baseline: dict[str, PairedRecallInterval] = {}
    for run_id in sorted(runs):
        rows = runs[run_id]
        _subject_ids(baseline, rows)
        summaries[run_id] = {
            "Recall@100": _run_average(rows, subject_ids, "Recall@100"),
            "NDCG@20": _run_average(rows, subject_ids, "NDCG@20"),
        }
        if run_id != baseline_id:
            comparisons_to_baseline[run_id] = paired_recall_interval(
                baseline, rows, seed=seed, draws=draws
            )

    baseline_recall = summaries[baseline_id]["Recall@100"]
    eligible = [
        run_id
        for run_id in sorted(runs)
        if run_id != baseline_id and summaries[run_id]["Recall@100"] > baseline_recall
    ]
    ties_to_top: dict[str, PairedRecallInterval] = {}
    tied_ids: tuple[str, ...]
    if not eligible:
        selected_id = baseline_id
        reference_id = baseline_id
        tied_ids = (baseline_id,)
        reason = "baseline_no_validation_recall_gain"
    else:
        reference_id = min(eligible, key=lambda run_id: (-summaries[run_id]["Recall@100"], run_id))
        tied = [reference_id]
        for run_id in sorted(runs):
            if run_id == reference_id or (run_id != baseline_id and run_id not in eligible):
                continue
            comparison = paired_recall_interval(
                runs[run_id], runs[reference_id], seed=seed, draws=draws
            )
            ties_to_top[run_id] = comparison
            if comparison["ci_lower"] <= 0.0 <= comparison["ci_upper"]:
                tied.append(run_id)
        tied_ids = tuple(sorted(tied))
        selected_id = min(
            tied_ids,
            key=lambda run_id: (
                -summaries[run_id]["NDCG@20"],
                0 if run_id == baseline_id else 1,
                run_id,
            ),
        )
        if len(tied_ids) == 1:
            reason = "highest_validation_recall_distinguishable"
        elif selected_id == baseline_id:
            reason = "baseline_won_paired_recall_tie"
        elif selected_id != reference_id:
            reason = "ndcg_at_20_won_paired_recall_tie"
        else:
            reason = "highest_validation_recall_won_paired_tie"

    return {
        "baseline_id": baseline_id,
        "selected_id": selected_id,
        "baseline_retained": selected_id == baseline_id,
        "decision_reason": reason,
        "reference_highest_recall_id": reference_id,
        "paired_recall_tie_ids": list(tied_ids),
        "validation_subject_count": len(subject_ids),
        "per_run_metrics": summaries,
        "paired_vs_baseline": comparisons_to_baseline,
        "paired_reference_vs_contender": ties_to_top,
        "selection_metric": "final_lhf.Recall@100",
        "paired_tie_break_metric": "final_lhf.NDCG@20",
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
    }


def both_modes_reliable_gain(
    intervals: Mapping[str, PairedRecallInterval],
) -> bool:
    """Require separate positive lower bounds for both paper protocols."""

    if set(intervals) != {"known_user", "history_only"}:
        raise ValueError("joint conclusion requires both paper query modes")
    return all(intervals[mode]["ci_lower"] > 0.0 for mode in ("known_user", "history_only"))
