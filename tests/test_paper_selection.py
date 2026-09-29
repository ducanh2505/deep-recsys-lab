from __future__ import annotations

import pytest

from deep_recsys_lifecycle.paper_selection import (
    both_modes_reliable_gain,
    paired_recall_interval,
    select_validation_runs,
)


def _rows(recalls: tuple[float, ...], *, ndcg_at_20: float = 0.5) -> dict[str, dict[str, float]]:
    return {
        str(subject_id): {"Recall@100": recall, "NDCG@20": ndcg_at_20}
        for subject_id, recall in enumerate(recalls, start=1)
    }


def test_paired_subject_bootstrap_preserves_hand_worked_constant_delta() -> None:
    # Each Subject improves by 0.1. Independently resampling the two systems
    # would create spurious uncertainty because their starting recalls differ.
    baseline = _rows((0.1, 0.8))
    selected = _rows((0.2, 0.9))

    interval = paired_recall_interval(baseline, selected, seed=17, draws=128)

    assert interval["subject_count"] == 2
    assert interval["baseline_recall_at_100"] == pytest.approx(0.45)
    assert interval["variant_recall_at_100"] == pytest.approx(0.55)
    assert interval["absolute_delta"] == pytest.approx(0.1)
    assert interval["ci_lower"] == pytest.approx(0.1)
    assert interval["ci_upper"] == pytest.approx(0.1)
    assert interval["reliable_gain"] is True
    assert interval["conclusion"] == "reliable_gain"
    assert interval["seed"] == 17
    assert interval["draws"] == 128
    assert interval["resampling_unit"] == "Subject"
    assert interval["bootstrap_method"] == "paired_subject_with_replacement"
    assert interval["quantile_method"] == "linear"


def test_mixed_subject_effects_have_inconclusive_paired_interval() -> None:
    baseline = _rows((0.0, 1.0, 0.0, 1.0))
    selected = _rows((1.0, 0.0, 1.0, 0.0))

    first = paired_recall_interval(baseline, selected, seed=9, draws=1_000)
    replay = paired_recall_interval(baseline, selected, seed=9, draws=1_000)

    assert first == replay
    assert first["absolute_delta"] == 0.0
    assert first["ci_lower"] < 0.0 < first["ci_upper"]
    assert first["reliable_gain"] is False
    assert first["conclusion"] == "inconclusive"


def test_paired_comparison_rejects_subject_mismatch_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="identical Subject IDs"):
        paired_recall_interval(_rows((0.1, 0.2)), {"1": {"Recall@100": 0.3}})
    with pytest.raises(ValueError, match="invalid Recall@100"):
        paired_recall_interval(_rows((0.1,)), _rows((float("nan"),)))
    with pytest.raises(ValueError, match="bootstrap draws"):
        paired_recall_interval(_rows((0.1,)), _rows((0.2,)), draws=1)
    with pytest.raises(ValueError, match="bootstrap seed"):
        paired_recall_interval(_rows((0.1,)), _rows((0.2,)), seed=True)
    with pytest.raises(ValueError, match="bootstrap seed"):
        paired_recall_interval(_rows((0.1,)), _rows((0.2,)), seed=-1)


def test_validation_selector_prioritizes_distinguishable_recall_gain() -> None:
    runs = {
        "baseline": _rows((0.2, 0.2, 0.2, 0.2), ndcg_at_20=0.9),
        "variant": _rows((0.6, 0.6, 0.6, 0.6), ndcg_at_20=0.1),
    }

    selection = select_validation_runs("baseline", runs, seed=4, draws=128)

    assert selection["selected_id"] == "variant"
    assert selection["baseline_retained"] is False
    assert selection["decision_reason"] == "highest_validation_recall_distinguishable"
    assert selection["paired_recall_tie_ids"] == ["variant"]
    assert selection["selection_metric"] == "final_lhf.Recall@100"


def test_validation_selector_uses_ndcg_for_uncertain_recall_gain() -> None:
    runs = {
        "baseline": _rows((0.0, 1.0, 0.0, 1.0), ndcg_at_20=0.3),
        "variant": _rows((1.0, 0.0, 1.0, 0.2), ndcg_at_20=0.7),
    }

    selection = select_validation_runs("baseline", runs, seed=4, draws=512)

    assert selection["selected_id"] == "variant"
    assert selection["decision_reason"] == "highest_validation_recall_won_paired_tie"
    assert selection["paired_recall_tie_ids"] == ["baseline", "variant"]
    assert selection["paired_tie_break_metric"] == "final_lhf.NDCG@20"


def test_validation_selector_retains_baseline_without_recall_gain() -> None:
    runs = {
        "baseline": _rows((0.4, 0.6), ndcg_at_20=0.2),
        "better_ndcg_only": _rows((0.4, 0.6), ndcg_at_20=0.9),
        "lower_recall": _rows((0.3, 0.6), ndcg_at_20=1.0),
    }

    selection = select_validation_runs("baseline", runs, seed=4, draws=64)

    assert selection["selected_id"] == "baseline"
    assert selection["baseline_retained"] is True
    assert selection["decision_reason"] == "baseline_no_validation_recall_gain"


def test_validation_selector_has_deterministic_id_fallback() -> None:
    baseline = _rows((0.0, 1.0, 0.0, 1.0), ndcg_at_20=0.2)
    tied = _rows((1.0, 0.0, 1.0, 0.2), ndcg_at_20=0.8)
    runs = {"zeta": tied, "baseline": baseline, "alpha": tied}

    first = select_validation_runs("baseline", runs, seed=4, draws=128)
    reversed_order = select_validation_runs(
        "baseline", dict(reversed(tuple(runs.items()))), seed=4, draws=128
    )

    assert first == reversed_order
    assert first["selected_id"] == "alpha"
    assert first["paired_recall_tie_ids"] == ["alpha", "baseline", "zeta"]


def test_selector_rejects_subject_mismatch_and_invalid_ndcg() -> None:
    with pytest.raises(ValueError, match="identical Subject IDs"):
        select_validation_runs(
            "baseline", {"baseline": _rows((0.1, 0.2)), "variant": _rows((0.3,))}
        )
    with pytest.raises(ValueError, match="invalid NDCG@20"):
        select_validation_runs(
            "baseline", {"baseline": _rows((0.1,)), "variant": _rows((0.2,), ndcg_at_20=1.2)}
        )


def test_joint_claim_needs_both_mode_specific_positive_lower_bounds() -> None:
    reliable = paired_recall_interval(_rows((0.2, 0.3)), _rows((0.4, 0.5)), draws=64)
    inconclusive = paired_recall_interval(_rows((0.2, 0.3)), _rows((0.2, 0.3)), draws=64)

    assert reliable["ci_lower"] > 0.0
    assert inconclusive["ci_lower"] == 0.0
    assert inconclusive["reliable_gain"] is False
    assert both_modes_reliable_gain({"known_user": reliable, "history_only": reliable}) is True
    assert both_modes_reliable_gain({"known_user": reliable, "history_only": inconclusive}) is False
    with pytest.raises(ValueError, match="both paper query modes"):
        both_modes_reliable_gain({"known_user": reliable})
