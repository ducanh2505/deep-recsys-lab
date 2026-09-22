import json
from pathlib import Path

from deep_recsys_lifecycle.benchmark import summarize_latency_samples
from deep_recsys_lifecycle.report import write_rolling_report


def _metrics(*, query_count: int, covered: int, success: int) -> dict[str, int | float]:
    denominator = query_count or 1
    covered_denominator = covered or 1
    return {
        "query_count": query_count,
        "covered_query_count": covered,
        "ranking_success_count": success,
        "Coverage@200": covered / denominator if query_count else 0.0,
        "ConditionalRecall@10": success / covered_denominator if covered else 0.0,
        "EndToEndRecall@10": success / denominator if query_count else 0.0,
        "NDCG@10": success / denominator if query_count else 0.0,
    }


def _evaluation_summary() -> dict[str, object]:
    metrics = _metrics(query_count=2, covered=1, success=1)
    return {
        "query_mode": "known_user",
        "cohort_size": 2,
        "best_single_name": "popularity",
        "best_single_source": "prior_validation",
        "best_single": metrics,
        "oracle_union": metrics,
        "oracle_union_coverage": 0.5,
        "oracle_headroom_denominator": 0.0,
        "oracle_headroom_realized": -1.0,
        "retrievers": {"popularity": metrics, "itemknn": metrics},
        "rrf": metrics,
        "lhf": metrics,
        "history_segments": {
            "1-4": {"cohort_size": 0, "metrics": {}},
            "5-19": {"cohort_size": 1, "metrics": {"lhf": metrics}},
            "20+": {"cohort_size": 1, "metrics": {"lhf": metrics}},
        },
    }


def _benchmark_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "completed_with_errors",
        "artifact_id": "artifact-100",
        "artifact_cutoff_percentage": 100,
        "top_n": 10,
        "concurrency": 1,
        "warmup_count": 2,
        "requested_sample_count": 4,
        "percentile_method": "statistics.quantiles(method='inclusive')",
        "method": "in-process ASGI invocation; no TCP network latency",
        "environment": {
            "os": "Test OS",
            "cpu": "Test CPU",
            "python_version": "3.12.test",
        },
        "modes": {
            "known_user": {
                "measurement_status": "measured",
                "slo_status": "fail",
                "successful_sample_count": 4,
                "failed_sample_count": 0,
                "p50_ms": 210.0,
                "p95_ms": 220.0,
                "p99_ms": 221.0,
                "throughput_rps": 4.0,
                "slo_target_p95_ms": 200.0,
                "reason": None,
            },
            "history_only": {
                "measurement_status": "not_measured",
                "slo_status": "not_measured",
                "successful_sample_count": 0,
                "failed_sample_count": 1,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "throughput_rps": None,
                "slo_target_p95_ms": 200.0,
                "reason": "warm-up failed",
            },
            "empty_history": {
                "measurement_status": "measured",
                "slo_status": "pass",
                "successful_sample_count": 4,
                "failed_sample_count": 0,
                "p50_ms": 2.0,
                "p95_ms": 3.0,
                "p99_ms": 3.0,
                "throughput_rps": 500.0,
                "slo_target_p95_ms": 20.0,
                "reason": None,
            },
        },
    }


def test_latency_percentiles_and_throughput_use_successful_durations() -> None:
    summary = summarize_latency_samples(
        (0.001, 0.002, 0.003, 0.004),
        slo_target_p95_ms=3.85,
    )

    assert summary["p50_ms"] == 2.5
    assert summary["p95_ms"] == 3.85
    assert summary["p99_ms"] == 3.97
    assert summary["throughput_rps"] == 400.0
    assert summary["slo_status"] == "pass"
    assert summary["percentile_method"] == "statistics.quantiles(method='inclusive')"


def test_latency_summary_keeps_missing_samples_missing() -> None:
    summary = summarize_latency_samples((), slo_target_p95_ms=20.0)

    assert summary["p50_ms"] is None
    assert summary["p95_ms"] is None
    assert summary["p99_ms"] is None
    assert summary["throughput_rps"] is None
    assert summary["slo_status"] == "not_measured"


def test_rolling_report_embeds_aggregate_evidence_and_renders_failures(tmp_path: Path) -> None:
    path = tmp_path / "report.html"
    stages = [
        {
            "percentage": 90,
            "snapshot_event_count": 36,
            "artifact_id": "artifact-90",
            "active": False,
            "evaluation_window": [90, 100],
            "evaluation": {
                "known_user": _evaluation_summary(),
                "history_only": _evaluation_summary(),
            },
            "timings": {"training": 1.0, "export": 0.1, "smoke": 0.1, "evaluation": 0.2},
            "artifact_manifest": {
                "source_revision": "revision-90",
                "dataset_checksum": "dataset-90",
                "random_seed": 42,
                "data_cutoff_percentage": 90,
                "configuration": {
                    "known_user_fusion": {
                        "retriever_bank": ["popularity", "itemknn", "multivae", "lightgcn"]
                    },
                    "history_only_fusion": {
                        "retriever_bank": ["popularity", "itemknn", "multivae"]
                    },
                },
                "multivae_training": {
                    "actual_device": "cpu",
                    "fallback_reason": None,
                    "duration_seconds": 1.0,
                },
                "lightgcn_training": {
                    "actual_device": "cpu",
                    "fallback_reason": "MPS unavailable",
                    "duration_seconds": 1.0,
                },
            },
            "raw_rating_events": [{"event_id": "raw-secret-event"}],
            "raw_validation_pools": {"subject-1": [{"movie_id": 99}]},
        },
        {
            "percentage": 100,
            "snapshot_event_count": 40,
            "artifact_id": "artifact-100",
            "active": True,
            "evaluation_window": None,
            "evaluation": None,
            "timings": {"training": 1.0, "export": 0.1, "smoke": 0.1, "evaluation": 0.0},
            "artifact_manifest": {
                "source_revision": "revision-100",
                "dataset_checksum": "dataset-100",
                "random_seed": 42,
                "data_cutoff_percentage": 100,
                "configuration": {},
                "multivae_training": {"actual_device": "cpu"},
                "lightgcn_training": {"actual_device": "cpu"},
            },
        },
    ]
    write_rolling_report(
        path,
        stages=stages,
        source_event_count=40,
        dataset_checksum="dataset-checksum",
        active_artifact_id="artifact-100",
        source_revision="revision-100",
        configuration={"random_seed": 42, "profile": "rolling"},
        latency_benchmark=_benchmark_record(),
    )

    report = path.read_text(encoding="utf-8")
    marker_start = '<script id="embedded-report-data" type="application/json">'
    marker_end = "</script>"
    embedded = json.loads(report.split(marker_start, 1)[1].split(marker_end, 1)[0])

    assert 'id="executive-summary"' in report
    assert 'id="architecture-data-flow"' in report
    assert 'id="quality-known-user"' in report
    assert 'id="quality-history-only"' in report
    assert 'id="cold-user-quality"' in report
    assert 'id="latency-evidence"' in report
    assert 'id="failure-cases"' in report
    assert 'id="limitations"' in report
    assert 'id="provenance"' in report
    assert "90→100%" in report
    assert "Stage 100 has no Future Window" in report
    assert "warm-up failed" in report
    assert "not measured" in report
    assert "raw-secret-event" not in report
    assert "subject-1" not in report
    assert (
        embedded["stages"][0]["evaluation"]["known_user"]["history_segments"]["1-4"]["cohort_size"]
        == 0
    )
    empty_segment = embedded["stages"][0]["evaluation"]["known_user"]["history_segments"]["1-4"]
    assert empty_segment["metrics"] == {}
    assert embedded["latency_benchmark"]["modes"]["known_user"]["p95_ms"] == 220.0
