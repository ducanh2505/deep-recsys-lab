from __future__ import annotations

# The report deliberately keeps its self-contained HTML template inline; long HTML fragments are
# easier to audit as document text than as a chain of opaque template files.
# ruff: noqa: E501
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifact import ServingArtifact
from .evaluation import EvaluationReport, RetrieverEvaluation
from .event_store import DataSnapshot
from .positive import derive_positive_interactions
from .query import (
    EmptyHistoryQuery,
    HistoryOnlyQuery,
    KnownUserQuery,
    RecommendationService,
)
from .temporal import TemporalSplit


def write_static_report(
    path: Path,
    snapshot: DataSnapshot,
    artifact: ServingArtifact,
    *,
    temporal_split: TemporalSplit | None = None,
    evaluation: EvaluationReport | None = None,
    history_only_evaluation: EvaluationReport | None = None,
    empty_history_evaluation: EvaluationReport | None = None,
) -> None:
    """Write a self-contained HTML report for the temporal evaluation stage."""

    threshold = float(artifact.manifest["implicit_signal_threshold"])
    positive_count = len(derive_positive_interactions(snapshot.events, threshold=threshold))
    service = RecommendationService(artifact)
    examples = []
    if artifact.model.subject_histories:
        subject_id = min(artifact.model.subject_histories)
        examples.append(service.recommend(KnownUserQuery(subject_id=subject_id)).to_dict())
    if artifact.model.catalog:
        examples.append(
            service.recommend(HistoryOnlyQuery(movie_ids=(artifact.model.catalog[0],))).to_dict()
        )
    examples.append(service.recommend(EmptyHistoryQuery()).to_dict())
    future_event_count = (
        len(temporal_split.future_window_events) if temporal_split is not None else 0
    )
    snapshot_boundary = (
        temporal_split.snapshot_boundary if temporal_split is not None else snapshot.event_count
    )
    future_boundary = (
        temporal_split.future_boundary if temporal_split is not None else snapshot.event_count
    )
    embedded_data = {
        "snapshot": snapshot.to_dict(),
        "data_snapshot": snapshot.to_dict(),
        "future_window": {
            "event_count": future_event_count,
            "snapshot_boundary": snapshot_boundary,
            "future_boundary": future_boundary,
        },
        "artifact_manifest": artifact.manifest,
        "positive_interaction_count": positive_count,
        "query_examples": examples,
        "evaluation": evaluation.to_dict() if evaluation is not None else None,
        "evaluations": {
            "known_user": evaluation.to_dict() if evaluation is not None else None,
            "history_only": (
                history_only_evaluation.to_dict() if history_only_evaluation is not None else None
            ),
            "empty_history": (
                empty_history_evaluation.to_dict() if empty_history_evaluation is not None else None
            ),
        },
        "multivae_resource_evidence": artifact.manifest.get("multivae_training", {}),
        "lightgcn_resource_evidence": artifact.manifest.get("lightgcn_training", {}),
        "fusion_training": artifact.manifest.get("fusion_training", {}),
    }
    data_json = html.escape(json.dumps(embedded_data, indent=2, sort_keys=True))
    evaluation_sections = _evaluation_sections(evaluation, history_only_evaluation)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html>\n"
        '<html lang="en">\n'
        '<head><meta charset="utf-8"><title>Movie Recommender Lifecycle Showcase</title>'
        "<style>body{font-family:system-ui;max-width:62rem;margin:2rem auto;line-height:1.5}"
        "code,pre{background:#f4f4f4;padding:.25rem}section{margin:2rem 0}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;"
        "padding:.45rem;text-align:left}th{background:#f4f4f4}</style></head>\n"
        "<body>\n"
        "<h1>Movie Recommender Lifecycle Showcase</h1>\n"
        "<p>Leakage-free evaluation compares query-mode Learned Hybrid Fusion (LHF), "
        "Popularity, ItemKNN, Mult-VAE, LightGCN, RRF, and the diagnostic Oracle Union.</p>\n"
        '<section id="data-snapshot"><h2>Data Snapshot 50%</h2>'
        f"<p>{snapshot.event_count} deduplicated Rating Events across "
        f"{snapshot.source_batch_count} append-only batches.</p>"
        f"<p>Fingerprint: <code>{html.escape(snapshot.fingerprint)}</code></p></section>\n"
        '<section id="future-window"><h2>Future Window 50–60%</h2>'
        f"<p>{future_event_count} Rating Events are withheld from fitting and used only for "
        f"Gold Candidates and metrics (boundaries {snapshot_boundary} and {future_boundary})."
        "</p></section>\n"
        '<section id="evaluation-cohort"><h2>Evaluation Cohort</h2>'
        f"<p>{evaluation.cohort.size if evaluation is not None else 0} deterministic Subject "
        "queries share the same snapshot history and Gold Candidate cohort.</p></section>\n"
        '<section id="positive-interactions"><h2>Positive Interactions</h2>'
        f"<p>{positive_count} Rating Events met the {threshold:.1f} threshold.</p></section>\n"
        f"{evaluation_sections}"
        f"{_empty_history_fast_section(empty_history_evaluation)}"
        f"{_multivae_resource_section(artifact)}"
        f"{_lightgcn_resource_section(artifact)}"
        '<section id="query-modes"><h2>Query modes</h2>'
        "<p>Known-User and History-Only routes serve their separate LHF banks; Empty-History "
        "uses Popularity. All routes exclude observed Movies and preserve artifact provenance. "
        "History-Only and Empty-History queries do not support LightGCN.</p>"
        f"<pre>{data_json}</pre></section>\n"
        "</body>\n</html>\n",
        encoding="utf-8",
    )


def _evaluation_sections(
    evaluation: EvaluationReport | None,
    history_only_evaluation: EvaluationReport | None,
) -> str:
    if evaluation is None:
        return (
            '<section id="popularity-metrics"><h2>Popularity metrics</h2>'
            "<p>Evaluation results are not available.</p></section>\n"
            '<section id="itemknn-metrics"><h2>ItemKNN metrics</h2>'
            "<p>Evaluation results are not available.</p></section>\n"
            '<section id="multivae-metrics"><h2>Mult-VAE metrics</h2>'
            "<p>Evaluation results are not available.</p></section>\n"
            '<section id="lightgcn-metrics"><h2>LightGCN metrics</h2>'
            "<p>Evaluation results are not available.</p></section>\n"
            '<section id="rrf-metrics"><h2>RRF metrics</h2>'
            "<p>Evaluation results are not available.</p></section>\n"
            '<section id="oracle-union"><h2>Oracle Union</h2>'
            "<p>Diagnostic coverage ceiling is not available.</p></section>\n"
        )

    history_section = (
        _mode_comparison(history_only_evaluation, "History-Only")
        if history_only_evaluation is not None
        else '<section id="history-only-comparison"><h2>History-Only comparison</h2>'
        "<p>History-Only evaluation is not available.</p></section>\n"
    )


    known_section = _mode_comparison(evaluation, "Known-User")
    rows = ""
    for name, retriever_evaluation in (
        *evaluation.retrievers.items(),
        ("best single", evaluation.best_single),
        ("RRF", evaluation.rrf),
        ("LHF", evaluation.lhf),
        ("Oracle Union", evaluation.oracle_union),
    ):
        if retriever_evaluation is None:
            continue
        metrics = retriever_evaluation.metrics
        display_name = {"multivae": "Mult-VAE", "lightgcn": "LightGCN"}.get(name, name)
        rows += (
            "<tr>"
            f"<td>{html.escape(display_name)}</td>"
            f"<td>{metrics.coverage_at_200:.4f}</td>"
            f"<td>{metrics.conditional_recall_at_10:.4f}</td>"
            f"<td>{metrics.end_to_end_recall_at_10:.4f}</td>"
            f"<td>{metrics.ndcg_at_10:.4f}</td>"
            f"<td>{_display_catalog_coverage(metrics.catalog_coverage_at_10)}</td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr><th>Retriever</th><th>Coverage@200</th>"
        "<th>ConditionalRecall@10</th><th>EndToEndRecall@10</th><th>NDCG@10</th>"
        "<th>CatalogCoverage@10</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )
    return (
        '<section id="retrieval-quality"><h2>Retrieval coverage</h2>'
        "<p>Candidate Pools are capped at 200 and exclude snapshot-observed Movies.</p>"
        f"{table}</section>\n"
        '<section id="popularity-metrics"><h2>Popularity metrics</h2>'
        f"{_metrics_table(evaluation.retrievers.get('popularity'))}</section>\n"
        '<section id="itemknn-metrics"><h2>ItemKNN metrics</h2>'
        f"{_metrics_table(evaluation.retrievers.get('itemknn'))}</section>\n"
        '<section id="multivae-metrics"><h2>Mult-VAE metrics</h2>'
        f"{_metrics_table(evaluation.retrievers.get('multivae'))}</section>\n"
        '<section id="lightgcn-metrics"><h2>LightGCN metrics</h2>'
        f"{_metrics_table(evaluation.retrievers.get('lightgcn'))}</section>\n"
        '<section id="rrf-metrics"><h2>RRF metrics</h2>'
        f"{_metrics_table(evaluation.rrf)}</section>\n"
        '<section id="lhf-metrics"><h2>LHF metrics</h2>'
        f"{_metrics_table(evaluation.lhf)}</section>\n"
        '<section id="oracle-union"><h2>Oracle Union</h2>'
        f"<p>Diagnostic-only unbudgeted union coverage: "
        f"{evaluation.oracle_union_coverage:.4f}. It is not served. "
        f"Best single is selected from {html.escape(evaluation.best_single_source)}; "
        f"realized Oracle headroom is {_headroom_text(evaluation)}.</p></section>\n"
        '<section id="final-ranking-quality"><h2>Final ranking quality</h2>'
        "<p>ConditionalRecall@10 and NDCG@10 describe ranking quality after retrieval coverage."
        "</p></section>\n"
        f"{known_section}{history_section}"
    )


def _empty_history_fast_section(evaluation: EvaluationReport | None) -> str:
    if evaluation is None:
        return '<section id="empty-history-quality"><h2>Empty-History quality</h2><p>n/a</p></section>\n'
    return (
        '<section id="empty-history-quality"><h2>Empty-History quality</h2>'
        "<p>Popularity fallback; Gold Candidates come from Subjects with no snapshot Positive "
        "Interactions and their first Future Window Positive Interaction. "
        f"Denominator: {_denominator_display(evaluation.cohort.size)}.</p>"
        f"{_metrics_table(evaluation.retrievers.get('popularity'))}</section>\n"
    )


def _mode_comparison(evaluation: EvaluationReport, label: str) -> str:
    rows = ""
    comparisons: list[tuple[str, RetrieverEvaluation | None]] = [
        *evaluation.retrievers.items(),
        ("best single", evaluation.best_single),
        ("RRF", evaluation.rrf),
        ("LHF", evaluation.lhf),
        ("Oracle Union", evaluation.oracle_union),
    ]
    for name, retriever_evaluation in comparisons:
        if retriever_evaluation is None:
            continue
        metrics = retriever_evaluation.metrics
        rows += (
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{metrics.coverage_at_200:.4f}</td>"
            f"<td>{metrics.conditional_recall_at_10:.4f}</td>"
            f"<td>{metrics.end_to_end_recall_at_10:.4f}</td>"
            f"<td>{metrics.ndcg_at_10:.4f}</td>"
            f"<td>{_display_catalog_coverage(metrics.catalog_coverage_at_10)}</td>"
            "</tr>"
        )
    return (
        f'<section id="{evaluation.query_mode.replace("_", "-")}-comparison">'
        f"<h2>{html.escape(label)} comparison</h2>"
        "<p>Best-single selection source: "
        f"<code>{html.escape(evaluation.best_single_source)}</code>; "
        f"Oracle headroom realized: {_headroom_text(evaluation)}.</p>"
        "<table><thead><tr><th>System</th><th>Coverage@200</th>"
        "<th>ConditionalRecall@10</th><th>EndToEndRecall@10</th><th>NDCG@10</th>"
        "<th>CatalogCoverage@10</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></section>\n"
    )


def _headroom_text(evaluation: EvaluationReport) -> str:
    if evaluation.oracle_headroom_realized is None:
        return "undefined (zero denominator)"
    return f"{evaluation.oracle_headroom_realized:.4f}"


def _metrics_table(evaluation: RetrieverEvaluation | None) -> str:
    if evaluation is None:
        return "<p>Evaluation results are not available.</p>"
    metrics = evaluation.metrics
    return (
        "<table><tbody>"
        f"<tr><th>Coverage@200</th><td>{metrics.coverage_at_200:.4f}</td></tr>"
        f"<tr><th>ConditionalRecall@10</th><td>{metrics.conditional_recall_at_10:.4f}</td></tr>"
        f"<tr><th>EndToEndRecall@10</th><td>{metrics.end_to_end_recall_at_10:.4f}</td></tr>"
        f"<tr><th>NDCG@10</th><td>{metrics.ndcg_at_10:.4f}</td></tr>"
        f"<tr><th>CatalogCoverage@10</th><td>{_display_catalog_coverage(metrics.catalog_coverage_at_10)}</td></tr>"
        "</tbody></table>"
    )


def _display_catalog_coverage(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _multivae_resource_section(artifact: ServingArtifact) -> str:
    metadata = artifact.manifest.get("multivae_training", {})
    if not isinstance(metadata, dict):
        return (
            '<section id="multivae-resource-evidence"><h2>Mult-VAE resource evidence</h2>'
            "<p>Training resource evidence is unavailable.</p></section>\n"
        )
    actual_device = html.escape(str(metadata.get("actual_device", "unknown")))
    duration = html.escape(str(metadata.get("duration_seconds", "unknown")))
    fallback = html.escape(str(metadata.get("fallback_reason") or "none"))
    return (
        '<section id="multivae-resource-evidence"><h2>Mult-VAE resource evidence</h2>'
        f"<p>Training duration: <code>{duration}</code> seconds; "
        f"actual device: <code>{actual_device}</code>; "
        f"fallback reason: <code>{fallback}</code>.</p></section>\n"
    )


def _lightgcn_resource_section(artifact: ServingArtifact) -> str:
    metadata = artifact.manifest.get("lightgcn_training", {})
    if not isinstance(metadata, dict) or not metadata:
        return (
            '<section id="lightgcn-resource-evidence"><h2>LightGCN resource evidence</h2>'
            "<p>Training resource evidence is unavailable.</p></section>\n"
        )
    actual_device = html.escape(str(metadata.get("actual_device", "unknown")))
    duration = html.escape(str(metadata.get("duration_seconds", "unknown")))
    fallback = html.escape(str(metadata.get("fallback_reason") or "none"))
    return (
        '<section id="lightgcn-resource-evidence"><h2>LightGCN resource evidence</h2>'
        f"<p>Training duration: <code>{duration}</code> seconds; "
        f"actual device: <code>{actual_device}</code>; "
        f"fallback reason: <code>{fallback}</code>.</p></section>\n"
    )


def write_rolling_report(
    path: Path,
    *,
    stages: Sequence[Mapping[str, Any]],
    source_event_count: int,
    dataset_checksum: str,
    active_artifact_id: str,
    source_revision: str,
    configuration: Mapping[str, Any] | None = None,
    latency_benchmark: Mapping[str, Any] | None = None,
) -> None:
    """Write a self-contained, aggregate-only portfolio report for the rolling lifecycle."""

    aggregate_stages = [_aggregate_stage_payload(stage) for stage in stages]
    aggregate_benchmark = _aggregate_benchmark(latency_benchmark)
    is_full_profile = (configuration or {}).get("profile") == "full"
    data_description = "MovieLens 20M" if is_full_profile else "MovieLens-shaped fixture"
    resource_description = (
        "full MovieLens 20M lifecycle" if is_full_profile else "local fast fixture lifecycle"
    )
    dataset_limitation = (
        "<li>The 20M results describe one local temporal replay and one deterministic cohort "
        "selection, not a multi-seed confidence interval or an online experiment.</li>"
        if is_full_profile
        else "<li>This report uses the deterministic local fast fixture. Fixture numbers must "
        "not be presented as MovieLens 20M results.</li>"
    )
    embedded = {
        "schema_version": 2,
        "source_event_count": source_event_count,
        "dataset_checksum": dataset_checksum,
        "active_artifact_id": active_artifact_id,
        "source_revision": source_revision,
        "configuration": dict(configuration or {}),
        "stages": aggregate_stages,
        "latency_benchmark": aggregate_benchmark,
    }
    data_json = json.dumps(embedded, indent=2, sort_keys=True, ensure_ascii=False).replace(
        "</", "<\\/"
    )
    headline = _headline_summary(aggregate_stages, aggregate_benchmark)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html>\n"
        '<html lang="en">\n'
        '<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Movie Recommender Lifecycle Showcase</title>"
        "<style>"
        ":root{color-scheme:light;--ink:#17202a;--muted:#5b6875;--line:#d9e0e7;"
        "--panel:#f6f8fa;--accent:#0b6e69;--warn:#9a5b00;--fail:#a32626}"
        "*{box-sizing:border-box}body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"
        "'Segoe UI',sans-serif;color:var(--ink);max-width:92rem;margin:0 auto;padding:2rem;"
        "line-height:1.5;background:#fff}h1{font-size:2.3rem;line-height:1.1;margin:.2rem 0 .7rem}"
        "h2{margin-top:0;font-size:1.45rem}h3{margin-bottom:.35rem}p{max-width:78rem}"
        "section{margin:2.2rem 0;padding:1.25rem 1.4rem;border:1px solid var(--line);"
        "border-radius:12px;background:#fff}section.hero{border:0;background:linear-gradient(135deg,#eef8f6,#f7f8fb);"
        "padding:2rem}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--panel);"
        "padding:.1rem .3rem;border-radius:4px;overflow-wrap:anywhere}pre{background:#101820;color:#eef5f7;"
        "padding:1rem;overflow:auto;border-radius:8px;font-size:.8rem}table{border-collapse:collapse;"
        "width:100%;font-size:.88rem;margin:1rem 0}th,td{border:1px solid var(--line);padding:.52rem;"
        "text-align:left;vertical-align:top}th{background:var(--panel);font-weight:650}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:.8rem}"
        ".card{border:1px solid var(--line);border-radius:9px;padding:.85rem;background:#fff}"
        ".value{display:block;font-size:1.25rem;font-weight:700;margin-top:.2rem}"
        ".muted{color:var(--muted)}.pass{color:var(--accent);font-weight:700}.fail{color:var(--fail);font-weight:700}"
        ".warn{color:var(--warn);font-weight:700}.flow{font-size:1.05rem;font-weight:650;word-spacing:.18rem;"
        "background:var(--panel);padding:1rem;border-radius:8px;overflow:auto}"
        ".small{font-size:.82rem}.nowrap{white-space:nowrap}ul{padding-left:1.3rem}"
        "@media(max-width:700px){body{padding:1rem}section{padding:1rem;overflow:auto}table{min-width:46rem}"
        "h1{font-size:1.8rem}}"
        "</style></head>\n"
        "<body>\n"
        '<section id="executive-summary" class="hero">'
        "<h1>Movie Recommender Lifecycle Showcase</h1>"
        f"<p>This local-first system turns chronologically replayed {data_description} Rating Events "
        "into time-bounded snapshots, candidate retrievers, query-mode Learned Hybrid Fusion (LHF), "
        "an immutable CPU Serving Artifact, and a FastAPI recommendation path.</p>"
        '<div class="grid">'
        f'<div class="card"><span class="muted">Final quality window</span><span class="value">90→100%</span>'
        f"<span>{html.escape(headline['quality'])}</span></div>"
        f'<div class="card"><span class="muted">Active artifact</span><span class="value"><code>'
        f"{html.escape(active_artifact_id)}</code></span><span>100% cutoff; serving only.</span></div>"
        f'<div class="card"><span class="muted">Latency evidence</span><span class="value">'
        f"{html.escape(headline['latency'])}</span><span>Warm CPU, top_n=10, concurrency=1.</span></div>"
        "</div>"
        '<p class="muted small">Stage 100 is an artifact-production and activation stage; it '
        "has no 100→110% Future Window, so no future-quality claim is assigned to it.</p>"
        "</section>\n"
        '<section id="architecture-data-flow"><h2>Architecture and data flow</h2>'
        '<div class="flow">Kafka → Event Store → snapshots → retrievers → LHF → artifact → CPU API</div>'
        "<p>One local Apache Kafka broker provides the ingestion boundary. The append-only Event "
        "Store is materialized into cumulative Data Snapshots; Popularity, ItemKNN, Mult-VAE, and "
        "LightGCN produce bounded Candidate Pools. Query-mode LHF orders the appropriate union, "
        "then the immutable artifact is loaded by the CPU-serving API.</p>"
        "<table><thead><tr><th>Boundary</th><th>Evidence and contract</th></tr></thead><tbody>"
        "<tr><td>Ingestion</td><td>Rating Events cross Kafka and are persisted append-only; deterministic "
        "Event IDs make replay deduplicable.</td></tr>"
        "<tr><td>Temporal evaluation</td><td>Stages 50–90 evaluate the next ten-percent Future Window "
        "before that window is ingested.</td></tr>"
        "<tr><td>Routing</td><td>Known-User uses the four-retriever bank; History-Only uses the "
        "three-retriever bank without LightGCN; Empty-History uses Popularity.</td></tr>"
        "<tr><td>Serving</td><td>Artifact loading, model fitting, Kafka replay, and Event Store reads "
        "are outside each request. Returned Candidates exclude the supplied/known history.</td></tr>"
        "</tbody></table></section>\n"
        '<section id="lifecycle-progression"><h2>Lifecycle progression: 50% → 100%</h2>'
        "<table><thead><tr><th>Stage</th><th>Snapshot Events</th><th>Artifact</th><th>Evaluation window</th>"
        "<th>Active</th><th>Known-User LHF EndToEndRecall@10</th>"
        "<th>History-Only LHF EndToEndRecall@10</th></tr></thead><tbody>"
        f"{_stage_rows(aggregate_stages)}</tbody></table>"
        "<p>Stages 50–90 are evaluated on their next Future Window, then the Event Store advances. "
        "The 100% artifact is smoke-tested and activated only after export. <strong>Stage 100 has "
        "no Future Window and carries no future-quality claim.</strong> The headline is the measured "
        "90→100% evaluation.</p></section>\n"
        '<section id="evaluation-boundaries"><h2>Evaluation protocol and metric definitions</h2>'
        "<p>Each eligible Subject contributes at most one Query: its snapshot Positive-Interaction "
        "history and the first positive Gold Candidate in the next Future Window. The Future Window "
        "is never used for current-stage fitting, LHF labels, or serving features.</p>"
        "<ul><li><strong>Coverage@200:</strong> fraction of Queries whose Gold Candidate occurs in a "
        "history-excluded Candidate Pool capped at 200.</li>"
        "<li><strong>ConditionalRecall@10:</strong> fraction of covered Queries whose Gold Candidate "
        "is in the final Top-10; denominator is covered Queries.</li>"
        "<li><strong>EndToEndRecall@10:</strong> fraction of all cohort Queries whose Gold Candidate "
        "is in the final Top-10; denominator is the full cohort.</li>"
        "<li><strong>NDCG@10:</strong> one-Gold discounted gain at the final rank, averaged over the "
        "full cohort.</li>"
        "<li><strong>CatalogCoverage@10:</strong> distinct Top-10 Movies across the cohort divided "
        "by the snapshot Candidate Catalog size.</li>"
        "<li><strong>Oracle headroom:</strong> LHF coverage gained over the validation-selected best "
        "single retriever divided by the remaining best-single-to-Oracle-Union coverage gap. A zero "
        "denominator is undefined, not zero.</li></ul>"
        "<p>The diagnostic Oracle Union is the unbudgeted union of the bounded retriever pools; it is "
        "not served. If a Gold Candidate is outside that union, the failure is retrieval failure and "
        "no downstream ranker can recover it.</p></section>\n"
        f"{_quality_section(_mode_evaluation(aggregate_stages, 'known_user'), 'Known-User', 'quality-known-user')}"
        f"{_quality_section(_mode_evaluation(aggregate_stages, 'history_only'), 'History-Only', 'quality-history-only')}"
        f"{_cold_user_section(aggregate_stages)}"
        f"{_empty_history_section(aggregate_stages)}"
        '<section id="retrieval-bottleneck"><h2>Retrieval bottleneck and fusion diagnosis</h2>'
        f"{_retrieval_diagnosis(aggregate_stages)}</section>\n"
        '<section id="resource-device-timing"><h2>Resource, device, and timing evidence</h2>'
        "<p>Neural training records the actual device and fallback reason. Serving is CPU-only; "
        f"these timings describe the {resource_description}, not production capacity.</p>"
        f"<table><thead><tr><th>Stage</th><th>Mult-VAE</th><th>LightGCN</th><th>Training s</th>"
        f"<th>Export s</th><th>Smoke s</th><th>Evaluation s</th></tr></thead><tbody>{_resource_rows(aggregate_stages)}"
        "</tbody></table></section>\n"
        f"{_latency_section(aggregate_benchmark)}"
        f"{_failure_section(aggregate_stages, aggregate_benchmark)}"
        '<section id="limitations"><h2>Limitations and honest interpretation</h2>'
        "<ul><li>Collaborative retrievers do not solve Interaction-New Movies: a movie without "
        "snapshot interaction evidence is not made recommendable by this system.</li>"
        f"{dataset_limitation}"
        "<li>Empty-History quality uses new Subjects in the Future Window; zero eligible Subjects "
        "are shown as n/a with denominator 0.</li>"
        "<li>LHF can be worse than a baseline. The report exposes the measured comparison and does not "
        "turn a non-uplift into an improvement claim.</li>"
        "<li>Latency evidence is an in-process ASGI request-path measurement, not loopback/network "
        "latency. It is a local benchmark rather than a capacity or load test.</li></ul></section>\n"
        '<section id="provenance"><h2>Provenance</h2>'
        f"{_provenance_section(aggregate_stages, embedded)}</section>\n"
        f'<script id="embedded-report-data" type="application/json">{data_json}</script>\n'
        "</body>\n</html>\n",
        encoding="utf-8",
    )


_REPORT_METRICS = (
    "Coverage@200",
    "ConditionalRecall@10",
    "EndToEndRecall@10",
    "NDCG@10",
    "CatalogCoverage@10",
)
_REPORT_SEGMENTS = ("1-4", "5-19", "20+")


def _aggregate_stage_payload(stage: Mapping[str, Any]) -> dict[str, object]:
    manifest = _mapping(stage.get("artifact_manifest"))
    evaluation = stage.get("evaluation")
    configuration = _mapping(manifest.get("configuration"))
    feature_banks = {
        mode: _aggregate_feature_bank(configuration.get(f"{mode}_fusion"))
        for mode in ("known_user", "history_only")
    }
    fusion_training = stage.get("lhf_training")
    if not isinstance(fusion_training, Mapping):
        fusion_training = manifest.get("fusion_training")
    return {
        "percentage": _integer(stage.get("percentage")),
        "snapshot_event_count": _integer(stage.get("snapshot_event_count")),
        "source_event_count": _integer(stage.get("source_event_count")),
        "snapshot_fingerprint": _text(stage.get("snapshot_fingerprint")),
        "artifact_id": _text(stage.get("artifact_id")),
        "active": bool(stage.get("active")),
        "evaluation_window": _evaluation_window(stage.get("evaluation_window")),
        "evaluation": _aggregate_stage_evaluation(evaluation),
        "timings": _numeric_mapping(stage.get("timings")),
        "resources": {
            "multivae": _aggregate_resource(manifest.get("multivae_training")),
            "lightgcn": _aggregate_resource(manifest.get("lightgcn_training")),
        },
        "lhf_training": _aggregate_fusion_training(fusion_training),
        "provenance": {
            "source_revision": _text(manifest.get("source_revision")),
            "dataset_checksum": _text(manifest.get("dataset_checksum")),
            "configuration_sha256": _text(manifest.get("configuration_sha256")),
            "random_seed": _integer(manifest.get("random_seed")),
            "data_cutoff_percentage": _integer(manifest.get("data_cutoff_percentage")),
            "query_mode_feature_banks": feature_banks,
        },
    }


def _aggregate_evaluation(value: object) -> dict[str, object] | None:
    raw = _mapping(value)
    if not raw:
        return None
    result: dict[str, object] = {
        "query_mode": _text(raw.get("query_mode")),
        "cohort_size": _integer(raw.get("cohort_size")),
        "best_single_name": _text(raw.get("best_single_name")),
        "best_single_source": _text(raw.get("best_single_source")),
        "best_single": _aggregate_metric(raw.get("best_single")),
        "oracle_union": _aggregate_metric(raw.get("oracle_union")),
        "oracle_union_coverage": _number(raw.get("oracle_union_coverage")),
        "retrieval_failure_count": _integer(raw.get("retrieval_failure_count")),
        "oracle_headroom_denominator": _number(raw.get("oracle_headroom_denominator")),
        "oracle_headroom_realized": _number(raw.get("oracle_headroom_realized")),
        "retrievers": {},
        "rrf": _aggregate_metric(raw.get("rrf")),
        "lhf": _aggregate_metric(raw.get("lhf")),
        "history_segments": _aggregate_segments(raw.get("history_segments")),
    }
    retrievers = raw.get("retrievers")
    if isinstance(retrievers, Mapping):
        result["retrievers"] = {
            str(name): _aggregate_metric(metrics) for name, metrics in retrievers.items()
        }
    return result


def _aggregate_stage_evaluation(value: object) -> dict[str, object] | None:
    raw = _mapping(value)
    if not raw:
        return None
    if any(mode in raw for mode in ("known_user", "history_only", "empty_history")):
        result: dict[str, object] = {}
        for mode in ("known_user", "history_only", "empty_history"):
            evaluation = _aggregate_evaluation(raw.get(mode))
            if evaluation is not None:
                result[mode] = evaluation
        return result
    return _aggregate_evaluation(raw)


def _aggregate_segments(value: object) -> dict[str, dict[str, object]]:
    raw_segments = value if isinstance(value, Mapping) else {}
    result: dict[str, dict[str, object]] = {}
    for segment in _REPORT_SEGMENTS:
        raw = _mapping(raw_segments.get(segment))
        cohort_size = _integer(raw.get("cohort_size"))
        if cohort_size is None or cohort_size == 0:
            result[segment] = {
                "cohort_size": cohort_size,
                "denominator": _integer(raw.get("denominator", cohort_size)),
                "metrics": {},
            }
            continue
        raw_metrics = raw.get("metrics")
        metrics = (
            {
                str(name): _aggregate_metric(metric)
                for name, metric in raw_metrics.items()
                if isinstance(name, str)
            }
            if isinstance(raw_metrics, Mapping)
            else {}
        )
        result[segment] = {
            "cohort_size": cohort_size,
            "denominator": _integer(raw.get("denominator", cohort_size)),
            "metrics": metrics,
        }
    return result


def _aggregate_metric(value: object) -> dict[str, int | float] | None:
    raw = _mapping(value)
    if not raw:
        return None
    result: dict[str, int | float] = {}
    for key in ("query_count", "covered_query_count", "ranking_success_count", *_REPORT_METRICS):
        number = raw.get(key)
        if isinstance(number, (int, float)) and not isinstance(number, bool):
            result[key] = number
    return result or None


def _aggregate_resource(value: object) -> dict[str, object]:
    raw = _mapping(value)
    result: dict[str, object] = {}
    for key in ("requested_device", "actual_device", "duration_seconds", "fallback_reason"):
        child = raw.get(key)
        if child is None or isinstance(child, (str, int, float)) and not isinstance(child, bool):
            result[key] = child
    benchmark_seconds = raw.get("benchmark_seconds")
    if isinstance(benchmark_seconds, Mapping):
        result["benchmark_seconds"] = _numeric_mapping(benchmark_seconds)
    policy = raw.get("benchmark_policy")
    if isinstance(policy, str):
        result["benchmark_policy"] = policy
    return result


def _aggregate_feature_bank(value: object) -> dict[str, object]:
    raw = _mapping(value)
    bank = raw.get("retriever_bank")
    result: dict[str, object] = {
        "retriever_bank": [item for item in bank if isinstance(item, str)]
        if isinstance(bank, list)
        else [],
    }
    schema = _mapping(raw.get("feature_schema"))
    if isinstance(schema.get("version"), int):
        result["feature_schema_version"] = schema["version"]
    if isinstance(schema.get("user_cold_history_threshold"), int):
        result["user_cold_history_threshold"] = schema["user_cold_history_threshold"]
    return result


def _aggregate_fusion_training(value: object) -> dict[str, dict[str, object]]:
    raw = value if isinstance(value, Mapping) else {}
    result: dict[str, dict[str, object]] = {}
    for mode in ("known_user", "history_only"):
        metadata = _mapping(raw.get(mode))
        selected: dict[str, object] = {}
        for key in (
            "row_count",
            "positive_row_count",
            "negative_row_count",
            "training_boundary",
            "future_window_used_for_training",
            "fallback_reason",
        ):
            child = metadata.get(key)
            if (
                child is None
                or isinstance(child, (str, int, float))
                and not isinstance(child, bool)
            ):
                selected[key] = child
        result[mode] = selected
    return result


def _aggregate_benchmark(value: Mapping[str, Any] | None) -> dict[str, object]:
    raw = value if isinstance(value, Mapping) else {}
    result: dict[str, object] = {
        "schema_version": _integer(raw.get("schema_version", 1)),
        "status": _text(raw.get("status")) or "not_measured",
        "artifact_id": _text(raw.get("artifact_id")),
        "artifact_cutoff_percentage": _integer(raw.get("artifact_cutoff_percentage")),
        "artifact_loaded_before_measurement": raw.get("artifact_loaded_before_measurement") is True,
        "serving_device": _text(raw.get("serving_device")) or "cpu",
        "top_n": _integer(raw.get("top_n", 10)),
        "concurrency": _integer(raw.get("concurrency", 1)),
        "warmup_count": _integer(raw.get("warmup_count")),
        "requested_sample_count": _integer(raw.get("requested_sample_count")),
        "percentile_method": _text(raw.get("percentile_method")),
        "method": _text(raw.get("method")) or "not measured",
        "environment": {},
        "modes": {},
    }
    environment = raw.get("environment")
    if isinstance(environment, Mapping):
        result["environment"] = {
            str(key): child
            for key, child in environment.items()
            if isinstance(key, str)
            and (
                child is None
                or isinstance(child, (str, int, float))
                and not isinstance(child, bool)
            )
        }
    modes = raw.get("modes")
    if isinstance(modes, Mapping):
        result["modes"] = {
            mode: _aggregate_benchmark_mode(modes.get(mode))
            for mode in ("known_user", "history_only", "empty_history")
        }
    else:
        result["modes"] = {
            mode: _aggregate_benchmark_mode(None)
            for mode in ("known_user", "history_only", "empty_history")
        }
    return result


def _aggregate_benchmark_mode(value: object) -> dict[str, object]:
    raw = _mapping(value)
    selected: dict[str, object] = {}
    for key in (
        "measurement_status",
        "slo_status",
        "warmup_count",
        "warmup_success_count",
        "requested_sample_count",
        "sample_count",
        "successful_sample_count",
        "failed_sample_count",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "throughput_rps",
        "duration_total_ms",
        "slo_target_p95_ms",
        "reason",
        "percentile_method",
    ):
        child = raw.get(key)
        if child is None or isinstance(child, (str, int, float)) and not isinstance(child, bool):
            selected[key] = child
    if "measurement_status" not in selected:
        selected["measurement_status"] = "not_measured"
    if "slo_status" not in selected:
        selected["slo_status"] = "not_measured"
    return selected


def _stage_rows(stages: Sequence[Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for stage in stages:
        evaluation = _mapping(stage.get("evaluation"))
        known = _mapping(evaluation.get("known_user"))
        history = _mapping(evaluation.get("history_only"))
        rows.append(
            "<tr>"
            f"<td>{_display(stage.get('percentage'), suffix='%')}</td>"
            f"<td>{_display(stage.get('snapshot_event_count'))}</td>"
            f"<td><code>{_display(stage.get('artifact_id'))}</code></td>"
            f"<td>{_window_display(stage.get('evaluation_window'))}</td>"
            f"<td>{'yes' if stage.get('active') else 'no'}</td>"
            f"<td>{_metric_with_denominator(_mapping(known.get('lhf')), 'EndToEndRecall@10')}</td>"
            f"<td>{_metric_with_denominator(_mapping(history.get('lhf')), 'EndToEndRecall@10')}</td>"
            "</tr>"
        )
    return "".join(rows)


def _quality_section(
    evaluation: Mapping[str, Any] | None,
    label: str,
    section_id: str,
) -> str:
    if evaluation is None:
        return (
            f'<section id="{section_id}"><h2>{html.escape(label)} quality</h2>'
            "<p>Quality evidence is not available.</p></section>\n"
        )
    rows: list[str] = []
    retrievers = evaluation.get("retrievers")
    if isinstance(retrievers, Mapping):
        for name, metrics in retrievers.items():
            rows.append(_quality_row(_display_name(str(name)), metrics))
    rows.extend(
        (
            _quality_row("Best single", evaluation.get("best_single")),
            _quality_row("RRF", evaluation.get("rrf")),
            _quality_row("LHF", evaluation.get("lhf")),
            _quality_row("Oracle Union (diagnostic)", evaluation.get("oracle_union")),
        )
    )
    cohort = _integer(evaluation.get("cohort_size"))
    failures = _integer(evaluation.get("retrieval_failure_count"))
    headroom = _headroom_display(evaluation)
    return (
        f'<section id="{section_id}"><h2>{html.escape(label)} quality</h2>'
        f"<p>Cohort size / denominator: <strong>{_display(cohort)}</strong>. "
        "Every metric below uses the full cohort except ConditionalRecall@10, whose denominator "
        "is the covered subset.</p>"
        "<table><thead><tr><th>System</th><th>Cohort / denominator</th>"
        "<th>Coverage@200</th><th>ConditionalRecall@10</th><th>EndToEndRecall@10</th>"
        f"<th>NDCG@10</th><th>CatalogCoverage@10</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        f"<p>Oracle headroom realized: <strong>{html.escape(headroom)}</strong>; "
        f"Gold Candidates outside the Oracle Union: <strong>{_display(failures)}</strong> retrieval "
        "failures.</p></section>\n"
    )


def _quality_row(name: str, value: object) -> str:
    metrics = _mapping(value)
    denominator = _integer(metrics.get("query_count"))
    return (
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{_denominator_display(denominator)}</td>"
        f"<td>{_metric_value(metrics, 'Coverage@200')}</td>"
        f"<td>{_metric_value(metrics, 'ConditionalRecall@10')}</td>"
        f"<td>{_metric_value(metrics, 'EndToEndRecall@10')}</td>"
        f"<td>{_metric_value(metrics, 'NDCG@10')}</td>"
        f"<td>{_metric_value(metrics, 'CatalogCoverage@10')}</td>"
        "</tr>"
    )


def _cold_user_section(stages: Sequence[Mapping[str, Any]]) -> str:
    known = _mode_evaluation(stages, "known_user")
    history = _mode_evaluation(stages, "history_only")
    return (
        '<section id="cold-user-quality"><h2>Cold-user and history-length quality</h2>'
        "<p>Known-User and History-Only cohorts are segmented by snapshot Positive-Interaction "
        "history length. A missing segment has no denominator and is rendered as <strong>n/a</strong>, "
        "never as 0.0.</p>"
        "<h3>Known-User segments</h3>"
        f"{_segment_table(known)}"
        "<h3>History-Only segments</h3>"
        f"{_segment_table(history)}"
        "</section>\n"
    )


def _empty_history_section(stages: Sequence[Mapping[str, Any]]) -> str:
    evaluation = _mode_evaluation(stages, "empty_history")
    if evaluation is None:
        return '<section id="quality-empty-history"><h2>Empty-History quality</h2><p>n/a</p></section>\n'
    cohort = _integer(evaluation.get("cohort_size"))
    return (
        '<section id="quality-empty-history"><h2>Empty-History quality</h2>'
        "<p>Popularity fallback only. Gold Candidate is the first Positive Interaction in the "
        "Future Window for a Subject with no Positive Interactions in the Data Snapshot. "
        f"Cohort / denominator: <strong>{_denominator_display(cohort)}</strong>.</p>"
        "<table><thead><tr><th>System</th><th>Cohort / denominator</th><th>Coverage@200</th>"
        "<th>ConditionalRecall@10</th><th>EndToEndRecall@10</th><th>NDCG@10</th>"
        "<th>CatalogCoverage@10</th></tr></thead><tbody>"
        f"{_quality_row('Popularity fallback', evaluation.get('best_single'))}"
        "</tbody></table></section>\n"
    )


def _segment_table(evaluation: Mapping[str, Any] | None) -> str:
    if evaluation is None:
        return "<p>Segment evidence is not available.</p>"
    segments = _mapping(evaluation.get("history_segments"))
    rows: list[str] = []
    for segment in _REPORT_SEGMENTS:
        raw = _mapping(segments.get(segment))
        cohort = _integer(raw.get("cohort_size"))
        metrics = _mapping(raw.get("metrics"))
        lhf = _mapping(metrics.get("lhf"))
        best = _mapping(metrics.get("best_single"))
        oracle = _mapping(metrics.get("oracle_union"))
        rows.append(
            "<tr>"
            f"<td>{html.escape(segment)}</td>"
            f"<td>{_denominator_display(cohort)}</td>"
            f"<td>{_metric_value(lhf, 'Coverage@200', missing_if_empty=True)}</td>"
            f"<td>{_metric_value(lhf, 'EndToEndRecall@10', missing_if_empty=True)}</td>"
            f"<td>{_metric_value(best, 'EndToEndRecall@10', missing_if_empty=True)}</td>"
            f"<td>{_metric_value(oracle, 'Coverage@200', missing_if_empty=True)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>History segment</th><th>Cohort / denominator</th>"
        "<th>LHF Coverage@200</th><th>LHF EndToEndRecall@10</th>"
        "<th>Best single EndToEndRecall@10</th><th>Oracle Union coverage</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _retrieval_diagnosis(stages: Sequence[Mapping[str, Any]]) -> str:
    evaluation = _stage_evaluation(stages, 90)
    if evaluation is None:
        return "<p>Stage 90 diagnosis is not available.</p>"
    fragments: list[str] = []
    for mode, label in (("known_user", "Known-User"), ("history_only", "History-Only")):
        summary = _mapping(evaluation.get(mode))
        best = _mapping(summary.get("best_single"))
        oracle = _mapping(summary.get("oracle_union"))
        lhf = _mapping(summary.get("lhf"))
        fragments.append(
            f"<li><strong>{label}:</strong> validation-selected best single "
            f"<code>{html.escape(_text(summary.get('best_single_name')) or 'n/a')}</code> has "
            f"Coverage@200 {_metric_value(best, 'Coverage@200')}; RRF is "
            f"{_metric_value(_mapping(summary.get('rrf')), 'Coverage@200')}; LHF is "
            f"{_metric_value(lhf, 'Coverage@200')}; Oracle Union is "
            f"{_metric_value(oracle, 'Coverage@200')}.</li>"
        )
    return (
        "<p>The first diagnostic question is whether the Gold Candidate was retrieved at all; only "
        "then does ranking quality have a chance to change the Top-10 result. LHF uses the same union "
        "as its mode-specific bank and can be below a baseline; no improvement is claimed without a "
        "metric showing it.</p><ul>"
        f"{''.join(fragments)}</ul>"
    )


def _resource_rows(stages: Sequence[Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for stage in stages:
        resources = _mapping(stage.get("resources"))
        multivae = _mapping(resources.get("multivae"))
        lightgcn = _mapping(resources.get("lightgcn"))
        timings = _mapping(stage.get("timings"))
        rows.append(
            "<tr>"
            f"<td>{_display(stage.get('percentage'), suffix='%')}</td>"
            f"<td>{_resource_display(multivae)}</td>"
            f"<td>{_resource_display(lightgcn)}</td>"
            f"<td>{_display(timings.get('training'))}</td>"
            f"<td>{_display(timings.get('export'))}</td>"
            f"<td>{_display(timings.get('smoke'))}</td>"
            f"<td>{_display(timings.get('evaluation'))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _latency_section(benchmark: Mapping[str, Any]) -> str:
    environment = _mapping(benchmark.get("environment"))
    modes = _mapping(benchmark.get("modes"))
    rows: list[str] = []
    for mode, label in (
        ("known_user", "Known-User"),
        ("history_only", "History-Only"),
        ("empty_history", "Empty-History"),
    ):
        value = _mapping(modes.get(mode))
        slo = _slo_display(value.get("slo_status"))
        rows.append(
            "<tr>"
            f"<td>{label}</td>"
            f"<td>{html.escape(_text(value.get('measurement_status')) or 'not measured')}</td>"
            f"<td>{_display(value.get('warmup_count'))}</td>"
            f"<td>{_display(value.get('successful_sample_count'))} / "
            f"{_display(value.get('failed_sample_count'))}</td>"
            f"<td>{_latency_value(value.get('p50_ms'))}</td>"
            f"<td>{_latency_value(value.get('p95_ms'))}</td>"
            f"<td>{_latency_value(value.get('p99_ms'))}</td>"
            f"<td>{_latency_value(value.get('throughput_rps'))}</td>"
            f"<td>{_latency_value(value.get('slo_target_p95_ms'))}</td>"
            f'<td class="{slo[1]}">{slo[0]}</td>'
            "</tr>"
        )
    reason_rows = []
    for mode, label in (
        ("known_user", "Known-User"),
        ("history_only", "History-Only"),
        ("empty_history", "Empty-History"),
    ):
        reason = _mapping(modes.get(mode)).get("reason")
        if reason:
            reason_rows.append(f"<li><strong>{label}:</strong> {html.escape(str(reason))}</li>")
    reasons = f"<ul>{''.join(reason_rows)}</ul>" if reason_rows else ""
    return (
        '<section id="latency-evidence"><h2>Warm CPU serving latency and SLO</h2>'
        f"<p>Status: <strong>{html.escape(_text(benchmark.get('status')) or 'not measured')}</strong>. "
        f"{html.escape(_text(benchmark.get('method')) or 'not measured')}. The active artifact was "
        f"loaded before timing: <code>{str(benchmark.get('artifact_loaded_before_measurement')).lower()}</code>. "
        f"Top-N: <code>{_display(benchmark.get('top_n'))}</code>; concurrency: "
        f"<code>{_display(benchmark.get('concurrency'))}</code>; warm-up count: "
        f"<code>{_display(benchmark.get('warmup_count'))}</code>; requested sample count: "
        f"<code>{_display(benchmark.get('requested_sample_count'))}</code>. Percentiles: "
        f"<code>{html.escape(_text(benchmark.get('percentile_method')) or 'not measured')}</code>.</p>"
        "<table><thead><tr><th>Query mode</th><th>Measurement</th><th>Warm-up</th>"
        "<th>Successful / failed samples</th><th>p50 ms</th><th>p95 ms</th><th>p99 ms</th>"
        "<th>Throughput req/s</th><th>p95 SLO ms</th><th>SLO</th></tr></thead><tbody>"
        f"{''.join(rows)}</tbody></table>"
        f"<p>Hardware: <code>{html.escape(_text(environment.get('cpu')) or 'not measured')}</code>; "
        f"OS: <code>{html.escape(_text(environment.get('os')) or 'not measured')}</code>; "
        f"Python: <code>{html.escape(_text(environment.get('python_version')) or 'not measured')}</code>; "
        f"serving device: <code>{html.escape(_text(benchmark.get('serving_device')) or 'not measured')}</code>."
        " Throughput is successful requests divided by their total measured duration; failed "
        "requests are excluded and missing values remain not measured.</p>"
        f"{reasons}</section>\n"
    )


def _failure_section(stages: Sequence[Mapping[str, Any]], benchmark: Mapping[str, Any]) -> str:
    failures: list[str] = []
    for stage in stages:
        resources = _mapping(stage.get("resources"))
        for name, label in (("multivae", "Mult-VAE"), ("lightgcn", "LightGCN")):
            fallback = _mapping(resources.get(name)).get("fallback_reason")
            if fallback:
                failures.append(
                    f"<li>Stage {_display(stage.get('percentage'), suffix='%')} {label} fallback: "
                    f"{html.escape(str(fallback))}</li>"
                )
        training = _mapping(stage.get("lhf_training"))
        for mode in ("known_user", "history_only"):
            reason = _mapping(training.get(mode)).get("fallback_reason")
            if reason:
                failures.append(
                    f"<li>Stage {_display(stage.get('percentage'), suffix='%')} {mode} LHF: "
                    f"{html.escape(str(reason))}</li>"
                )
    modes = _mapping(benchmark.get("modes"))
    for mode, label in (
        ("known_user", "Known-User"),
        ("history_only", "History-Only"),
        ("empty_history", "Empty-History"),
    ):
        value = _mapping(modes.get(mode))
        reason = value.get("reason")
        if reason:
            failures.append(f"<li>{label} latency benchmark: {html.escape(str(reason))}</li>")
        elif value.get("slo_status") == "fail":
            failures.append(
                f"<li>{label} latency SLO failed: measured p95 exceeded its target.</li>"
            )
    body = (
        "<p>No failure or fallback evidence was recorded in the aggregate inputs.</p>"
        if not failures
        else f"<ul>{''.join(failures)}</ul>"
    )
    return f'<section id="failure-cases"><h2>Failure cases and fallbacks</h2>{body}</section>\n'


def _provenance_section(stages: Sequence[Mapping[str, Any]], embedded: Mapping[str, Any]) -> str:
    active_stage = next((stage for stage in stages if stage.get("active")), {})
    provenance = _mapping(active_stage.get("provenance"))
    feature_banks = provenance.get("query_mode_feature_banks")
    artifacts = [
        {
            "percentage": stage.get("percentage"),
            "artifact_id": stage.get("artifact_id"),
            "snapshot_fingerprint": stage.get("snapshot_fingerprint"),
            "cutoff": _mapping(stage.get("provenance")).get("data_cutoff_percentage"),
        }
        for stage in stages
    ]
    compact = {
        "source_revision": embedded.get("source_revision"),
        "dataset_checksum": embedded.get("dataset_checksum"),
        "configuration": embedded.get("configuration"),
        "active_artifact_id": embedded.get("active_artifact_id"),
        "stage_artifacts": artifacts,
        "active_query_mode_feature_banks": feature_banks,
        "active_devices": _mapping(active_stage.get("resources")),
        "benchmark_method": _mapping(embedded.get("latency_benchmark")).get("method"),
    }
    return (
        "<table><tbody>"
        f"<tr><th>Source revision</th><td><code>{html.escape(_text(embedded.get('source_revision')))}</code></td></tr>"
        f"<tr><th>Dataset checksum</th><td><code>{html.escape(_text(embedded.get('dataset_checksum')))}</code></td></tr>"
        f"<tr><th>Active artifact</th><td><code>{html.escape(_text(embedded.get('active_artifact_id')))}</code></td></tr>"
        f"<tr><th>Random seed / configuration</th><td><code>{html.escape(str(_mapping(embedded.get('configuration')).get('random_seed', 'n/a')))}</code></td></tr>"
        f"<tr><th>Feature banks</th><td><code>{html.escape(json.dumps(feature_banks, sort_keys=True))}</code></td></tr>"
        "</tbody></table>"
        "<p>Stage artifacts, snapshot fingerprints, checksums, actual devices/fallbacks, and the "
        "benchmark method are retained as aggregate provenance. Raw Rating Events, per-user "
        "histories, validation pools, and model weights are intentionally not embedded in this HTML.</p>"
        f"<pre>{html.escape(json.dumps(compact, indent=2, sort_keys=True, ensure_ascii=False))}</pre>"
    )


def _headline_summary(
    stages: Sequence[Mapping[str, Any]], benchmark: Mapping[str, Any]
) -> dict[str, str]:
    evaluation = _stage_evaluation(stages, 90)
    if evaluation is None:
        quality = "not measured"
    else:
        known = _mapping(evaluation.get("known_user"))
        history = _mapping(evaluation.get("history_only"))
        quality = (
            f"Known-User LHF {_metric_with_denominator(_mapping(known.get('lhf')), 'EndToEndRecall@10')}; "
            f"History-Only LHF {_metric_with_denominator(_mapping(history.get('lhf')), 'EndToEndRecall@10')}"
        )
    modes = _mapping(benchmark.get("modes"))
    latency_parts: list[str] = []
    for mode, label in (
        ("known_user", "Known-User"),
        ("history_only", "History-Only"),
        ("empty_history", "Empty-History"),
    ):
        value = _mapping(modes.get(mode))
        p95 = value.get("p95_ms")
        status = _text(value.get("slo_status")) or "not measured"
        if isinstance(p95, (int, float)) and not isinstance(p95, bool):
            latency_parts.append(f"{label} p95 {float(p95):.2f} ms ({status})")
        else:
            latency_parts.append(f"{label} not measured ({status})")
    return {"quality": quality, "latency": "; ".join(latency_parts)}


def _mode_evaluation(stages: Sequence[Mapping[str, Any]], mode: str) -> Mapping[str, Any] | None:
    evaluation = _stage_evaluation(stages, 90)
    if evaluation is None:
        return None
    value = evaluation.get(mode)
    return value if isinstance(value, Mapping) else None


def _stage_evaluation(
    stages: Sequence[Mapping[str, Any]], percentage: int
) -> Mapping[str, Any] | None:
    for stage in stages:
        if stage.get("percentage") == percentage:
            value = stage.get("evaluation")
            return value if isinstance(value, Mapping) else None
    return None


def _metric_with_denominator(metrics: Mapping[str, Any], name: str) -> str:
    query_count = _integer(metrics.get("query_count"))
    if query_count is None or name not in metrics:
        return "n/a"
    return f"{_metric_value(metrics, name)} (n={query_count})"


def _metric_value(metrics: Mapping[str, Any], name: str, *, missing_if_empty: bool = False) -> str:
    query_count = _integer(metrics.get("query_count"))
    if missing_if_empty and (query_count is None or query_count == 0):
        return "n/a"
    raw = metrics.get(name)
    if (
        query_count is None
        or query_count == 0
        or not isinstance(raw, (int, float))
        or isinstance(raw, bool)
    ):
        return "n/a"
    return f"{float(raw):.4f}"


def _denominator_display(value: int | None) -> str:
    return "n/a" if value is None or value == 0 else f"n={value}/{value}"


def _headroom_display(evaluation: Mapping[str, Any]) -> str:
    denominator = evaluation.get("oracle_headroom_denominator")
    realized = evaluation.get("oracle_headroom_realized")
    if not isinstance(denominator, (int, float)) or denominator == 0:
        return "undefined (zero headroom denominator)"
    if not isinstance(realized, (int, float)) or realized < 0:
        return "not measured"
    return f"{float(realized):.4f}"


def _resource_display(value: Mapping[str, Any]) -> str:
    device = _text(value.get("actual_device")) or "n/a"
    fallback = value.get("fallback_reason")
    return f"{html.escape(device)}" + (
        f'<br><span class="small muted">{html.escape(str(fallback))}</span>' if fallback else ""
    )


def _slo_display(value: object) -> tuple[str, str]:
    if value == "pass":
        return "pass", "pass"
    if value == "fail":
        return "fail", "fail"
    return "not measured", "warn"


def _latency_value(value: object) -> str:
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        return "not measured"
    return f"{float(value):.3f}"


def _window_display(value: object) -> str:
    if isinstance(value, list) and len(value) == 2:
        return f"{html.escape(str(value[0]))}–{html.escape(str(value[1]))}%"
    return "none"


def _display(value: object, *, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        text = f"{value:.4f}"
    else:
        text = str(value)
    return html.escape(text + suffix)


def _display_name(name: str) -> str:
    return {"multivae": "Mult-VAE", "lightgcn": "LightGCN", "itemknn": "ItemKNN"}.get(name, name)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _numeric_mapping(value: object) -> dict[str, int | float]:
    raw = _mapping(value)
    return {
        str(key): child
        for key, child in raw.items()
        if isinstance(key, str) and isinstance(child, (int, float)) and not isinstance(child, bool)
    }


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: object) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _evaluation_window(value: object) -> list[int] | None:
    if isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) for item in value):
        return list(value)
    return None
