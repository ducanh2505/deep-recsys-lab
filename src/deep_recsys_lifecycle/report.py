from __future__ import annotations

import html
import json
from pathlib import Path

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
        "multivae_resource_evidence": artifact.manifest.get("multivae_training", {}),
        "lightgcn_resource_evidence": artifact.manifest.get("lightgcn_training", {}),
    }
    data_json = html.escape(json.dumps(embedded_data, indent=2, sort_keys=True))
    evaluation_sections = _evaluation_sections(evaluation)
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
        "<p>Leakage-free evaluation compares Popularity, ItemKNN, Mult-VAE, and LightGCN "
        "Candidate Pools from one temporal lifecycle stage.</p>\n"
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
        f"{_multivae_resource_section(artifact)}"
        f"{_lightgcn_resource_section(artifact)}"
        '<section id="query-modes"><h2>Query modes</h2>'
        "<p>Known-User, History-Only, and Empty-History routes all exclude supplied history "
        "where applicable and return unseen Candidates. History-Only and Empty-History queries "
        "do not support LightGCN. The public API remains on Popularity until Issue #39 adds "
        "fusion.</p>"
        f"<pre>{data_json}</pre></section>\n"
        "</body>\n</html>\n",
        encoding="utf-8",
    )


def _evaluation_sections(evaluation: EvaluationReport | None) -> str:
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

    rows = ""
    for name, retriever_evaluation in (*evaluation.retrievers.items(), ("RRF", evaluation.rrf)):
        metrics = retriever_evaluation.metrics
        display_name = {"multivae": "Mult-VAE", "lightgcn": "LightGCN"}.get(name, name)
        rows += (
            "<tr>"
            f"<td>{html.escape(display_name)}</td>"
            f"<td>{metrics.coverage_at_200:.4f}</td>"
            f"<td>{metrics.conditional_recall_at_10:.4f}</td>"
            f"<td>{metrics.end_to_end_recall_at_10:.4f}</td>"
            f"<td>{metrics.ndcg_at_10:.4f}</td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr><th>Retriever</th><th>Coverage@200</th>"
        "<th>ConditionalRecall@10</th><th>EndToEndRecall@10</th><th>NDCG@10</th>"
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
        '<section id="oracle-union"><h2>Oracle Union</h2>'
        f"<p>Diagnostic-only unbudgeted union coverage: "
        f"{evaluation.oracle_union_coverage:.4f}. It is not served.</p></section>\n"
        '<section id="final-ranking-quality"><h2>Final ranking quality</h2>'
        "<p>ConditionalRecall@10 and NDCG@10 describe ranking quality after retrieval coverage."
        "</p></section>\n"
    )


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
        "</tbody></table>"
    )


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
