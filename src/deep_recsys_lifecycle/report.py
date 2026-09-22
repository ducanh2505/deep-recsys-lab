from __future__ import annotations

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


def write_rolling_report(
    path: Path,
    *,
    stages: Sequence[Mapping[str, Any]],
    source_event_count: int,
    dataset_checksum: str,
    active_artifact_id: str,
    source_revision: str,
) -> None:
    """Write the compact static report for the rolling 50-to-100 showcase."""

    rows: list[str] = []
    evidence_rows: list[str] = []
    for stage in stages:
        percentage = int(stage.get("percentage", 0))
        evaluation_window = stage.get("evaluation_window")
        window_text = (
            f"{evaluation_window[0]}–{evaluation_window[1]}%"
            if isinstance(evaluation_window, list) and len(evaluation_window) == 2
            else "none"
        )
        evaluation = stage.get("evaluation")
        known = _rolling_mode_summary(evaluation, "known_user")
        history = _rolling_mode_summary(evaluation, "history_only")
        rows.append(
            "<tr>"
            f"<td>{percentage}%</td>"
            f"<td>{int(stage.get('snapshot_event_count', 0))}</td>"
            f"<td><code>{html.escape(str(stage.get('artifact_id', '')))}</code></td>"
            f"<td>{html.escape(window_text)}</td>"
            f"<td>{'yes' if stage.get('active') else 'no'}</td>"
            f"<td>{_rolling_metric(known, 'lhf', 'EndToEndRecall@10')}</td>"
            f"<td>{_rolling_metric(known, 'popularity', 'EndToEndRecall@10')}</td>"
            f"<td>{_rolling_metric(history, 'lhf', 'EndToEndRecall@10')}</td>"
            f"<td>{_rolling_metric(history, 'popularity', 'EndToEndRecall@10')}</td>"
            "</tr>"
        )
        manifest = stage.get("artifact_manifest", {})
        if isinstance(manifest, Mapping):
            multivae = manifest.get("multivae_training", {})
            lightgcn = manifest.get("lightgcn_training", {})
        else:
            multivae = {}
            lightgcn = {}
        timings = stage.get("timings", {})
        evidence_rows.append(
            "<tr>"
            f"<td>{percentage}%</td>"
            f"<td>{html.escape(str(_evidence_value(multivae, 'actual_device')))}</td>"
            f"<td>{html.escape(str(_evidence_value(lightgcn, 'actual_device')))}</td>"
            f"<td>{html.escape(str(_evidence_value(timings, 'training')))}</td>"
            f"<td>{html.escape(str(_evidence_value(timings, 'export')))}</td>"
            f"<td>{html.escape(str(_evidence_value(timings, 'smoke')))}</td>"
            f"<td>{html.escape(str(_evidence_value(timings, 'evaluation')))}</td>"
            "</tr>"
        )

    embedded = {
        "source_event_count": source_event_count,
        "dataset_checksum": dataset_checksum,
        "active_artifact_id": active_artifact_id,
        "source_revision": source_revision,
        "stages": [dict(stage) for stage in stages],
    }
    data_json = html.escape(json.dumps(embedded, indent=2, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html>\n"
        '<html lang="en">\n'
        '<head><meta charset="utf-8"><title>Rolling Movie Recommender Lifecycle</title>'
        "<style>body{font-family:system-ui;max-width:90rem;margin:2rem auto;line-height:1.5}"
        "code,pre{background:#f4f4f4;padding:.25rem}section{margin:2rem 0}"
        "table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border:1px solid #ddd;"
        "padding:.45rem;text-align:left}th{background:#f4f4f4}</style></head>\n"
        "<body>\n"
        "<h1>Rolling 50-to-100 Movie Recommender Lifecycle</h1>\n"
        '<section id="rolling-overview"><h2>Lifecycle overview</h2>'
        f"<p>{source_event_count} source Rating Events advance through six cumulative Data "
        "Snapshots. The Event Store ingests each Future Window only after its prior stage has "
        "been evaluated.</p>"
        f"<p>Active 100% artifact: <code>{html.escape(active_artifact_id)}</code></p></section>\n"
        '<section id="rolling-stages"><h2>Stages 50% → 100%</h2>'
        "<table><thead><tr><th>Stage</th><th>Snapshot Events</th><th>Artifact</th>"
        "<th>Evaluation Window</th><th>Active</th><th>Known-User LHF Recall@10</th>"
        "<th>Known-User Popularity Recall@10</th><th>History-Only LHF Recall@10</th>"
        f"<th>History-Only Popularity Recall@10</th></tr></thead><tbody>"
        f"{''.join(rows)}</tbody></table>"
        "</section>\n"
        '<section id="evaluation-boundaries"><h2>Evaluation boundaries</h2>'
        "<p>Stages 50–90 report Known-User and History-Only quality against the next Future "
        "Window before ingestion. Baselines include Popularity, ItemKNN, Mult-VAE, LightGCN, "
        "RRF, and the diagnostic Oracle Union in the embedded stage records.</p>"
        "<p><strong>Stage 100 has no Future Window and carries no future-quality claim.</strong> "
        "The final headline quality is the 90% → 100% evaluation.</p></section>\n"
        '<section id="training-device-evidence"><h2>Training and device evidence</h2>'
        "<table><thead><tr><th>Stage</th><th>Mult-VAE device</th><th>LightGCN device</th>"
        "<th>Training seconds</th><th>Export seconds</th><th>Smoke seconds</th>"
        f"<th>Evaluation seconds</th></tr></thead><tbody>{''.join(evidence_rows)}</tbody></table>"
        "</section>\n"
        '<section id="provenance"><h2>Provenance</h2>'
        f"<p>Dataset checksum: <code>{html.escape(dataset_checksum)}</code>; source revision: "
        f"<code>{html.escape(source_revision)}</code>.</p>"
        f"<pre>{data_json}</pre></section>\n"
        "</body>\n</html>\n",
        encoding="utf-8",
    )


def _rolling_mode_summary(value: object, mode: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    summary = value.get(mode)
    return summary if isinstance(summary, Mapping) else {}


def _rolling_metric(summary: Mapping[str, Any], name: str, metric: str) -> str:
    if name == "popularity":
        retrievers = summary.get("retrievers", {})
        value = retrievers.get("popularity") if isinstance(retrievers, Mapping) else None
    elif name == "lhf":
        value = summary.get("lhf")
    else:
        value = None
    if not isinstance(value, Mapping):
        return "n/a"
    raw = value.get(metric)
    return f"{float(raw):.4f}" if isinstance(raw, (int, float)) else "n/a"


def _evidence_value(value: object, key: str) -> object:
    return value.get(key, "n/a") if isinstance(value, Mapping) else "n/a"
