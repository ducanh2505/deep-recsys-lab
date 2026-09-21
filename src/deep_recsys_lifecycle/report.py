from __future__ import annotations

import html
import json
from pathlib import Path

from .artifact import ServingArtifact
from .event_store import DataSnapshot
from .positive import derive_positive_interactions
from .query import (
    EmptyHistoryQuery,
    HistoryOnlyQuery,
    KnownUserQuery,
    RecommendationService,
)


def write_static_report(
    path: Path,
    snapshot: DataSnapshot,
    artifact: ServingArtifact,
) -> None:
    """Write a self-contained HTML report for the Popularity tracer bullet."""

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
    embedded_data = {
        "snapshot": snapshot.to_dict(),
        "artifact_manifest": artifact.manifest,
        "positive_interaction_count": positive_count,
        "query_examples": examples,
    }
    data_json = html.escape(json.dumps(embedded_data, indent=2, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!doctype html>\n"
        '<html lang="en">\n'
        '<head><meta charset="utf-8"><title>Movie Recommender Lifecycle Showcase</title>'
        "<style>body{font-family:system-ui;max-width:62rem;margin:2rem auto;line-height:1.5}"
        "code,pre{background:#f4f4f4;padding:.25rem}section{margin:2rem 0}</style></head>\n"
        "<body>\n"
        "<h1>Movie Recommender Lifecycle Showcase</h1>\n"
        "<p>Popularity lifecycle tracer bullet: source Rating Events become "
        "served Candidates.</p>\n"
        '<section id="data-snapshot"><h2>Data Snapshot</h2>'
        f"<p>{snapshot.event_count} deduplicated Rating Events across "
        f"{snapshot.source_batch_count} append-only batches.</p>"
        f"<p>Fingerprint: <code>{html.escape(snapshot.fingerprint)}</code></p></section>\n"
        '<section id="positive-interactions"><h2>Positive Interactions</h2>'
        f"<p>{positive_count} Rating Events met the {threshold:.1f} threshold.</p></section>\n"
        '<section id="popularity"><h2>Popularity</h2>'
        "<p>Popularity is the only Candidate Retriever in this tracer bullet.</p></section>\n"
        '<section id="query-modes"><h2>Query modes</h2>'
        "<p>Known-User, History-Only, and Empty-History routes all exclude supplied history "
        "where applicable and return unseen Candidates.</p>"
        f"<pre>{data_json}</pre></section>\n"
        "</body>\n</html>\n",
        encoding="utf-8",
    )
