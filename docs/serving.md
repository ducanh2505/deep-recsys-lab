# Serving

`RecommendationEngine` loads an explicit artifact directory, verifies its manifest and
checksums, reconstructs the declared runtime, maps external IDs, masks seen items, and emits
deterministically ordered recommendations. It has no dependency on BentoML.

## Request

```json
{
  "interactions": [
    {"item_id": "item-100", "value": 1.0, "timestamp": "2026-01-10T08:30:00Z"}
  ],
  "top_k": 10,
  "exclude_seen": true
}
```

Use `user_id` instead of `interactions` for a known-user query. Exactly one mode is required;
supplying both is invalid. Unknown identifiers, unsupported modes, and catalog exhaustion
produce structured domain errors.

## Response

```json
{
  "request_id": "request-correlation-id",
  "artifact_id": "sha256:<digest>",
  "recommendations": [
    {"item_id": "item-245", "score": 0.72, "metadata": {"category": "books"}}
  ]
}
```

`GET /model` returns the interaction/output schema, artifact identity, plugin, capabilities,
runtime, and catalog size. `/livez`, `/readyz`, and `/metrics` are public probes. If
`RECSYS_API_KEY` is set, clients provide it through `X-API-Key` on `/recommend` and `/model`.
Request bodies are bounded by `serving.max_body_bytes`.

## BentoML adapter

```bash
recsys serve --artifact var/models/<plugin>/<digest> --host 0.0.0.0 --port 3000
```

The command verifies the artifact before starting BentoML. The adapter reads the artifact
from the explicit read-only path supplied by the caller; Bento Model Store is not an
artifact source of truth.

Docker Compose runs only this API service. Set `RECSYS_ARTIFACT` to a host artifact directory
before starting it.
