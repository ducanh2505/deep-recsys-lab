# Use Kafka with an append-only Event Store

Rating Events cross a real Kafka boundary before a consumer persists them as append-only
Parquet batches. A single local broker and one ordered topic provide the ingestion behaviour
needed for the showcase without a distributed cluster, while at-least-once delivery and
deterministic event identifiers keep the design honest without introducing exactly-once
infrastructure.
