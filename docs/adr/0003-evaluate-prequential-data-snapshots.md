# Evaluate prequential Data Snapshots

The lifecycle starts with a 50% Data Snapshot and advances in ten-percent increments, using
each next increment as the current snapshot's Future Window before ingesting it. This design
makes the report describe time-respecting model behaviour and retraining rather than a random
offline split; the 100% Serving Artifact therefore has no claimed future-window quality, and
the final headline quality comes from the 90% snapshot.
