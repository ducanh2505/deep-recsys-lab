# Store academic sources metadata first

The Git repository stores a versioned Research Source Record for every consulted academic
work and does not archive source binaries by default. Each Record identifies the citation,
persistent identifier and version, retrieval location and date, checksum when obtainable,
and redistribution status. Committing every PDF would improve offline access but permanently
grow repository history and may redistribute material without permission; storing only a
bare URL would stay small but would not identify the exact version used. Full text enters
version control only when redistribution is explicitly permitted and the file satisfies the
repository size policy, using large-file or artifact storage when appropriate. Otherwise it
remains in an ignored local cache or authorized external store. The Lab accepts that some
sources require retrieval during setup in exchange for durable citations without treating
Git as an indiscriminate paper mirror.
