# Domain Docs

This repository uses a single-context domain documentation layout.

## Before exploring, read these

- `CONTEXT.md` at the repository root.
- Relevant ADRs under `docs/adr/`.

If these files do not exist, proceed silently. Do not create them preemptively.
The `/domain-modeling` skill creates them when domain terminology or architectural
decisions are actually established.

## File structure

/
├── CONTEXT.md
├── docs/
│   └── adr/
└── src/

## Use the glossary's vocabulary

When output names a domain concept—in an issue title, refactor proposal,
hypothesis, or test name—use the term defined in `CONTEXT.md`.

If a required concept is absent from the glossary, reconsider whether the term
belongs to the project or record the gap for `/domain-modeling`.

## Flag ADR conflicts

If proposed work contradicts an existing ADR, surface the conflict explicitly
instead of silently overriding the decision.
