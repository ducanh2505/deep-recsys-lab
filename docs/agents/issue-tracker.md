# Issue tracker: GitHub

Issues and specs for this repository live in GitHub Issues at
`ducanh2505/deep-recsys-lab`. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`
- **Read an issue**: `gh issue view <number> --comments`, including its labels.
- **List issues**: use `gh issue list` with the appropriate label and state filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply or remove labels**: `gh issue edit <number> --add-label "..."` or
  `gh issue edit <number> --remove-label "..."`
- **Close an issue**: `gh issue close <number> --comment "..."`

The `gh` CLI infers the repository from `git remote -v` when run inside this clone.

## Pull requests as a triage surface

**PRs as a request surface: no.**

External pull requests are not included automatically in the triage queue. A pull
request explicitly named by the maintainer may still be inspected directly.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.

## Wayfinding operations

The `/wayfinder` skill represents a map as one GitHub issue with child issues as
tickets.

- Label the map issue `wayfinder:map`.
- Label child tickets with `wayfinder:<type>`, where type is `research`,
  `prototype`, `grilling`, or `task`.
- Use GitHub sub-issues and native issue dependencies when available.
- If those features are unavailable, record `Part of #<map>` and
  `Blocked by: #<number>` in the issue body.
- A ticket enters the frontier when all its blockers are closed and it has no
  assignee.
- Claim a ticket by assigning it to the driving developer.
- Resolve a ticket by posting its result, closing it, and adding a context
  pointer to the map's decisions.
