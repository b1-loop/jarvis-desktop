# 0001. Record architecture decisions

Date: 2026-10-21
Status: Accepted

## Context

Jarvis is a portfolio project that will be read by people who were not present
when its design choices were made. Chat history and commit messages do not
survive as a readable narrative. Reviewers consistently ask *why*, not *what*.

## Decision

We keep Architecture Decision Records in `docs/adr/`, one file per decision,
in the lightweight format popularised by Michael Nygard. Each record states the
context, the decision, the alternatives rejected and the consequences. Records
are immutable once accepted; a change is a new record that supersedes the old.

## Alternatives considered

- **Design section in README only** – drifts out of date and mixes rationale with usage.
- **Wiki** – lives outside the repo and is not reviewed with the code.

## Consequences

Slightly more writing per significant change. In return the repository explains
itself, and `CLAUDE.md` can require an ADR for every non-obvious decision.
