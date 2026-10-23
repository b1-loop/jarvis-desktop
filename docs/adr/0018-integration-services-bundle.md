# 0018. One IntegrationServices bundle on ToolContext

Date: 2026-10-22
Status: Accepted

## Context

Step 8 adds twenty-one tools backed by external services: email, calendar,
music, chat, smart home (the providers of ADR-0009) and web search, page
reading and YouTube transcripts (ADR-0012). ADR-0006 says tools receive
services through `ToolContext` and never import a concrete provider. Three
questions followed:

- How do eight kinds of service reach the tools without `ToolContext`
  growing a field per service, every one of which the composition root and
  every test must then fill?
- Gmail and Outlook ids are only unique per provider, yet listings merge
  both mailboxes. How does an id the LLM passes back find its provider?
- `draft_email_reply` and `send_email` are separate plugins that may not
  import each other, and the email interface cannot look a draft up by id.
  Where does a draft live between "here is the reply" and "send it"?

## Decision

**One optional field, one frozen bundle.** `ToolContext.integrations` is an
`IntegrationServices | None` (`integrations/services.py`): tuples of
provider ABCs per account-backed service, the `WebSearchProvider` and
`TranscriptFetcher` Protocols, the shared `httpx.AsyncClient` for page
reads, and the session's `EmailDraftBook`. The composition root builds it
in one call, `build_integration_services(settings, http)`, which reuses one
`IntegrationFactory` (so Google and Microsoft sign-ins stay shared) and
picks Brave or DuckDuckGo search from settings. (An `IntegrationProviders`
alias kept for the old bootstrap code was removed once nothing used it.)

**Missing means "not configured", never a crash.** `None` and an empty
bundle are treated alike: every integration tool returns `ok=False` with a
spoken "that integration isn't configured" and the environment variables
to set, so a text-only run without wiring still answers sensibly.

**Provider-qualified references.** Every id handed to the LLM is
`"<provider>:<id>"` (`gmail:18c2…`, `stub-email:stub-mail-1`), minted and
resolved in one place. Splitting is on the first colon only, so provider
ids that contain colons survive.

**Session-scoped draft book.** Drafts created by `draft_email_reply` are
kept in the bundle's `EmailDraftBook` under their reference until sent.
They are not persisted; after a restart the user dictates the reply again.

**Confirmation restates the draft.** `ConfirmationPolicy` builds the prompt
from the validated arguments only, so `send_email` takes the draft id plus
its recipients, subject and first line, the prompt reads those back, and
`run` refuses to send when they differ from the stored draft.

## Alternatives considered

- **A field per service on `ToolContext`** – eight fields, each a merge
  conflict with every other branch touching the context, and every test
  building contexts would need to know them all.
- **Tools pulling providers from a global registry** – shorter wiring,
  but hidden dependencies and no way to run two differently wired contexts
  (tests, a future multi-user front end) side by side.
- **Passing the context to `confirmation_prompt`** – lets `send_email` look
  the draft up for the prompt, but changes the `Tool` contract for every
  plugin. Restating three fields costs the LLM a few tokens and turns the
  check into a safety property.
- **Persisting drafts (SQLite)** – survives restarts, but a stale draft
  that the server has since deleted or the user edited elsewhere is worse
  than asking again.

## Consequences

- Adding a service is one field on `IntegrationServices`, one line in the
  builder and one accessor in `tools/_integration_support.py`; no tool or
  test that does not use it changes.
- The bundle imports `factory`, `web` and `youtube` at module level; real
  provider SDKs stay lazy, so a stub-only checkout still imports cleanly.
- References make ids opaque to the user but unambiguous to the LLM; the
  spoken output never reads them aloud.
- The draft book is the one mutable member of a frozen bundle, documented
  as deliberate session state.
