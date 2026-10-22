# 0009. Integration providers: interfaces, stub by default, least privilege

Date: 2026-10-21
Status: Accepted

## Context

Phase 2 connects Jarvis to the user's email (Gmail, Outlook), calendar
(Google, Outlook), music (Spotify), chat (Slack, Discord, Teams) and smart
home (Home Assistant, Philips Hue). Each service has its own SDK, auth flow,
data shapes and failure modes, yet the agent needs one question answered the
same way everywhere: "what is unread?", "when am I free?", "play this".

Three constraints shape the design:

- The project must run, demo and test **without any credentials**: a
  reviewer cloning the repository, the CI runner, and the developer in
  text-only mode all need working providers.
- `CLAUDE.md` requires **least-privilege OAuth scopes** and no secrets in
  code or logs, and stubs that are obvious in the class name.
- Tests may not touch the network, so every I/O path must be injectable.

## Decision

**One interface per service, provider-neutral models, stubs by default.**
`integrations/base.py` defines ABCs (`EmailProvider`, `CalendarProvider`,
`MusicProvider`, `ChatProvider`, `SmartHomeProvider`) over frozen dataclasses
(`EmailMessage`, `CalendarEvent`, `Track`, `ChatMessage`, `Device`). Tools
depend only on these. `factory.py` builds the real provider when its settings
are present and the `Stub*` provider otherwise, and logs the choice with
`stub=True`. Stubs hold realistic fixtures, record every call and never send
anything; their status reports `configured=False, stub=True` so the dashboard
can never mistake one for a live account.

**Shared behaviour lives in the interface.** `today`, `upcoming` and
`find_free_slots` are implemented once on `CalendarProvider` from `agenda`,
so the date arithmetic is tested once and every provider (including the
stub) behaves identically.

**Least-privilege scopes, one consent per identity.**

| Provider | Scopes |
| --- | --- |
| Gmail | `gmail.readonly`, `gmail.compose` |
| Google Calendar | `calendar.readonly`, `calendar.events` |
| Outlook | `Mail.ReadWrite`, `Mail.Send`, `Calendars.ReadWrite` |
| Teams (opt-in, work or school accounts) | `Chat.Read`, `ChatMessage.Send` |
| Spotify | `user-read-playback-state`, `user-modify-playback-state` |
| Slack / Discord / Home Assistant | bot or long-lived token; what the admin granted |

Nothing modifies or deletes mail, nothing reads contacts or profiles, nothing
touches the Spotify library. *Amended in step 8:* the reply draft that the
user approves before sending needs `gmail.compose` (Gmail's `drafts.create`
and `drafts.send` reject `gmail.send`) and `Mail.ReadWrite` (Graph's
`createReply` rejects `Mail.Read`). `Mail.ReadWrite` is broader than the
feature needs; the code still only creates and sends reply drafts. Gmail and Calendar share one `GoogleAuth`, and
Outlook and Teams share one `MicrosoftAuth`, each requesting the union of its
providers' scopes: one consent screen per account rather than one per
feature, with no scope that a single feature would not need anyway.

**OAuth state behind `TokenStore`.** Authorized-user JSON (Google) and the
serialised msal cache (Microsoft) are persisted as JSON files under
`Settings.data_dir / "tokens"`, never in the repository; spotipy keeps its
own cache file in the same directory. The blocking consent flows run in a
worker thread behind an `asyncio.Lock`, so a burst of tool calls on a fresh
install produces one prompt.

**REST over SDK clients wherever the SDK adds little.** Gmail, Calendar,
Graph, Spotify, Discord and Home Assistant are called with the injected
`httpx.AsyncClient` through one `RestSession` that adds the bearer token,
maps 401/403 to `AuthError`, 429 to `RateLimitedError` and everything else
to `ProviderError`, and records the last error for the dashboard. The SDKs
are kept only for what they do best: google-auth-oauthlib and msal for the
consent flows, spotipy for the authorization-code flow, slack-sdk's
`AsyncWebClient.api_call` behind a one-method Protocol.

**Discord and Teams via REST, not gateway/bot clients.** A Discord gateway
client or a Teams bot would keep a connection and an event loop integration
alive for the whole session to answer "what did I miss?". On-demand REST
calls answer that with nothing running in the background. The limits are
documented in the modules: Discord bots see only channels reachable by id or
listed in `DISCORD_GUILD_ID` and have no unread state; Teams reads chats
(1:1, group, meeting), not team channels.

**Philips Hue through Home Assistant.** Hue lights are ordinary `light.*`
entities once the Hue integration is enabled in Home Assistant, so a
separate Hue provider would duplicate the interface and move bridge
credentials into Jarvis for no gain.

## Alternatives considered

- **google-api-python-client / spotipy.Spotify / discord.py for calls** –
  synchronous, dynamically typed (discovery documents fetched over the
  network at build time) and awkward to fake; httpx + `MockTransport` tests
  request shaping exactly.
- **Protocols instead of ABCs** – rejected because the calendar interface
  carries real shared behaviour and `OutlookProvider` implements two
  interfaces at once, which ABCs express directly.
- **One narrow token per feature** – more consent prompts for the same
  account with no privacy gain; the union of scopes is still the minimum for
  the enabled features.
- **A single "unified inbox" provider** – would push merging policy into the
  integration layer; the factory returns a list and the tool layer decides.
- **Returning the stub silently when credentials are present but broken** –
  never. Real providers raise `NotConfiguredError` naming the exact
  variables, and auth failures surface as `AuthError`.

## Consequences

- Tools and the agent are written once against `base.py`; adding a provider
  is one module plus a factory branch.
- Every provider is unit-tested without network: `RestSession` with
  `MockTransport`, auth flows through injected seams, Slack through a fake
  `api_call`.
- Each service's first real use triggers a consent flow (browser for Google
  and Spotify, device code for Microsoft); the token store makes it a
  one-time event. *Amended (connect accounts):* that is only true in a
  terminal. The windowless desktop app and voice mode have no console, so
  a browser opened by a hidden process or a device code written to the log
  would be a sign-in nobody can complete. `GoogleAuth` and `MicrosoftAuth`
  therefore have a non-interactive mode, the default everywhere except
  `--text-only`: when consent is needed they raise `ConsentRequiredError`
  whose message is the spoken remedy ("Google isn't connected yet; run
  `uv run python scripts/connect_accounts.py google`"), and the first such
  error per account becomes one system notification. Consent happens once,
  in the foreground, through `scripts/connect_accounts.py` (also
  `jarvis connect`), which uses the same token sources, scopes and store
  and then verifies each product with one cheap call. A refresh that fails
  for network reasons is a `ProviderError` and keeps the stored token.
- *Amended (connect accounts):* Teams is opt-in (`JARVIS_TEAMS_ENABLED`,
  `connect_accounts.py microsoft --teams`). The `Chat.*` scopes are only
  grantable by work or school accounts, and requesting them for a personal
  account can fail the whole Microsoft sign-in, Outlook included, so the
  default Microsoft consent asks for the Outlook scopes only.
- The `integrations` extra is optional: the factory imports real provider
  modules only when their settings are set, so a stub-only checkout needs
  none of the SDKs.
- The msal app is constructed lazily on first use because its constructor
  performs tenant discovery over the network.
- Listing mail on Gmail is N+1 (ids, then metadata per id, concurrently);
  the batch endpoint is a known optimisation if listings grow.
