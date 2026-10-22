"""Shared plumbing for the integration tools (not a plugin itself).

The leading underscore keeps the registry from treating this module as a
plugin. It holds what every integration tool needs and none should
re-implement: resolving providers from ``ToolContext.integrations``, the
uniform "that integration isn't configured" answer, fan-out across several
providers of one service, and JSON-able renderings of the provider models so
the LLM sees one shape for an email whether it came from Gmail or Outlook.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final

import httpx
from pydantic import JsonValue

from jarvis.agent.tool import ToolContext, ToolResult
from jarvis.integrations.base import (
    CalendarEvent,
    CalendarProvider,
    ChatMessage,
    ChatProvider,
    Device,
    EmailMessage,
    EmailProvider,
    IntegrationError,
    MusicProvider,
    Provider,
    SmartHomeProvider,
    Track,
)
from jarvis.integrations.services import (
    IntegrationServices,
    UnknownReferenceError,
    make_reference,
    split_reference,
)
from jarvis.integrations.web import WebSearchProvider
from jarvis.integrations.youtube import TranscriptFetcher
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

ELLIPSIS: Final = "…"
_HOURS_ON_A_CLOCK_FACE: Final = 12
_DAYS_IN_A_WEEK: Final = 7


class Service(StrEnum):
    """The external services integration tools depend on."""

    EMAIL = "email"
    CALENDAR = "calendar"
    MUSIC = "music"
    CHAT = "messaging"
    SMART_HOME = "smart home"
    WEB_SEARCH = "web search"
    WEB_PAGES = "web page reading"
    YOUTUBE = "YouTube transcript"


SETUP_VARIABLES: Final[Mapping[Service, tuple[str, ...]]] = {
    Service.EMAIL: ("GOOGLE_CLIENT_SECRETS_FILE", "MS_CLIENT_ID"),
    Service.CALENDAR: ("GOOGLE_CLIENT_SECRETS_FILE", "MS_CLIENT_ID"),
    Service.MUSIC: ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"),
    Service.CHAT: ("SLACK_BOT_TOKEN", "DISCORD_BOT_TOKEN", "MS_CLIENT_ID"),
    Service.SMART_HOME: ("HOME_ASSISTANT_URL", "HOME_ASSISTANT_TOKEN"),
    Service.WEB_SEARCH: ("BRAVE_SEARCH_API_KEY",),
    Service.WEB_PAGES: ("JARVIS_WEB_USER_AGENT",),
    Service.YOUTUBE: ("JARVIS_WEB_USER_AGENT",),
}
"""Environment variables that enable each service, as named in ``.env.example``.

Web search, page reading and YouTube need no credentials; for them the
variables are optional tuning and a missing service means the composition
root did not wire integrations at all.
"""


def not_configured(service: Service) -> ToolResult:
    """The uniform answer when a tool's service is absent; never an exception."""
    names = SETUP_VARIABLES[service]
    listed = " or ".join(names)
    return ToolResult.failure(
        f"That integration isn't configured: {service.value}. Set {listed} in your .env file.",
        detail=(
            f"No {service.value} provider is available in ToolContext.integrations. "
            f"Set {listed} in .env (see .env.example) and make sure the composition root "
            "builds the services with build_integration_services()."
        ),
        data={"configured": False, "service": service.value, "env_vars": list(names)},
    )


def _services(context: ToolContext) -> IntegrationServices:
    return context.integrations if context.integrations is not None else IntegrationServices()


def email_providers(context: ToolContext) -> tuple[EmailProvider, ...]:
    """Email providers wired into ``context``; empty when none are."""
    return _services(context).email


def calendar_providers(context: ToolContext) -> tuple[CalendarProvider, ...]:
    """Calendar providers wired into ``context``; empty when none are."""
    return _services(context).calendar


def music_providers(context: ToolContext) -> tuple[MusicProvider, ...]:
    """Music providers wired into ``context``; empty when none are."""
    return _services(context).music


def chat_providers(context: ToolContext) -> tuple[ChatProvider, ...]:
    """Chat providers wired into ``context``; empty when none are."""
    return _services(context).chat


def smart_home_providers(context: ToolContext) -> tuple[SmartHomeProvider, ...]:
    """Smart-home providers wired into ``context``; empty when none are."""
    return _services(context).smart_home


def web_search_provider(context: ToolContext) -> WebSearchProvider | None:
    """The search backend, or ``None``."""
    return _services(context).web_search


def transcript_fetcher(context: ToolContext) -> TranscriptFetcher | None:
    """The YouTube caption source, or ``None``."""
    return _services(context).transcript_fetcher


def web_client(context: ToolContext) -> httpx.AsyncClient | None:
    """The HTTP client for page reads, or ``None``."""
    return _services(context).web_client


# ---------------------------------------------------------------------------
# Fan-out across providers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Gathered[P: Provider, T]:
    """Per-provider results of one fan-out call.

    Attributes:
        results: ``(provider, value)`` for every provider that answered.
        failures: ``(provider, error)`` for every provider that failed.
    """

    results: list[tuple[P, T]] = field(default_factory=list)
    failures: list[tuple[P, IntegrationError]] = field(default_factory=list)

    def failure_note(self) -> str:
        """One line naming the failed providers, empty when none failed."""
        if not self.failures:
            return ""
        parts = [f"{provider.provider_name}: {error}" for provider, error in self.failures]
        return "Some providers failed: " + "; ".join(parts)

    def failure_data(self) -> list[JsonValue]:
        """The failures as JSON for ``ToolResult.data``."""
        return [
            {"provider": provider.provider_name, "error": str(error)}
            for provider, error in self.failures
        ]


async def gather_each[P: Provider, T](
    providers: Sequence[P], call: Callable[[P], Awaitable[T]]
) -> Gathered[P, T]:
    """Run ``call`` on every provider concurrently and keep partial results.

    A Gmail outage should not hide the Outlook inbox, so integration errors
    from some providers are logged and reported alongside the others'
    results. When every provider fails the first error is raised, and any
    non-integration exception (a bug) is always re-raised.
    """
    outcomes = await asyncio.gather(*(call(p) for p in providers), return_exceptions=True)
    gathered: Gathered[P, T] = Gathered()
    for provider, outcome in zip(providers, outcomes, strict=True):
        if isinstance(outcome, IntegrationError):
            log.warning(
                "tools.provider_failed", provider=provider.provider_name, error=str(outcome)
            )
            gathered.failures.append((provider, outcome))
        elif isinstance(outcome, BaseException):
            raise outcome
        else:
            gathered.results.append((provider, outcome))
    if not gathered.results and gathered.failures:
        raise gathered.failures[0][1]
    return gathered


def flatten[P: Provider, T](gathered: Gathered[P, list[T]]) -> list[tuple[P, T]]:
    """``(provider, item)`` pairs from a fan-out whose calls returned lists."""
    return [(provider, item) for provider, items in gathered.results for item in items]


async def with_bodies(
    pairs: Sequence[tuple[EmailProvider, EmailMessage]], count: int
) -> list[tuple[EmailProvider, EmailMessage]]:
    """Replace the first ``count`` listing entries with their full messages.

    Listings carry only snippets; questions like "what did the client say
    about the invoice?" need the body. A body that cannot be fetched keeps
    its snippet (the failure is logged), so one broken message does not
    sink the whole answer.
    """
    head = pairs[:count]
    fetched = await asyncio.gather(
        *(provider.get(message.id) for provider, message in head), return_exceptions=True
    )
    enriched: list[tuple[EmailProvider, EmailMessage]] = []
    for (provider, message), outcome in zip(head, fetched, strict=True):
        if isinstance(outcome, IntegrationError):
            log.warning(
                "tools.email_body_failed", provider=provider.provider_name, error=str(outcome)
            )
            enriched.append((provider, message))
        elif isinstance(outcome, BaseException):
            raise outcome
        else:
            enriched.append((provider, outcome))
    return [*enriched, *pairs[count:]]


def provider_named[P: Provider](providers: Sequence[P], name: str) -> P:
    """The provider whose ``provider_name`` is ``name``.

    Raises:
        UnknownReferenceError: If none matches; the message lists the choices.
    """
    for provider in providers:
        if provider.provider_name == name:
            return provider
    choices = ", ".join(p.provider_name for p in providers) or "none"
    raise UnknownReferenceError(name, f"no such provider (available: {choices})")


def resolve_reference[P: Provider](providers: Sequence[P], reference: str) -> tuple[P, str]:
    """Split ``"<provider>:<id>"`` and find the provider that issued it."""
    provider_name, item_id = split_reference(reference)
    return provider_named(providers, provider_name), item_id


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def trim(text: str, limit: int) -> str:
    """Cut ``text`` to ``limit`` characters on a word boundary, marking the cut."""
    if len(text) <= limit:
        return text
    longest = max(0, limit - len(ELLIPSIS))
    cut = text.rfind(" ", 0, longest + 1)
    return text[: cut if cut > 0 else longest].rstrip() + ELLIPSIS


def first_line(text: str) -> str:
    """The first non-blank line of ``text``, stripped; empty if there is none."""
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def spoken_list(items: Sequence[str]) -> str:
    """``["a", "b", "c"]`` -> ``"a, b and c"``, the way it is said aloud."""
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def count_phrase(count: int, singular: str, plural: str | None = None) -> str:
    """``(1, "email")`` -> ``"1 email"``, ``(0, "email")`` -> ``"no emails"``."""
    word = singular if count == 1 else plural or f"{singular}s"
    return f"{count or 'no'} {word}"


def display_name(sender: str) -> str:
    """``"Maya Lindqvist <maya@example.com>"`` -> ``"Maya Lindqvist"``."""
    name = sender.split("<", 1)[0].strip().strip('"')
    return name or sender.strip("<> ")


def twelve_hour(moment: datetime) -> str:
    """``15:05`` -> ``"3:05 PM"`` in ``moment``'s own zone; ``%-I`` is not portable."""
    hour = moment.hour % _HOURS_ON_A_CLOCK_FACE or _HOURS_ON_A_CLOCK_FACE
    meridiem = "AM" if moment.hour < _HOURS_ON_A_CLOCK_FACE else "PM"
    return f"{hour}:{moment:%M} {meridiem}"


def spoken_clock(moment: datetime, now: datetime) -> str:
    """``moment`` converted to the user's zone (that of ``now``), twelve-hour clock."""
    return twelve_hour(moment.astimezone(now.tzinfo))


def spoken_day(moment: datetime, now: datetime) -> str:
    """``"today"``, ``"tomorrow"``, ``"on Friday"`` or ``"on 3 November"``."""
    local = moment.astimezone(now.tzinfo)
    days = (local.date() - now.date()).days
    if days == 0:
        return "today"
    if days == 1:
        return "tomorrow"
    if 1 < days < _DAYS_IN_A_WEEK:
        return f"on {local:%A}"
    return f"on {local.day} {local:%B}"


def spoken_when(moment: datetime, now: datetime) -> str:
    """``"tomorrow at 9:00 AM"``."""
    return f"{spoken_day(moment, now)} at {spoken_clock(moment, now)}"


def spoken_minutes(duration: timedelta) -> str:
    """``timedelta(minutes=90)`` -> ``"90 minutes"``."""
    return count_phrase(round(duration.total_seconds() / 60), "minute")


# ---------------------------------------------------------------------------
# JSON renderings of the provider models
# ---------------------------------------------------------------------------


def email_item(
    provider: Provider, message: EmailMessage, *, body_chars: int = 0
) -> dict[str, JsonValue]:
    """One email for ``ToolResult.data``; the body only when ``body_chars`` > 0."""
    item: dict[str, JsonValue] = {
        "id": make_reference(provider.provider_name, message.id),
        "provider": provider.provider_name,
        "from": message.sender,
        "subject": message.subject,
        "snippet": message.snippet,
        "received_at": message.received_at.isoformat(timespec="minutes"),
        "unread": message.unread,
    }
    if body_chars > 0 and message.body:
        item["body"] = trim(message.body.strip(), body_chars)
    return item


def email_line(provider: Provider, message: EmailMessage) -> str:
    """One email as a line of ``detail`` text for the LLM."""
    reference = make_reference(provider.provider_name, message.id)
    when = message.received_at.isoformat(timespec="minutes")
    return f"- [{reference}] {when} from {message.sender}: {message.subject} — {message.snippet}"


def email_block(provider: Provider, message: EmailMessage, body_chars: int) -> str:
    """One email with headers and a trimmed body, as ``detail`` text for the LLM."""
    reference = make_reference(provider.provider_name, message.id)
    text = trim((message.body or message.snippet).strip(), body_chars)
    return (
        f"[{reference}] {message.received_at:%Y-%m-%d %H:%M}\n"
        f"From: {message.sender}\nSubject: {message.subject}\n{text}"
    )


def event_item(provider: Provider, event: CalendarEvent) -> dict[str, JsonValue]:
    """One calendar event for ``ToolResult.data``."""
    return {
        "id": make_reference(provider.provider_name, event.id),
        "provider": provider.provider_name,
        "title": event.title,
        "start": event.start.isoformat(timespec="minutes"),
        "end": event.end.isoformat(timespec="minutes"),
        "location": event.location,
        "attendees": list(event.attendees),
        "all_day": event.all_day,
    }


def event_line(event: CalendarEvent, now: datetime) -> str:
    """One event as a line of ``detail`` text, in the user's zone."""
    start = event.start.astimezone(now.tzinfo)
    end = event.end.astimezone(now.tzinfo)
    span = f"{start:%a %d %b} all day" if event.all_day else f"{start:%a %d %b %H:%M}-{end:%H:%M}"
    where = f" @ {event.location}" if event.location else ""
    with_whom = f" with {', '.join(event.attendees)}" if event.attendees else ""
    return f"- {span}: {event.title}{where}{with_whom}"


def chat_item(provider: Provider, message: ChatMessage) -> dict[str, JsonValue]:
    """One chat message for ``ToolResult.data``."""
    return {
        "id": make_reference(provider.provider_name, message.id),
        "provider": provider.provider_name,
        "channel_id": message.channel_id,
        "channel": message.channel_name,
        "author": message.author,
        "text": message.text,
        "sent_at": message.sent_at.isoformat(timespec="minutes"),
        "mentions_me": message.mentions_me,
        "unread": message.unread,
    }


def track_item(provider: Provider, track: Track) -> dict[str, JsonValue]:
    """One track for ``ToolResult.data``."""
    return {
        "provider": provider.provider_name,
        "title": track.title,
        "artist": track.artist,
        "album": track.album,
        "uri": track.uri,
        "duration_ms": track.duration_ms,
    }


def device_item(provider: Provider, device: Device) -> dict[str, JsonValue]:
    """One smart-home device for ``ToolResult.data``."""
    return {
        "id": device.id,
        "provider": provider.provider_name,
        "name": device.name,
        "kind": device.kind,
        "state": device.state,
        "brightness": device.brightness,
    }


# Design notes:
# - A private module inside tools/ rather than more API on the integration
#   layer: everything here produces ToolResults or LLM-facing text, which is
#   tool-layer vocabulary. The registry skips "_" modules, and plugins still
#   never import one another.
# - A missing IntegrationServices is treated exactly like an empty one, so
#   each tool has a single "no provider" branch instead of two.
# - References ("gmail:18c2") are minted here and resolved here, so the
#   round trip listing -> LLM -> follow-up tool is defined in one place.
