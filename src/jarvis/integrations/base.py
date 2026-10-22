"""Provider-neutral models, interfaces and errors for external services.

Tools and the agent only ever see the types in this module. A tool that
summarises unread mail works the same against Gmail, Outlook or the in-memory
stub because each returns :class:`EmailMessage`, and the composition root
decides which provider is wired in (see ``factory.py``). Keeping the models
free of SDK types also means the LLM-facing schemas never leak a vendor's
field names.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from typing import ClassVar

from jarvis.utils.errors import JarvisError

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class IntegrationError(JarvisError):
    """Base class for anything that goes wrong talking to an external service.

    Attributes:
        provider: Short provider name (``"gmail"``, ``"spotify"`` ...) so the
            agent can tell the user *which* service failed.
    """

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"{provider}: {message}")


class NotConfiguredError(IntegrationError):
    """The provider cannot be built because settings are missing.

    Attributes:
        missing: The exact environment variable names to set.
    """

    def __init__(self, provider: str, missing: Sequence[str]) -> None:
        self.missing = tuple(missing)
        names = ", ".join(self.missing)
        super().__init__(provider, f"not configured; set {names} in .env (see .env.example)")


class AuthError(IntegrationError):
    """Credentials are missing, expired, revoked or lack a required scope."""


class ConsentRequiredError(AuthError):
    """Signing in needs the user, but this session cannot ask them.

    Raised by a non-interactive token source (the windowless app, voice
    mode) instead of opening a browser or printing a device code nobody
    would see. The message is the remedy, phrased to be read aloud.

    Attributes:
        reason: Why consent is needed (no token yet, refresh rejected), for logs.
        remedy: What the user should do, e.g. the connect command to run.
    """

    def __init__(self, provider: str, *, reason: str, remedy: str) -> None:
        self.reason = reason
        self.remedy = remedy
        super().__init__(provider, remedy)


class RateLimitedError(IntegrationError):
    """The service asked us to slow down.

    Attributes:
        retry_after: Seconds to wait, when the service said so.
    """

    def __init__(self, provider: str, message: str, *, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(provider, message)


class ProviderError(IntegrationError):
    """Any other failure: transport error, 5xx, malformed response.

    Attributes:
        status_code: HTTP status when the failure was an HTTP response.
    """

    def __init__(self, provider: str, message: str, *, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(provider, message)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    """What the dashboard shows for one provider.

    Attributes:
        name: Provider name as reported by :attr:`Provider.provider_name`.
        configured: Settings for a real provider are present. Always False for
            a stub, which exists precisely because nothing is configured.
        authenticated: A credential has been used successfully in this process.
        last_error: Message of the most recent failure, or ``None``.
        stub: True when the provider is an in-memory stand-in.
    """

    name: str
    configured: bool
    authenticated: bool
    last_error: str | None = None
    stub: bool = False


class Provider(ABC):
    """Common surface of every provider, real or stub."""

    provider_name: ClassVar[str] = "provider"

    @abstractmethod
    async def status(self) -> ProviderStatus:
        """Report configuration and health for the dashboard. Never raises."""


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmailMessage:
    """One email, as much of it as the listing call returned.

    Attributes:
        id: Provider message id.
        thread_id: Provider conversation id, used to thread replies.
        sender: ``Name <address>`` or bare address.
        recipients: Addresses in the To header.
        subject: Subject line.
        snippet: First sentence or two, for listings.
        received_at: Timezone-aware receipt time.
        unread: Whether the message is still unread.
        body: Full plain-text body; empty until :meth:`EmailProvider.get`.
        rfc822_id: The ``Message-ID`` header, when the provider exposes it;
            used for ``In-Reply-To`` so replies thread in every mail client.
    """

    id: str
    thread_id: str
    sender: str
    recipients: tuple[str, ...]
    subject: str
    snippet: str
    received_at: datetime
    unread: bool = True
    body: str = ""
    rfc822_id: str | None = None


@dataclass(frozen=True, slots=True)
class EmailDraft:
    """A reply prepared by :meth:`EmailProvider.draft_reply`, not yet sent.

    Attributes:
        id: Provider draft id; ``send`` needs it.
        to: Recipients.
        subject: Subject, already prefixed with ``Re:`` where appropriate.
        body: Plain-text body.
        in_reply_to: Id of the message being answered.
        thread_id: Conversation the draft belongs to.
    """

    id: str
    to: tuple[str, ...]
    subject: str
    body: str
    in_reply_to: str | None = None
    thread_id: str | None = None


class EmailProvider(Provider):
    """Read, search and reply to email.

    Sending is split into ``draft_reply`` and ``send`` so the agent can read
    the draft back to the user and wait for confirmation in between.
    """

    @abstractmethod
    async def list_unread(self, limit: int = 10) -> list[EmailMessage]:
        """Newest unread messages first, at most ``limit``."""

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> list[EmailMessage]:
        """Messages matching ``query`` in the provider's own search syntax."""

    @abstractmethod
    async def get(self, message_id: str) -> EmailMessage:
        """One message with its full body.

        Raises:
            ProviderError: If no message has that id.
        """

    @abstractmethod
    async def draft_reply(self, message: EmailMessage, body: str) -> EmailDraft:
        """Create a reply to ``message`` on the server without sending it."""

    @abstractmethod
    async def send(self, draft: EmailDraft) -> str:
        """Send a draft and return the id of the sent message."""


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimeRange:
    """Half-open interval ``[start, end)`` of timezone-aware datetimes."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"end must be after start, got {self.start} .. {self.end}")

    @property
    def duration(self) -> timedelta:
        """Length of the range."""
        return self.end - self.start

    def overlaps(self, other: TimeRange) -> bool:
        """True if the two ranges share any instant."""
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True, slots=True)
class WorkingHours:
    """Daily window inside which free slots are offered.

    Attributes:
        start: Local start of the working day.
        end: Local end of the working day.
        weekdays_only: Skip Saturday and Sunday.
    """

    start: time = time(9)
    end: time = time(17)
    weekdays_only: bool = True


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    """One calendar entry.

    Attributes:
        id: Provider event id.
        title: Summary line.
        start: Timezone-aware start; midnight for all-day events.
        end: Timezone-aware end (exclusive).
        location: Free-text location, may be empty.
        description: Notes, may be empty.
        attendees: Email addresses of attendees.
        all_day: True for date-only events, which do not block free slots.
    """

    id: str
    title: str
    start: datetime
    end: datetime
    location: str = ""
    description: str = ""
    attendees: tuple[str, ...] = ()
    all_day: bool = False


_SATURDAY = 5


class CalendarProvider(Provider):
    """Agenda, availability and event creation.

    Only ``agenda`` and ``create_event`` touch the service; ``today``,
    ``upcoming`` and ``find_free_slots`` are derived here so every provider
    (and the stub) shares one tested implementation of the date arithmetic.
    """

    @abstractmethod
    async def agenda(self, window: TimeRange) -> list[CalendarEvent]:
        """Events overlapping ``window``, ordered by start."""

    @abstractmethod
    async def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        *,
        description: str = "",
        location: str = "",
        attendees: Sequence[str] = (),
    ) -> CalendarEvent:
        """Create an event and return it as stored by the service."""

    async def today(self, now: datetime | None = None) -> list[CalendarEvent]:
        """Events on the calendar day containing ``now`` (default: current UTC time)."""
        moment = now or datetime.now(UTC)
        day_start = moment.replace(hour=0, minute=0, second=0, microsecond=0)
        return await self.agenda(TimeRange(day_start, day_start + timedelta(days=1)))

    async def upcoming(self, within: timedelta, now: datetime | None = None) -> list[CalendarEvent]:
        """Events starting in the next ``within``; feeds the meeting reminder."""
        moment = now or datetime.now(UTC)
        events = await self.agenda(TimeRange(moment, moment + within))
        return [event for event in events if event.start >= moment]

    async def find_free_slots(
        self,
        duration: timedelta,
        window: TimeRange,
        working_hours: WorkingHours | None = None,
    ) -> list[TimeRange]:
        """Gaps of at least ``duration`` inside working hours within ``window``.

        All-day events are ignored: a birthday should not make the whole day
        look busy.
        """
        hours = working_hours or WorkingHours()
        busy = [
            TimeRange(event.start, event.end)
            for event in await self.agenda(window)
            if not event.all_day
        ]
        slots: list[TimeRange] = []
        for day in _working_days(window, hours):
            slots.extend(_gaps(day, busy, duration))
        return slots


def _working_days(window: TimeRange, hours: WorkingHours) -> list[TimeRange]:
    """Working-hour ranges for each calendar day of ``window``, clipped to it."""
    days: list[TimeRange] = []
    cursor = window.start.date()
    while cursor <= window.end.date():
        if not (hours.weekdays_only and cursor.weekday() >= _SATURDAY):
            start = max(datetime.combine(cursor, hours.start, window.start.tzinfo), window.start)
            end = min(datetime.combine(cursor, hours.end, window.start.tzinfo), window.end)
            if start < end:
                days.append(TimeRange(start, end))
        cursor += timedelta(days=1)
    return days


def _gaps(day: TimeRange, busy: Sequence[TimeRange], duration: timedelta) -> list[TimeRange]:
    """Free ranges of at least ``duration`` in ``day`` once ``busy`` is removed."""
    gaps: list[TimeRange] = []
    cursor = day.start
    for block in sorted((b for b in busy if b.overlaps(day)), key=lambda b: b.start):
        if block.start - cursor >= duration:
            gaps.append(TimeRange(cursor, block.start))
        cursor = max(cursor, block.end)
    if day.end - cursor >= duration:
        gaps.append(TimeRange(cursor, day.end))
    return gaps


# ---------------------------------------------------------------------------
# Music
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Track:
    """One playable track.

    Attributes:
        id: Provider track id.
        title: Track name.
        artist: Primary artist(s), comma separated.
        album: Album name.
        uri: Provider URI usable with ``play`` and ``queue``.
        duration_ms: Length in milliseconds, 0 if unknown.
    """

    id: str
    title: str
    artist: str
    album: str
    uri: str
    duration_ms: int = 0


@dataclass(frozen=True, slots=True)
class NowPlaying:
    """Current playback position."""

    track: Track
    is_playing: bool
    progress_ms: int = 0


class MusicProvider(Provider):
    """Control playback on the user's active device."""

    @abstractmethod
    async def now_playing(self) -> NowPlaying | None:
        """What is playing, or ``None`` when nothing is."""

    @abstractmethod
    async def play(self, query_or_uri: str) -> Track | None:
        """Start playing a URI, or the best search hit for a free-text query.

        Returns:
            The track now playing, or ``None`` when a context such as a
            playlist was started and the service did not say which track.
        """

    @abstractmethod
    async def pause(self) -> None:
        """Pause playback."""

    @abstractmethod
    async def resume(self) -> None:
        """Resume paused playback."""

    @abstractmethod
    async def next_track(self) -> None:
        """Skip to the next track."""

    @abstractmethod
    async def queue(self, query: str) -> Track:
        """Append the best hit for ``query`` to the queue and return it."""

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[Track]:
        """Tracks matching ``query``, best first."""


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a channel or direct conversation.

    Attributes:
        id: Provider message id.
        channel_id: Id of the channel or DM the message is in.
        channel_name: Human-readable channel name, may equal the id.
        author: Display name of the sender.
        text: Plain text.
        sent_at: Timezone-aware send time.
        mentions_me: True if the message mentions the authenticated user/bot.
        unread: True if the provider knows the message is unread.
    """

    id: str
    channel_id: str
    channel_name: str
    author: str
    text: str
    sent_at: datetime
    mentions_me: bool = False
    unread: bool = False


class ChatProvider(Provider):
    """Read recent messages and post replies."""

    @abstractmethod
    async def list_recent(
        self,
        channel: str | None = None,
        limit: int = 20,
        *,
        unread_only: bool = False,
        mentions_only: bool = False,
    ) -> list[ChatMessage]:
        """Newest messages first.

        Args:
            channel: Channel or DM id; ``None`` means every reachable channel.
            limit: Maximum number of messages across all channels.
            unread_only: Keep only messages the provider marks unread.
            mentions_only: Keep only messages that mention the current user.
        """

    @abstractmethod
    async def send(self, channel: str, text: str) -> ChatMessage:
        """Post ``text`` to ``channel`` and return the stored message."""


# ---------------------------------------------------------------------------
# Smart home
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Device:
    """One controllable entity.

    Attributes:
        id: Provider entity id, e.g. ``light.kitchen``.
        name: Friendly name.
        kind: Domain such as ``light``, ``switch`` or ``scene``.
        state: Raw state string (``on``, ``off``, ``unavailable`` ...).
        brightness: Percent 0-100 for dimmable devices, else ``None``.
        attributes: Remaining provider attributes, for the dashboard.
    """

    id: str
    name: str
    kind: str
    state: str
    brightness: int | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)

    @property
    def is_on(self) -> bool:
        """True when the device reports an ``on`` state."""
        return self.state == "on"


class SmartHomeProvider(Provider):
    """Lights, switches and scenes."""

    @abstractmethod
    async def list_devices(self, kind: str | None = None) -> list[Device]:
        """All devices, optionally only those of one ``kind``."""

    @abstractmethod
    async def state(self, device_id: str) -> Device:
        """Current state of one device."""

    @abstractmethod
    async def turn_on(self, device_id: str) -> Device:
        """Switch a device on and return its new state."""

    @abstractmethod
    async def turn_off(self, device_id: str) -> Device:
        """Switch a device off and return its new state."""

    @abstractmethod
    async def set_brightness(self, device_id: str, percent: int) -> Device:
        """Set brightness 0-100 on a dimmable device and return its new state."""

    @abstractmethod
    async def set_scene(self, scene_id: str) -> None:
        """Activate a scene."""


# Design notes:
# - ABCs rather than Protocols: the calendar interface carries real shared
#   behaviour (today/upcoming/free slots derived from agenda), and OutlookProvider
#   implements two interfaces at once, which ABCs express directly.
# - Frozen, slotted dataclasses instead of pydantic models: these cross no
#   trust boundary (the provider builds them from a response it already
#   validated), so pydantic would add cost without safety. Tool input schemas,
#   which do cross a boundary, stay pydantic.
# - All datetimes are timezone-aware. Providers convert at the edge; nothing in
#   the agent has to guess a zone.
# - draft_reply and send are separate on purpose: the confirmation policy
#   (ADR-0006) needs a moment between "here is what I would send" and sending.
