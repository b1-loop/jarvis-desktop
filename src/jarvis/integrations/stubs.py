"""In-memory stand-ins for every integration interface.

These are **stubs**, not fakes of a real service: they hold a few realistic
fixtures, record every call in ``calls`` and never contact the outside
world. ``send`` on the email stub returns an id and nothing leaves the
machine. The factory selects them whenever a service's credentials are
absent, and tests use them to exercise tools and the agent without network.

Every class says "Stub" in its name and in its status (``stub=True``) so the
dashboard, the logs and a reviewer can never mistake one for a live provider.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

from jarvis.integrations.base import (
    CalendarEvent,
    CalendarProvider,
    ChatMessage,
    ChatProvider,
    Device,
    EmailDraft,
    EmailMessage,
    EmailProvider,
    MusicProvider,
    NowPlaying,
    ProviderError,
    ProviderStatus,
    SmartHomeProvider,
    TimeRange,
    Track,
)

_FIXTURE_TIME = datetime(2026, 10, 21, 8, 30, tzinfo=UTC)


class _CallRecorder:
    """Records ``(method, kwargs)`` so tests can assert what a tool asked for."""

    provider_name: ClassVar[str] = "stub"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, method: str, **kwargs: Any) -> None:
        self.calls.append((method, kwargs))

    async def status(self) -> ProviderStatus:
        """A stub is never configured or authenticated; it only exists."""
        return ProviderStatus(
            name=self.provider_name, configured=False, authenticated=False, stub=True
        )


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


def _sample_emails() -> list[EmailMessage]:
    return [
        EmailMessage(
            id="stub-mail-1",
            thread_id="stub-thread-1",
            sender="Maya Lindqvist <maya@example.com>",
            recipients=("you@example.com",),
            subject="Design review moved to Thursday",
            snippet="Can we move the design review to Thursday 14:00? The venue is booked.",
            received_at=_FIXTURE_TIME - timedelta(minutes=12),
            body=(
                "Hi,\n\nCan we move the design review to Thursday 14:00? "
                "The venue is booked on Wednesday.\n\nMaya"
            ),
        ),
        EmailMessage(
            id="stub-mail-2",
            thread_id="stub-thread-2",
            sender="GitHub <noreply@github.com>",
            recipients=("you@example.com",),
            subject="[jarvis-desktop] CI failed on feat/audio-engine",
            snippet="The native job failed: clang-tidy reported 2 warnings.",
            received_at=_FIXTURE_TIME - timedelta(hours=2),
            body="The native job failed: clang-tidy reported 2 warnings in playback.cpp.",
        ),
        EmailMessage(
            id="stub-mail-3",
            thread_id="stub-thread-3",
            sender="Newsletter <news@example.org>",
            recipients=("you@example.com",),
            subject="This week in systems programming",
            snippet="Lock-free queues, io_uring and a new C++26 paper.",
            received_at=_FIXTURE_TIME - timedelta(days=1),
            unread=False,
            body="Lock-free queues, io_uring and a new C++26 paper. Read on...",
        ),
    ]


class StubEmailProvider(_CallRecorder, EmailProvider):
    """Stub email provider with three fixture messages. Sends nothing."""

    provider_name = "stub-email"

    def __init__(self, messages: list[EmailMessage] | None = None) -> None:
        super().__init__()
        self.messages = messages if messages is not None else _sample_emails()
        self.drafts: dict[str, EmailDraft] = {}
        self.sent: list[EmailDraft] = []

    async def list_unread(self, limit: int = 10) -> list[EmailMessage]:
        """Unread fixtures, newest first."""
        self._record("list_unread", limit=limit)
        unread = [m for m in self.messages if m.unread]
        return sorted(unread, key=lambda m: m.received_at, reverse=True)[:limit]

    async def search(self, query: str, limit: int = 10) -> list[EmailMessage]:
        """Case-insensitive substring match on subject, sender and body."""
        self._record("search", query=query, limit=limit)
        needle = query.lower()
        hits = [
            m
            for m in self.messages
            if needle in m.subject.lower() or needle in m.sender.lower() or needle in m.body.lower()
        ]
        return hits[:limit]

    async def get(self, message_id: str) -> EmailMessage:
        """Fixture with that id, or ``ProviderError``."""
        self._record("get", message_id=message_id)
        for message in self.messages:
            if message.id == message_id:
                return message
        raise ProviderError(self.provider_name, f"no message with id {message_id!r}")

    async def draft_reply(self, message: EmailMessage, body: str) -> EmailDraft:
        """Keep the draft in memory under a predictable id."""
        self._record("draft_reply", message_id=message.id, body=body)
        draft = EmailDraft(
            id=f"stub-draft-{len(self.drafts) + 1}",
            to=(message.sender,),
            subject=_reply_subject(message.subject),
            body=body,
            in_reply_to=message.id,
            thread_id=message.thread_id,
        )
        self.drafts[draft.id] = draft
        return draft

    async def send(self, draft: EmailDraft) -> str:
        """Record the send; nothing leaves the process."""
        self._record("send", draft_id=draft.id)
        self.sent.append(draft)
        self.drafts.pop(draft.id, None)
        return f"stub-sent-{len(self.sent)}"


def _reply_subject(subject: str) -> str:
    return subject if subject.lower().startswith("re:") else f"Re: {subject}"


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def _sample_events(now: datetime) -> list[CalendarEvent]:
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return [
        CalendarEvent(
            id="stub-event-1",
            title="Sprint planning",
            start=day + timedelta(hours=10),
            end=day + timedelta(hours=11),
            location="Room 3B",
            attendees=("maya@example.com", "tom@example.com"),
        ),
        CalendarEvent(
            id="stub-event-2",
            title="1:1 with Maya",
            start=day + timedelta(hours=14),
            end=day + timedelta(hours=14, minutes=30),
        ),
        CalendarEvent(
            id="stub-event-3",
            title="Dentist",
            start=day + timedelta(days=1, hours=9),
            end=day + timedelta(days=1, hours=9, minutes=45),
            location="Dental clinic",
        ),
    ]


class StubCalendarProvider(_CallRecorder, CalendarProvider):
    """Stub calendar with three fixture events around ``now``."""

    provider_name = "stub-calendar"

    def __init__(
        self, events: list[CalendarEvent] | None = None, *, now: datetime | None = None
    ) -> None:
        super().__init__()
        self.events = events if events is not None else _sample_events(now or datetime.now(UTC))

    async def agenda(self, window: TimeRange) -> list[CalendarEvent]:
        """Fixture events overlapping ``window``, by start time."""
        self._record("agenda", window=window)
        hits = [e for e in self.events if TimeRange(e.start, e.end).overlaps(window)]
        return sorted(hits, key=lambda e: e.start)

    async def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        *,
        description: str = "",
        location: str = "",
        attendees: Any = (),
    ) -> CalendarEvent:
        """Add the event to the in-memory list."""
        self._record("create_event", title=title, start=start, end=end)
        event = CalendarEvent(
            id=f"stub-event-{len(self.events) + 1}",
            title=title,
            start=start,
            end=end,
            description=description,
            location=location,
            attendees=tuple(attendees),
        )
        self.events.append(event)
        return event


# ---------------------------------------------------------------------------
# Music
# ---------------------------------------------------------------------------


def _sample_tracks() -> list[Track]:
    return [
        Track(
            "stub-track-1", "Clair de Lune", "Claude Debussy", "Suite bergamasque", "stub:track:1"
        ),
        Track("stub-track-2", "Blue in Green", "Miles Davis", "Kind of Blue", "stub:track:2"),
        Track("stub-track-3", "Weightless", "Marconi Union", "Weightless", "stub:track:3"),
    ]


class StubMusicProvider(_CallRecorder, MusicProvider):
    """Stub music provider over a three-track catalogue."""

    provider_name = "stub-music"

    def __init__(self, catalogue: list[Track] | None = None) -> None:
        super().__init__()
        self.catalogue = catalogue if catalogue is not None else _sample_tracks()
        self.current: NowPlaying | None = None
        self.queued: list[Track] = []

    async def now_playing(self) -> NowPlaying | None:
        """Current fixture state."""
        self._record("now_playing")
        return self.current

    async def play(self, query_or_uri: str) -> Track:
        """Start the matching track."""
        self._record("play", query_or_uri=query_or_uri)
        track = self._find(query_or_uri)
        self.current = NowPlaying(track=track, is_playing=True)
        return track

    async def pause(self) -> None:
        """Mark playback paused."""
        self._record("pause")
        if self.current is not None:
            self.current = replace(self.current, is_playing=False)

    async def resume(self) -> None:
        """Mark playback resumed."""
        self._record("resume")
        if self.current is not None:
            self.current = replace(self.current, is_playing=True)

    async def next_track(self) -> None:
        """Pop the queue, or stop when it is empty."""
        self._record("next_track")
        self.current = NowPlaying(self.queued.pop(0), is_playing=True) if self.queued else None

    async def queue(self, query: str) -> Track:
        """Append the best match to the queue."""
        self._record("queue", query=query)
        track = self._find(query)
        self.queued.append(track)
        return track

    async def search(self, query: str, limit: int = 5) -> list[Track]:
        """Substring match on title and artist."""
        self._record("search", query=query, limit=limit)
        needle = query.lower()
        return [
            t for t in self.catalogue if needle in t.title.lower() or needle in t.artist.lower()
        ][:limit]

    def _find(self, query_or_uri: str) -> Track:
        for track in self.catalogue:
            if track.uri == query_or_uri:
                return track
        hits = [
            t
            for t in self.catalogue
            if query_or_uri.lower() in t.title.lower() or query_or_uri.lower() in t.artist.lower()
        ]
        if not hits:
            raise ProviderError(self.provider_name, f"nothing matches {query_or_uri!r}")
        return hits[0]


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def _sample_chat() -> list[ChatMessage]:
    return [
        ChatMessage(
            id="stub-msg-1",
            channel_id="C-general",
            channel_name="general",
            author="Tom",
            text="Lunch at 12:30? Pizza place around the corner.",
            sent_at=_FIXTURE_TIME - timedelta(minutes=5),
            unread=True,
        ),
        ChatMessage(
            id="stub-msg-2",
            channel_id="C-dev",
            channel_name="dev",
            author="Maya",
            text="@you can you review the ring buffer PR before standup?",
            sent_at=_FIXTURE_TIME - timedelta(minutes=40),
            mentions_me=True,
            unread=True,
        ),
        ChatMessage(
            id="stub-msg-3",
            channel_id="C-dev",
            channel_name="dev",
            author="CI bot",
            text="Build #412 passed on main.",
            sent_at=_FIXTURE_TIME - timedelta(hours=3),
        ),
    ]


class StubChatProvider(_CallRecorder, ChatProvider):
    """Stub chat provider with two channels and three fixture messages."""

    provider_name = "stub-chat"

    def __init__(self, messages: list[ChatMessage] | None = None) -> None:
        super().__init__()
        self.messages = messages if messages is not None else _sample_chat()

    async def list_recent(
        self,
        channel: str | None = None,
        limit: int = 20,
        *,
        unread_only: bool = False,
        mentions_only: bool = False,
    ) -> list[ChatMessage]:
        """Filter fixtures, newest first."""
        self._record(
            "list_recent",
            channel=channel,
            limit=limit,
            unread_only=unread_only,
            mentions_only=mentions_only,
        )
        hits = [
            m
            for m in self.messages
            if (channel is None or channel in (m.channel_id, m.channel_name))
            and (not unread_only or m.unread)
            and (not mentions_only or m.mentions_me)
        ]
        return sorted(hits, key=lambda m: m.sent_at, reverse=True)[:limit]

    async def send(self, channel: str, text: str) -> ChatMessage:
        """Append to the in-memory channel; nothing is posted anywhere."""
        self._record("send", channel=channel, text=text)
        message = ChatMessage(
            id=f"stub-msg-{len(self.messages) + 1}",
            channel_id=channel,
            channel_name=channel,
            author="you",
            text=text,
            sent_at=datetime.now(UTC),
        )
        self.messages.append(message)
        return message


# ---------------------------------------------------------------------------
# Smart home
# ---------------------------------------------------------------------------

_FULL_BRIGHTNESS = 100


def _sample_devices() -> dict[str, Device]:
    devices = [
        Device("light.living_room", "Living room", "light", "on", brightness=80),
        Device("light.bedroom", "Bedroom", "light", "off", brightness=0),
        Device("switch.desk_lamp", "Desk lamp", "switch", "on"),
        Device("scene.movie_night", "Movie night", "scene", "scening"),
    ]
    return {device.id: device for device in devices}


class StubSmartHomeProvider(_CallRecorder, SmartHomeProvider):
    """Stub smart home with two lights, a switch and a scene."""

    provider_name = "stub-smart-home"

    def __init__(self, devices: dict[str, Device] | None = None) -> None:
        super().__init__()
        self.devices = devices if devices is not None else _sample_devices()

    async def list_devices(self, kind: str | None = None) -> list[Device]:
        """All fixtures, optionally filtered by ``kind``."""
        self._record("list_devices", kind=kind)
        return [d for d in self.devices.values() if kind is None or d.kind == kind]

    async def state(self, device_id: str) -> Device:
        """Fixture with that id, or ``ProviderError``."""
        self._record("state", device_id=device_id)
        return self._get(device_id)

    async def turn_on(self, device_id: str) -> Device:
        """Set state to ``on`` (full brightness for a light that was dark)."""
        self._record("turn_on", device_id=device_id)
        device = self._get(device_id)
        brightness = device.brightness or _FULL_BRIGHTNESS if device.kind == "light" else None
        return self._update(replace(device, state="on", brightness=brightness))

    async def turn_off(self, device_id: str) -> Device:
        """Set state to ``off``."""
        self._record("turn_off", device_id=device_id)
        device = self._get(device_id)
        brightness = 0 if device.kind == "light" else None
        return self._update(replace(device, state="off", brightness=brightness))

    async def set_brightness(self, device_id: str, percent: int) -> Device:
        """Set brightness; zero turns the light off."""
        self._record("set_brightness", device_id=device_id, percent=percent)
        device = self._get(device_id)
        if device.kind != "light":
            raise ProviderError(self.provider_name, f"{device_id} is not dimmable")
        state = "on" if percent > 0 else "off"
        return self._update(replace(device, state=state, brightness=percent))

    async def set_scene(self, scene_id: str) -> None:
        """Record the activation; scenes have no observable state here."""
        self._record("set_scene", scene_id=scene_id)
        self._get(scene_id)

    def _get(self, device_id: str) -> Device:
        try:
            return self.devices[device_id]
        except KeyError:
            raise ProviderError(self.provider_name, f"unknown device {device_id!r}") from None

    def _update(self, device: Device) -> Device:
        self.devices[device.id] = device
        return device


# Design notes:
# - Fixtures are anchored at a fixed instant (email, chat) or at a caller-
#   supplied ``now`` (calendar), so tests are deterministic and the "good
#   morning" routine still sees something sensible when run live without keys.
# - Mutating stubs (sent drafts, queued tracks, device state) keep their state
#   in plain attributes rather than behind accessors; tests read them directly.
# - The recorder stores kwargs, not positional args, so an assertion reads as
#   ``("send", {"channel": "C-dev", "text": "..."})`` and survives reordering.
