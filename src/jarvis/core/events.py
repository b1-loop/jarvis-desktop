"""Typed event bus connecting pipeline stages.

Stages publish events (state transitions, transcripts, tool calls, latency
samples) and subscribers such as the overlay, tray icon, dashboard and logger
react without the publisher knowing they exist. The bus is deliberately
simple: in-process, single event loop, no persistence. Its one hard
guarantee is that a slow or dead subscriber can never stall the publisher,
because the pipeline's latency budget must not depend on the UI.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.core.state_machine import AssistantState

log = get_logger(__name__)

DEFAULT_QUEUE_SIZE: Final = 256
"""Per-subscriber queue depth; a UI that falls this far behind is not keeping up anyway."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Base class for everything published on the bus.

    Attributes:
        timestamp: Wall-clock seconds since the epoch at creation. Wall clock
            rather than monotonic so the dashboard can show real times; stage
            durations are measured separately with ``utils.timing``.
    """

    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class StateChanged(Event):
    """The assistant moved to a new state.

    Attributes:
        old: State before the transition.
        new: State after the transition.
        reason: Short, human-readable cause (``"wake word"``, ``"barge-in"``).
    """

    old: AssistantState
    new: AssistantState
    reason: str


@dataclass(frozen=True, slots=True)
class WakeWordHeard(Event):
    """The wake word (or its hotkey fallback) started a turn.

    Attributes:
        score: Detector confidence; ``1.0`` for the hotkey.
    """

    score: float = 1.0


@dataclass(frozen=True, slots=True)
class TranscriptReady(Event):
    """Speech-to-text finished for one utterance.

    Attributes:
        text: What the user said.
        latency_ms: Time from end of speech to transcript.
    """

    text: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class AssistantSaid(Event):
    """Jarvis produced a spoken (or printed, in text-only mode) answer."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStarted(Event):
    """A tool is about to run.

    Attributes:
        name: Tool name as registered.
        args_summary: Truncated, JSON-ish rendering of the validated input.
    """

    name: str
    args_summary: str


@dataclass(frozen=True, slots=True)
class ToolCallFinished(Event):
    """A tool returned (successfully or not).

    Attributes:
        name: Tool name as registered.
        args_summary: Same summary as the matching ``ToolCallStarted``.
        duration_ms: Wall time spent inside the tool.
        ok: Whether the tool reported success.
        error: Failure detail when ``ok`` is False.
    """

    name: str
    args_summary: str
    duration_ms: float
    ok: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ConfirmationRequested(Event):
    """The confirmation policy is asking the user before running a tool."""

    tool_name: str
    prompt: str


@dataclass(frozen=True, slots=True)
class ConfirmationResolved(Event):
    """The user answered a confirmation request."""

    tool_name: str
    approved: bool


@dataclass(frozen=True, slots=True)
class LatencySample(Event):
    """One timed pipeline stage, for the dashboard and the benchmark."""

    stage: str
    ms: float


class _Signal(Enum):
    """In-band marker telling an iterator that its subscription ended."""

    CLOSED = "closed"


class Subscription[E: Event]:
    """An async iterator over events of one type (or base class).

    Created by :meth:`EventBus.subscribe`. Iterate it with ``async for`` or
    use it as an async context manager so the subscription is removed from
    the bus when the consumer is done. Each subscription owns a bounded
    queue; when the consumer falls behind, the oldest queued event is
    dropped and counted rather than blocking the publisher.
    """

    def __init__(
        self,
        event_type: type[E],
        *,
        maxsize: int,
        on_close: Callable[[Subscription[Any]], None],
    ) -> None:
        self._event_type = event_type
        self._queue: asyncio.Queue[E | _Signal] = asyncio.Queue(maxsize=maxsize)
        self._on_close = on_close
        self._dropped = 0
        self._closed = False

    @property
    def event_type(self) -> type[E]:
        """The type (or base class) this subscription filters on."""
        return self._event_type

    @property
    def dropped(self) -> int:
        """Events discarded because the consumer fell behind."""
        return self._dropped

    @property
    def closed(self) -> bool:
        """True once ``close`` ran, from either side."""
        return self._closed

    def matches(self, event: Event) -> bool:
        """Whether ``event`` should be delivered here."""
        return isinstance(event, self._event_type)

    def deliver(self, event: E) -> None:
        """Enqueue ``event`` without ever blocking; drop the oldest if full."""
        if self._closed:
            return
        self._make_room()
        self._queue.put_nowait(event)

    def close(self) -> None:
        """Detach from the bus and end any pending ``async for``. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._on_close(self)
        self._make_room()
        self._queue.put_nowait(_Signal.CLOSED)

    def _make_room(self) -> None:
        if not self._queue.full():
            return
        self._queue.get_nowait()
        self._dropped += 1
        log.warning(
            "events.subscriber.overflow",
            event_type=self._event_type.__name__,
            dropped=self._dropped,
        )

    def __aiter__(self) -> Subscription[E]:
        return self

    async def __anext__(self) -> E:
        item = await self._queue.get()
        if isinstance(item, _Signal):
            # Leave the marker in place so every later __anext__ also stops.
            self._queue.put_nowait(item)
            raise StopAsyncIteration
        return item

    async def __aenter__(self) -> Subscription[E]:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.close()


Unsubscribe = Callable[[], None]
"""Returned by ``subscribe_callback``; call it to stop receiving events."""


class EventBus:
    """Publish/subscribe hub for one asyncio event loop.

    Not thread-safe by design: native threads never publish directly, they
    hand data to the Python side which publishes from the loop.
    """

    def __init__(self, *, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self._queue_size = queue_size
        self._subscriptions: list[Subscription[Any]] = []
        self._callbacks: list[tuple[type[Event], Callable[[Any], Awaitable[None] | None]]] = []
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        self._published = 0

    @property
    def closed(self) -> bool:
        """True after ``close``; publishing is then a logged no-op."""
        return self._closed

    @property
    def published(self) -> int:
        """Total events accepted since construction, for the dashboard."""
        return self._published

    def subscribe[E: Event](
        self, event_type: type[E], *, queue_size: int | None = None
    ) -> Subscription[E]:
        """Return an async iterator over every ``event_type`` published from now on.

        Args:
            event_type: Concrete event class, or ``Event`` for everything.
            queue_size: Override the bus-wide per-subscriber queue depth.
        """
        subscription = Subscription(
            event_type,
            maxsize=queue_size or self._queue_size,
            on_close=self._remove_subscription,
        )
        self._subscriptions.append(subscription)
        return subscription

    def subscribe_callback[E: Event](
        self, event_type: type[E], callback: Callable[[E], Awaitable[None] | None]
    ) -> Unsubscribe:
        """Call ``callback`` synchronously on publish for every ``event_type``.

        A coroutine-returning callback is scheduled as a task so the publisher
        still never waits. Exceptions raised by a callback are logged with a
        traceback and do not reach the publisher; one broken UI listener
        must not take the pipeline down with it.

        Returns:
            A function that removes the callback.
        """
        entry = (event_type, callback)
        self._callbacks.append(entry)

        def unsubscribe() -> None:
            if entry in self._callbacks:
                self._callbacks.remove(entry)

        return unsubscribe

    def publish(self, event: Event) -> None:
        """Deliver ``event`` to every matching subscriber without blocking.

        Synchronous on purpose: delivery is ``put_nowait`` into bounded queues,
        so there is nothing to await, and non-async code such as the state
        machine can publish directly.
        """
        if self._closed:
            log.debug("events.publish.after_close", event_type=type(event).__name__)
            return
        self._published += 1
        for subscription in list(self._subscriptions):
            if subscription.matches(event):
                subscription.deliver(event)
        for event_type, callback in list(self._callbacks):
            if isinstance(event, event_type):
                self._invoke(callback, event)

    async def close(self) -> None:
        """End every subscription and wait for in-flight callback tasks."""
        self._closed = True
        for subscription in list(self._subscriptions):
            subscription.close()
        self._callbacks.clear()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def _remove_subscription(self, subscription: Subscription[Any]) -> None:
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)

    def _invoke(self, callback: Callable[[Any], Awaitable[None] | None], event: Event) -> None:
        try:
            result = callback(event)
        except Exception:
            log.exception("events.callback.failed", event_type=type(event).__name__)
            return
        if inspect.isawaitable(result):
            task = asyncio.ensure_future(result)
            self._tasks.add(task)
            task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            log.error("events.callback.task_failed", error=repr(error))


# Design notes:
# - Drop-oldest, never block: the publisher is usually the pipeline's hot
#   path (state machine, STT, agent). Back-pressure from a UI subscriber
#   would add latency to the voice loop, which is the one thing the whole
#   project is built to minimise. Drops are counted and logged so they are
#   visible on the dashboard instead of silently lost.
# - Synchronous publish: there is no I/O in delivery, and making it a
#   coroutine would force every publisher to be async for no benefit.
# - Callback exceptions are logged and isolated rather than propagated. This
#   is not swallowing: the traceback is emitted at ERROR level. It is the
#   same decision an HTTP server makes for one failing handler.
# - Events are frozen slotted dataclasses so a subscriber cannot mutate what
#   another subscriber sees, and ``StateChanged`` refers to ``AssistantState``
#   under TYPE_CHECKING only, which keeps ``state_machine`` -> ``events`` a
#   one-way runtime import.
