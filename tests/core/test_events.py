import asyncio

import pytest

from jarvis.core.events import (
    AssistantSaid,
    Event,
    EventBus,
    LatencySample,
    TranscriptReady,
)

WAIT = 1.0


@pytest.fixture
def bus() -> EventBus:
    return EventBus(queue_size=4)


async def test_subscriber_receives_only_its_event_type(bus: EventBus) -> None:
    subscription = bus.subscribe(AssistantSaid)

    bus.publish(TranscriptReady(text="hello", latency_ms=12.0))
    bus.publish(AssistantSaid(text="hi"))

    event = await asyncio.wait_for(anext(subscription), WAIT)
    assert event.text == "hi"
    assert bus.published == 2


async def test_subscribing_to_the_base_class_receives_everything(bus: EventBus) -> None:
    subscription = bus.subscribe(Event)

    bus.publish(TranscriptReady(text="hello", latency_ms=12.0))
    bus.publish(LatencySample(stage="stt", ms=3.0))

    first = await asyncio.wait_for(anext(subscription), WAIT)
    second = await asyncio.wait_for(anext(subscription), WAIT)
    assert isinstance(first, TranscriptReady)
    assert isinstance(second, LatencySample)


async def test_events_carry_a_timestamp_and_are_immutable() -> None:
    event = AssistantSaid(text="hi")

    assert event.timestamp > 0
    with pytest.raises(AttributeError):
        event.text = "changed"  # type: ignore[misc]


async def test_overflow_drops_the_oldest_event_and_counts_it(bus: EventBus) -> None:
    subscription = bus.subscribe(LatencySample)

    for index in range(6):
        bus.publish(LatencySample(stage="s", ms=float(index)))

    received = [await asyncio.wait_for(anext(subscription), WAIT) for _ in range(4)]
    assert [event.ms for event in received] == [2.0, 3.0, 4.0, 5.0]
    assert subscription.dropped == 2


async def test_publish_never_blocks_even_with_a_dead_subscriber(bus: EventBus) -> None:
    subscription = bus.subscribe(LatencySample)  # never consumed

    # publish is synchronous: if it could block, this loop would hang the test.
    for index in range(100):
        bus.publish(LatencySample(stage="s", ms=float(index)))

    assert subscription.dropped == 96


async def test_closing_a_subscription_ends_iteration_and_detaches(bus: EventBus) -> None:
    subscription = bus.subscribe(AssistantSaid)
    subscription.close()

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(subscription), WAIT)
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(subscription), WAIT)
    bus.publish(AssistantSaid(text="late"))
    assert subscription.closed


async def test_subscription_is_an_async_context_manager(bus: EventBus) -> None:
    async with bus.subscribe(AssistantSaid) as subscription:
        bus.publish(AssistantSaid(text="hi"))
        event = await asyncio.wait_for(anext(subscription), WAIT)
        assert event.text == "hi"

    assert subscription.closed


async def test_async_for_ends_when_the_bus_closes(bus: EventBus) -> None:
    subscription = bus.subscribe(AssistantSaid)
    bus.publish(AssistantSaid(text="one"))

    async def consume() -> list[str]:
        return [event.text async for event in subscription]

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await bus.close()

    assert await asyncio.wait_for(consumer, WAIT) == ["one"]
    assert bus.closed


async def test_publish_after_close_is_a_no_op(bus: EventBus) -> None:
    await bus.close()

    bus.publish(AssistantSaid(text="late"))

    assert bus.published == 0


async def test_sync_callback_runs_on_publish_and_can_unsubscribe(bus: EventBus) -> None:
    seen: list[str] = []
    unsubscribe = bus.subscribe_callback(AssistantSaid, lambda event: seen.append(event.text))

    bus.publish(AssistantSaid(text="one"))
    unsubscribe()
    unsubscribe()  # idempotent
    bus.publish(AssistantSaid(text="two"))

    assert seen == ["one"]


async def test_async_callback_is_scheduled_without_blocking_publish(bus: EventBus) -> None:
    seen: list[str] = []
    started = asyncio.Event()

    async def listener(event: AssistantSaid) -> None:
        await started.wait()
        seen.append(event.text)

    bus.subscribe_callback(AssistantSaid, listener)
    bus.publish(AssistantSaid(text="one"))

    assert seen == []
    started.set()
    await bus.close()
    assert seen == ["one"]


async def test_failing_callbacks_are_logged_and_isolated(bus: EventBus) -> None:
    seen: list[str] = []

    def broken(event: AssistantSaid) -> None:
        raise RuntimeError("boom")

    async def broken_async(event: AssistantSaid) -> None:
        raise RuntimeError("async boom")

    bus.subscribe_callback(AssistantSaid, broken)
    bus.subscribe_callback(AssistantSaid, broken_async)
    bus.subscribe_callback(AssistantSaid, lambda event: seen.append(event.text))

    bus.publish(AssistantSaid(text="one"))
    await bus.close()

    assert seen == ["one"]


async def test_cancelled_callback_task_is_ignored_on_close(bus: EventBus) -> None:
    async def forever(event: AssistantSaid) -> None:
        await asyncio.sleep(WAIT * 10)

    bus.subscribe_callback(AssistantSaid, forever)
    bus.publish(AssistantSaid(text="one"))
    for task in list(bus._tasks):
        task.cancel()

    await bus.close()
