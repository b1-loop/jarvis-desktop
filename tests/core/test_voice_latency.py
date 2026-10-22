from jarvis.core.events import EventBus, LatencySample
from jarvis.core.voice_latency import TRANSCRIPT_TO_LLM_START, FirstTokenProbe, TurnTimeline
from jarvis.llm.base import UserMessage
from tests.agent.llm_fakes import ScriptedLLM, text_round
from tests.fakes import FakeClock


async def test_first_event_after_a_transcript_is_measured_once() -> None:
    bus, clock = EventBus(), FakeClock()
    samples: list[LatencySample] = []
    bus.subscribe_callback(LatencySample, samples.append)
    timeline = TurnTimeline(bus, clock=clock)
    inner = ScriptedLLM(rounds=[text_round("Hello ", "there."), text_round("Again.")])
    probe = FirstTokenProbe(inner, timeline)

    timeline.transcript_ready()
    clock.advance(0.25)
    first = [event async for event in probe.stream([UserMessage("hi")])]
    second = [event async for event in probe.stream([UserMessage("more")])]

    assert len(first) == 3 and len(second) == 2
    assert [(s.stage, round(s.ms)) for s in samples] == [(TRANSCRIPT_TO_LLM_START, 250)]
    assert probe.model == "scripted"
    assert probe.inner is inner
    assert inner.closed == 2


async def test_closing_the_probe_closes_the_provider_stream() -> None:
    inner = ScriptedLLM(rounds=[text_round("One. ", "Two.")])
    probe = FirstTokenProbe(inner, TurnTimeline(EventBus()))

    stream = probe.stream([UserMessage("hi")])
    await anext(stream)
    await stream.aclose()

    assert inner.closed == 1
