import asyncio

from jarvis.core.events import EventBus, TranscriptReady, WakeWordHeard
from jarvis.core.overlay_bridge import OverlayBridge
from jarvis.core.state_machine import AssistantState, StateMachine
from jarvis.ui.overlay import Overlay, StubOverlay
from tests.core.voice_fakes import eventually


async def test_bridge_mirrors_state_text_and_levels() -> None:
    bus, native = EventBus(), StubOverlay()
    state = StateMachine(bus)
    bridge = OverlayBridge(
        Overlay(native), bus, input_level=lambda: 0.3, output_level=lambda: 0.7, interval=0.001
    )
    bridge.start()

    await asyncio.sleep(0.01)
    assert native.levels == []  # idle: flat waveform

    bus.publish(WakeWordHeard(score=0.9))
    state.transition(AssistantState.LISTENING, "wake word")
    await eventually(lambda: 0.3 in native.levels)
    bus.publish(TranscriptReady(text="What time is it?", latency_ms=120.0))
    state.transition(AssistantState.THINKING, "request received")
    state.transition(AssistantState.SPEAKING, "answer ready")
    bridge.show_response("Half past nine.")
    await eventually(lambda: 0.7 in native.levels)
    await bridge.aclose()
    await bridge.aclose()

    assert native.wake_flashes == 1
    assert native.state == "SPEAKING"
    assert native.transcript == "What time is it?"
    assert native.response == "Half past nine."
