import asyncio

import pytest

from jarvis.core.events import (
    AssistantSaid,
    EventBus,
    ToolCallFinished,
    TranscriptReady,
    WakeWordHeard,
)
from jarvis.core.overlay_bridge import OverlayBridge
from jarvis.core.state_machine import AssistantState, StateMachine
from jarvis.ui.hud_protocol import (
    CloseRequested,
    HudProtocolError,
    Minimized,
    OpenLog,
    PlacementChanged,
    QuitRequested,
    RestartRequested,
    Restored,
    SetExpanded,
    SettingChanged,
    SettingsOpened,
    SubmitText,
    ToggleMute,
)
from jarvis.ui.overlay import Overlay, StubOverlay
from jarvis.utils.errors import JarvisError
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


class RecordingHandlers:
    """``HudHandlers`` that records every call in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.fail_next = False

    def submit(self, text: str) -> None:
        self.calls.append(("submit", text))

    def close(self) -> None:
        self.calls.append(("close",))

    def quit(self) -> None:
        self.calls.append(("quit",))

    async def toggle_mute(self) -> None:
        if self.fail_next:
            self.fail_next = False
            raise JarvisError("microphone would not reopen")
        self.calls.append(("toggle_mute",))

    def expanded(self, expanded: bool) -> None:
        self.calls.append(("expanded", expanded))

    async def setting_changed(self, key: str, value: str) -> None:
        self.calls.append(("setting", key, value))

    async def settings_opened(self) -> None:
        self.calls.append(("settings_opened",))

    def open_log(self) -> None:
        self.calls.append(("open_log",))

    def restart(self) -> None:
        self.calls.append(("restart",))

    async def placement_changed(self, event: PlacementChanged) -> None:
        self.calls.append(("placement", event.expanded, event.custom, event.monitor, event.orb))

    async def minimized(self, automatic: bool) -> None:
        self.calls.append(("minimized", automatic))

    async def restored(self, automatic: bool) -> None:
        self.calls.append(("restored", automatic))


def hud_bridge() -> tuple[OverlayBridge, StubOverlay, Overlay, RecordingHandlers]:
    native = StubOverlay()
    overlay = Overlay(native)
    bridge = OverlayBridge(overlay, EventBus(), input_level=lambda: 0.0, output_level=lambda: 0.0)
    handlers = RecordingHandlers()
    bridge.attach_handlers(handlers)
    return bridge, native, overlay, handlers


async def test_every_hud_event_reaches_its_handler_in_order() -> None:
    bridge, _native, overlay, handlers = hud_bridge()
    for event in (
        SubmitText("hej"),
        CloseRequested(),
        QuitRequested(),
        ToggleMute(),
        SetExpanded(expanded=True),
        SettingChanged("overlay_theme", "amber"),
        SettingsOpened(),
        OpenLog(),
        RestartRequested(),
    ):
        overlay.inject_ui_event(event)

    assert await bridge.drain_ui_events() == 9
    assert handlers.calls == [
        ("submit", "hej"),
        ("close",),
        ("quit",),
        ("toggle_mute",),
        ("expanded", True),
        ("setting", "overlay_theme", "amber"),
        ("settings_opened",),
        ("open_log",),
        ("restart",),
    ]
    assert await bridge.drain_ui_events() == 0


async def test_placement_events_are_parsed_and_dispatched() -> None:
    bridge, native, _overlay, handlers = hud_bridge()
    native.inject_ui_event(
        {
            "kind": "placement_changed",
            "expanded": False,
            "custom": True,
            "monitor": "DISPLAY1",
            "x": 10.0,
            "y": 20.0,
            "width": 0.0,
            "height": 0.0,
        }
    )

    await bridge.drain_ui_events()

    assert handlers.calls == [("placement", False, True, "DISPLAY1", False)]


async def test_minimize_and_restore_events_reach_their_handlers() -> None:
    bridge, native, overlay, handlers = hud_bridge()
    overlay.inject_ui_event(Minimized(automatic=False))
    overlay.inject_ui_event(Restored(automatic=True))
    # Exactly the dicts the native queue produces, orb placement included.
    native.inject_ui_event({"kind": "minimize", "automatic": True})
    native.inject_ui_event(
        {
            "kind": "placement_changed",
            "expanded": False,
            "custom": True,
            "monitor": "D2",
            "x": 12.0,
            "y": 900.0,
            "width": 48.0,
            "height": 48.0,
            "orb": True,
        }
    )
    native.inject_ui_event({"kind": "restore", "automatic": False})

    assert await bridge.drain_ui_events() == 5
    assert handlers.calls == [
        ("minimized", False),
        ("restored", True),
        ("minimized", True),
        ("placement", False, True, "D2", True),
        ("restored", False),
    ]


async def test_a_failing_handler_does_not_stop_the_next_event() -> None:
    bridge, _native, overlay, handlers = hud_bridge()
    handlers.fail_next = True
    overlay.inject_ui_event(ToggleMute())
    overlay.inject_ui_event(CloseRequested())

    await bridge.drain_ui_events()

    assert handlers.calls == [("close",)]


async def test_events_without_handlers_are_logged_not_lost_silently() -> None:
    native = StubOverlay()
    overlay = Overlay(native)
    bridge = OverlayBridge(overlay, EventBus(), input_level=lambda: 0.0, output_level=lambda: 0.0)
    overlay.inject_ui_event(CloseRequested())

    assert await bridge.drain_ui_events() == 1


async def test_the_pump_drains_events_on_the_loop() -> None:
    native = StubOverlay()
    overlay = Overlay(native)
    bridge = OverlayBridge(
        overlay, EventBus(), input_level=lambda: 0.0, output_level=lambda: 0.0, ui_interval=0.001
    )
    handlers = RecordingHandlers()
    bridge.attach_handlers(handlers)
    bridge.start()
    overlay.inject_ui_event(SubmitText("from the pump"))

    await eventually(lambda: handlers.calls == [("submit", "from the pump")])
    await bridge.aclose()


async def test_spoken_lines_tools_and_replies_build_the_history() -> None:
    bus, native = EventBus(), StubOverlay()
    bridge = OverlayBridge(Overlay(native), bus, input_level=lambda: 0.0, output_level=lambda: 0.0)
    bridge.start()

    bus.publish(TranscriptReady(text="what's on today", latency_ms=90.0))
    bus.publish(
        ToolCallFinished(name="calendar_agenda", args_summary="{}", duration_ms=120.4, ok=True)
    )
    bus.publish(
        ToolCallFinished(
            name="send_email", args_summary="{}", duration_ms=5.0, ok=False, error="no account"
        )
    )
    bus.publish(AssistantSaid(text="Two meetings."))
    await bridge.aclose()

    assert [(role, text) for role, text, _time in native.history] == [
        ("user", "what's on today"),
        ("tool", "calendar_agenda · 120 ms"),
        ("tool", "send_email failed · no account"),
        ("jarvis", "Two meetings."),
    ]


async def test_an_unknown_native_event_is_loud() -> None:
    bridge, native, _overlay, _handlers = hud_bridge()
    native.inject_ui_event({"kind": "teleport"})

    with pytest.raises(HudProtocolError):
        await bridge.drain_ui_events()
