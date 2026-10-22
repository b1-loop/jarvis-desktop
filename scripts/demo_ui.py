"""Show the tray icon, dashboard and overlay driven by a scripted conversation.

No microphone, model or API key is involved: an ``EventBus`` and a real
``StateMachine`` replay a few typical turns (a calendar question, a shell
command that needs confirmation and fails, a declined e-mail, opening an
app) so a reviewer can see every UI surface react to the same events. The
memory tab reads a throwaway SQLite database seeded with a profile, facts,
notes, todos and reminders. The overlay appears only if ``jarvis_overlay``
is built; the tray and dashboard degrade to their stubs on a headless box.

Usage::

    uv run python scripts/demo_ui.py                    # 60 s, then exit
    uv run python scripts/demo_ui.py --duration 0       # until tray -> Quit
    uv run python scripts/demo_ui.py --screenshot docs/media/dashboard.png
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import itertools
import math
import sys
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from PIL import ImageGrab

from jarvis.core.events import (
    AssistantSaid,
    ConfirmationRequested,
    ConfirmationResolved,
    EventBus,
    LatencySample,
    StateChanged,
    ToolCallFinished,
    ToolCallStarted,
    TranscriptReady,
    WakeWordHeard,
)
from jarvis.core.state_machine import AssistantState, StateMachine
from jarvis.memory.sqlite_store import SqliteStore
from jarvis.ui.dashboard import DashboardConfig, DashboardSurface, create_dashboard
from jarvis.ui.overlay import Overlay, OverlayState, create_overlay
from jarvis.ui.tray import TraySurface, create_tray
from jarvis.utils.logging import configure_logging
from jarvis.utils.timing import LatencyTracker

DASHBOARD_TITLE: Final = "Jarvis dashboard (demo)"
STEP_S: Final = 0.6
LEVEL_INTERVAL_S: Final = 0.03
_PER_MONITOR_AWARE_V2: Final = -4
_SM_XVIRTUALSCREEN: Final = 76
_SM_YVIRTUALSCREEN: Final = 77
_DWMWA_EXTENDED_FRAME_BOUNDS: Final = 9


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--duration", type=float, default=60.0, help="seconds; 0 = until Quit")
    parser.add_argument("--screenshot", type=Path, help="save the dashboard window here")
    parser.add_argument("--no-overlay", action="store_true", help="skip the native overlay")
    return parser.parse_args(argv)


@dataclass
class DemoCallbacks:
    """Tray actions for the demo; a real app wires these to its services."""

    overlay: Overlay
    stop: asyncio.Event
    dashboard: DashboardSurface | None = None
    tray: TraySurface | None = None
    muted: bool = False

    def show_dashboard(self) -> None:
        """Raise the dashboard window."""
        if self.dashboard is not None:
            self.dashboard.show()

    def toggle_overlay(self) -> None:
        """Show or hide the overlay."""
        if self.overlay.visible:
            self.overlay.hide()
        else:
            self.overlay.show()

    def toggle_microphone(self) -> None:
        """Pretend to mute: there is no microphone in the demo."""
        self.muted = not self.muted
        if self.tray is not None:
            self.tray.set_microphone_muted(self.muted)

    def quit(self) -> None:
        """End the demo."""
        self.stop.set()


def wire_overlay(bus: EventBus, overlay: Overlay) -> None:
    """Forward bus events to the overlay, as the orchestrator does in the real app."""
    bus.subscribe_callback(
        StateChanged, lambda event: overlay.set_state(OverlayState[event.new.name])
    )
    bus.subscribe_callback(TranscriptReady, lambda event: overlay.set_transcript(event.text))
    bus.subscribe_callback(AssistantSaid, lambda event: overlay.set_response(event.text))
    bus.subscribe_callback(WakeWordHeard, lambda _event: overlay.flash_wake_word())


async def seed_memory(store: SqliteStore) -> None:
    """Fill the throwaway database with something worth inspecting."""
    await store.upsert_fact("profile.name", "Bozhidar")
    await store.upsert_fact("profile.preference.units", "metric")
    await store.upsert_fact("profile.project.jarvis", "voice assistant, Python + C++")
    await store.upsert_fact("work.editor", "VS Code", source="tool", confidence=0.9)
    await store.upsert_fact("home.city", "Sofia", source="llm", confidence=0.7)
    await store.add_note("Ask Anna about the Q3 roadmap", "Meeting prep")
    await store.add_note("Dear ImGui overlay looks great on 150% DPI")
    await store.add_todo("Record the README demo GIF")
    await store.add_todo("Run the latency benchmark in LOCAL mode")
    now = datetime.now(UTC)
    await store.add_reminder("Stand-up", now + timedelta(minutes=25))
    await store.add_reminder("Call the dentist", now + timedelta(hours=3))


@dataclass(frozen=True)
class Turn:
    """One scripted exchange."""

    said: str
    tool: str
    args: str
    reply: str
    confirm: bool | None = None
    error: str | None = None


TURNS: Final = (
    Turn("What's on my calendar today?", "calendar_agenda", '{"day": "today"}',
         "You have stand-up at 9:30 and a design review at 14:00."),
    Turn("Delete the old build folder", "shell_run", '{"command": "rmdir /s build-old"}',
         "I couldn't delete it: access was denied.", confirm=True,
         error="exit code 5: Access is denied."),
    Turn("Send Anna the roadmap", "email_send", '{"to": "anna@example.com"}',
         "Okay, I won't send it.", confirm=False),
    Turn("Open VS Code", "app_open", '{"name": "code"}', "VS Code is open."),
)  # fmt: skip


class Script:
    """Replays :data:`TURNS` on the bus through a real state machine."""

    def __init__(self, bus: EventBus, overlay: Overlay) -> None:
        self._bus = bus
        self._overlay = overlay
        self._machine = StateMachine(bus)
        self._jitter = itertools.cycle((0.8, 1.0, 1.3, 0.9, 1.6, 1.1))

    async def turn(self, turn: Turn) -> None:
        """Wake, listen, think (tools), speak, idle."""
        self._bus.publish(WakeWordHeard(score=0.93))
        self._machine.transition(AssistantState.LISTENING, "wake word")
        await self._levels(STEP_S * 2)
        self._bus.publish(TranscriptReady(text=turn.said, latency_ms=self._ms(240)))
        self._sample("stt", 240)
        self._machine.transition(AssistantState.THINKING, "transcript ready")
        self._sample("llm_first_token", 650)
        await self._tool(turn)
        self._bus.publish(AssistantSaid(text=turn.reply))
        self._machine.transition(AssistantState.SPEAKING, "reply ready")
        self._sample("tts_first_chunk", 180)
        await self._levels(STEP_S * 2)
        self._sample("turn_total", 2100)
        self._machine.transition(AssistantState.IDLE, "reply finished")
        await asyncio.sleep(STEP_S)

    async def _tool(self, turn: Turn) -> None:
        if turn.confirm is not None:
            self._bus.publish(ConfirmationRequested(tool_name=turn.tool, prompt="Are you sure?"))
            await asyncio.sleep(STEP_S)
            self._bus.publish(ConfirmationResolved(tool_name=turn.tool, approved=turn.confirm))
            if not turn.confirm:
                return
        self._bus.publish(ToolCallStarted(name=turn.tool, args_summary=turn.args))
        await asyncio.sleep(STEP_S)
        self._bus.publish(
            ToolCallFinished(
                name=turn.tool,
                args_summary=turn.args,
                duration_ms=self._ms(420),
                ok=turn.error is None,
                error=turn.error,
            )
        )

    def _ms(self, base: float) -> float:
        return base * next(self._jitter)

    def _sample(self, stage: str, base: float) -> None:
        self._bus.publish(LatencySample(stage=stage, ms=self._ms(base)))

    async def _levels(self, seconds: float) -> None:
        steps = int(seconds / LEVEL_INTERVAL_S)
        for step in range(steps):
            self._overlay.push_level(0.5 + 0.45 * math.sin(step / 3))
            await asyncio.sleep(LEVEL_INTERVAL_S)


def grab_window(title: str, output: Path) -> bool:
    """Save a screenshot of the top-level window called ``title`` (Windows only)."""
    if sys.platform != "win32":
        return False
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return False
    rect = (ctypes.c_long * 4)()
    # The visible frame; GetWindowRect would include Windows 11's invisible resize border.
    status = ctypes.windll.dwmapi.DwmGetWindowAttribute(
        hwnd, _DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)
    )
    if status != 0:
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
    # With all_screens the grab's origin is the virtual screen's top-left,
    # which is negative when a monitor sits left of or above the primary.
    left = user32.GetSystemMetrics(_SM_XVIRTUALSCREEN)
    top = user32.GetSystemMetrics(_SM_YVIRTUALSCREEN)
    bbox = (rect[0] - left, rect[1] - top, rect[2] - left, rect[3] - top)
    output.parent.mkdir(parents=True, exist_ok=True)
    ImageGrab.grab(all_screens=True).crop(bbox).save(output, optimize=True)
    return True


async def run_until(stop: asyncio.Event, duration: float, work: Awaitable[None]) -> None:
    """Run ``work`` until it ends, the user quits, or ``duration`` elapses (0 = no limit)."""
    task = asyncio.ensure_future(work)
    waiters: set[asyncio.Future[object]] = {task, asyncio.ensure_future(stop.wait())}
    done, pending = await asyncio.wait(
        waiters, timeout=duration or None, return_when=asyncio.FIRST_COMPLETED
    )
    for future in pending:
        future.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for future in done:
        future.result()


async def conversation(script: Script, screenshot: Path | None) -> None:
    """Loop the scripted turns forever, taking the screenshot after the first round."""
    for round_number in itertools.count():
        for turn in TURNS:
            await script.turn(turn)
        if round_number == 0 and screenshot is not None:
            saved = await asyncio.to_thread(grab_window, DASHBOARD_TITLE, screenshot)
            print(f"screenshot {'saved to ' + str(screenshot) if saved else 'not taken'}")


async def demo(args: argparse.Namespace, database: Path) -> None:
    """Build the UI surfaces, run the script, tear everything down."""
    bus = EventBus()
    tracker = LatencyTracker()
    tracker.attach(bus)
    overlay = create_overlay(enabled=not args.no_overlay)
    wire_overlay(bus, overlay)
    stop = asyncio.Event()
    callbacks = DemoCallbacks(overlay, stop)
    async with SqliteStore(database) as store:
        await seed_memory(store)
        config = DashboardConfig(title=DASHBOARD_TITLE)
        callbacks.dashboard = await create_dashboard(bus, store, tracker, config=config)
        callbacks.tray = await create_tray(bus, callbacks)
        surfaces = (type(callbacks.dashboard).__name__, type(callbacks.tray).__name__)
        print(f"dashboard: {surfaces[0]}, tray: {surfaces[1]}, overlay: {overlay.running}")
        script = Script(bus, overlay)
        try:
            await run_until(stop, args.duration, conversation(script, args.screenshot))
        finally:
            await shutdown(callbacks, bus, overlay)


async def shutdown(callbacks: DemoCallbacks, bus: EventBus, overlay: Overlay) -> None:
    """Stop every surface, newest first."""
    closers: list[Callable[[], Awaitable[None]]] = []
    if callbacks.tray is not None:
        closers.append(callbacks.tray.stop)
    if callbacks.dashboard is not None:
        closers.append(callbacks.dashboard.stop)
    for close in closers:
        await close()
    overlay.close()
    await bus.close()


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    configure_logging("INFO")
    if sys.platform == "win32":
        # Tk, the overlay and ImageGrab must agree on pixel coordinates.
        # The context is a pointer-sized handle; a plain int would be passed as 32 bits.
        context = ctypes.c_ssize_t(_PER_MONITOR_AWARE_V2)
        if not ctypes.windll.user32.SetProcessDpiAwarenessContext(context):
            print("warning: could not make the demo DPI-aware; the screenshot may be cropped")
    with tempfile.TemporaryDirectory(prefix="jarvis-demo-") as folder:
        asyncio.run(demo(args, Path(folder) / "memory.sqlite3"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
