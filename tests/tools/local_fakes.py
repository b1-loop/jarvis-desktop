"""Context builders and fakes for the local (computer, files, dev, notes) tool tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from jarvis.agent.tool import BackgroundJobTracker, MemoryFacade, ToolContext, TrackedProcess
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.os_integration.base import Platform, WindowInfo
from tests.agent.fakes import FakeConfirmation
from tests.fakes import FakeClock

NOTEPAD = WindowInfo(
    handle=101,
    title="notes.txt - Notepad",
    process_name="notepad.exe",
    executable_path="C:/Windows/notepad.exe",
    pid=4000,
)
CODE = WindowInfo(
    handle=202,
    title="tool.py - jarvis - Visual Studio Code",
    process_name="Code.exe",
    executable_path="C:/Program Files/VS Code/Code.exe",
    pid=4004,
)


def local_context(
    settings: Settings,
    *,
    platform: Platform | None = None,
    memory: MemoryFacade | None = None,
    jobs: BackgroundJobTracker | None = None,
    clock: FakeClock | None = None,
) -> ToolContext:
    """A ``ToolContext`` with fakes and whichever services the test supplies."""
    return ToolContext(
        settings=settings,
        bus=EventBus(),
        ask_confirmation=FakeConfirmation(),
        clock=clock or FakeClock(),
        platform=platform,
        memory=memory,
        jobs=jobs,
    )


@dataclass
class RecordingJobs:
    """``BackgroundJobTracker`` that only records what it was asked to track."""

    tracked: list[tuple[str, TrackedProcess]] = field(default_factory=list)

    def track(self, command: str, process: TrackedProcess) -> str:
        self.tracked.append((command, process))
        return f"job-{len(self.tracked)}"
