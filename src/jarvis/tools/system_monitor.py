"""Tool: CPU, memory, disk, battery and the processes using the most of them.

Measurements sit behind the :class:`SystemProbe` Protocol so the tool is
tested with a fake. The real probe reads machine totals from psutil (an
optional ``desktop`` extra) and the per-process table from a
:class:`~jarvis.tools._process_table.ProcessSource`, which on Windows is a
single kernel call for every process instead of psutil's per-process reads
(ADR-0026). One request costs about a quarter of a second, almost all of it
the CPU sampling window, and runs in a worker thread.

The output keeps the whole computer's numbers and each process's own
numbers apart in wording, layout and data keys, because a small model
otherwise reads one process's memory back as "your RAM".
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol

import psutil
from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._process_table import ProcessSample, ProcessSource, default_process_source

SortKey = Literal["cpu", "memory"]

DEFAULT_SAMPLE_SECONDS: Final = 0.2
"""CPU load is a difference between two readings; this is the gap between them.

Windows updates CPU times on a 15.6 ms clock tick, so 0.2 s resolves about
a percent of one core while keeping the whole request under 300 ms.
"""

_BYTES_PER_MB: Final = 1024 * 1024
_MB_PER_GB: Final = 1024
_BYTES_PER_GB: Final = 1024**3
_MAX_TOP_N: Final = 10
_PERCENT: Final = 100.0
_IDLE_PID: Final = 0
"""The idle process "uses" all unused CPU; listing it as the heaviest is noise."""


@dataclass(frozen=True, slots=True)
class ProcessInfo:
    """One row of the "top processes" table: a single process's own usage."""

    pid: int
    name: str
    cpu_percent: float
    memory_mb: float


@dataclass(frozen=True, slots=True)
class SystemSnapshot:
    """Everything the tool reports, measured at one moment.

    Attributes:
        cpu_percent: Whole-machine CPU load over the sample interval.
        memory_used_mb: Physical memory in use by the whole computer.
        memory_total_mb: Physical memory installed.
        memory_percent: ``used / total``.
        disk_used_gb: Space used on the drive holding the home directory.
        disk_total_gb: Size of that drive.
        disk_percent: ``used / total``.
        battery_percent: Charge level, ``None`` on machines without a battery.
        battery_plugged: Whether mains power is connected, ``None`` if unknown.
        top_processes: Heaviest processes by the requested key.
    """

    cpu_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    battery_percent: float | None
    battery_plugged: bool | None
    top_processes: tuple[ProcessInfo, ...]


class SystemProbe(Protocol):
    """Source of system measurements; blocking, called from a worker thread."""

    def snapshot(self, top_n: int, sort_by: SortKey) -> SystemSnapshot:
        """Measure everything once, including the ``top_n`` heaviest processes."""


class LiveSystemProbe:
    """The real probe: psutil for machine totals, a process source for the table."""

    def __init__(
        self,
        *,
        processes: ProcessSource | None = None,
        sample_seconds: float = DEFAULT_SAMPLE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Build the probe.

        Args:
            processes: Per-process counters; the platform's fastest by default.
            sample_seconds: CPU sampling window; ``0`` measures since the
                previous call instead of blocking (used by tests).
            clock: Monotonic time source, injected so tests control the window.
        """
        self._processes = processes or default_process_source()
        self._sample_seconds = sample_seconds
        self._clock = clock

    def snapshot(self, top_n: int, sort_by: SortKey) -> SystemSnapshot:
        """Sample CPU over the window (blocking) and read the rest once."""
        # Each clock reading is taken just before its table is read, so the
        # elapsed time spans the same interval as the two sets of counters.
        started = self._clock()
        before = self._processes.sample() if top_n else ()
        cpu_percent = float(psutil.cpu_percent(interval=self._sample_seconds or None))
        elapsed = self._clock() - started
        after = self._processes.sample() if top_n else ()
        cpu_count = psutil.cpu_count() or 1
        top = rank_processes(before, after, elapsed * cpu_count, top_n, sort_by)
        return _with_totals(cpu_percent, top)


def rank_processes(
    before: tuple[ProcessSample, ...],
    after: tuple[ProcessSample, ...],
    core_seconds: float,
    top_n: int,
    sort_by: SortKey,
) -> tuple[ProcessInfo, ...]:
    """Turn two process-table samples into the ``top_n`` heaviest processes.

    Args:
        before: Table at the start of the window.
        after: Table at the end of the window; its processes are the ones listed.
        core_seconds: Window length times the core count, so a process's
            CPU share is on the same 0-100 scale as the machine-wide figure.
        top_n: How many rows to keep.
        sort_by: Ranking key.
    """
    # Keyed by (pid, name) so a pid reused by a new process is not compared
    # with the old one; a process that started inside the window counts as idle.
    start = {(row.pid, row.name): row.cpu_seconds for row in before}
    rows = [
        ProcessInfo(
            pid=row.pid,
            name=row.name,
            cpu_percent=_share(_cpu_used(row, start), core_seconds),
            memory_mb=row.memory_bytes / _BYTES_PER_MB,
        )
        for row in after
        if row.pid != _IDLE_PID
    ]
    key = (lambda row: row.cpu_percent) if sort_by == "cpu" else (lambda row: row.memory_mb)
    return tuple(sorted(rows, key=key, reverse=True)[:top_n])


def _cpu_used(row: ProcessSample, start: dict[tuple[int, str], float]) -> float:
    return row.cpu_seconds - start.get((row.pid, row.name), row.cpu_seconds)


def _share(cpu_seconds: float, core_seconds: float) -> float:
    if core_seconds <= 0.0:
        return 0.0
    return min(max(cpu_seconds / core_seconds * _PERCENT, 0.0), _PERCENT)


def _with_totals(cpu_percent: float, top: tuple[ProcessInfo, ...]) -> SystemSnapshot:
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.home().anchor or Path.home()))
    battery_percent, battery_plugged = _battery()
    return SystemSnapshot(
        cpu_percent=cpu_percent,
        memory_used_mb=memory.used / _BYTES_PER_MB,
        memory_total_mb=memory.total / _BYTES_PER_MB,
        memory_percent=float(memory.percent),
        disk_used_gb=disk.used / _BYTES_PER_GB,
        disk_total_gb=disk.total / _BYTES_PER_GB,
        disk_percent=float(disk.percent),
        battery_percent=battery_percent,
        battery_plugged=battery_plugged,
        top_processes=top,
    )


def _battery() -> tuple[float | None, bool | None]:
    try:
        battery = psutil.sensors_battery()
    except (AttributeError, RuntimeError):
        return None, None
    if battery is None:
        return None, None
    return float(battery.percent), bool(battery.power_plugged)


class SystemMonitorInput(ToolInput):
    """Arguments for :class:`SystemMonitorTool`."""

    top_n: int = Field(
        default=3,
        ge=0,
        le=_MAX_TOP_N,
        description="How many of the heaviest processes to list (0 for none).",
    )
    sort_by: SortKey = Field(
        default="cpu",
        description="Rank processes by 'cpu' or 'memory'; use 'memory' for RAM questions.",
    )


class SystemMonitorTool(Tool[SystemMonitorInput]):
    """Answers "how is the computer doing" with numbers the user can act on."""

    name = "system_monitor"
    description = (
        "Report the whole computer's CPU load, RAM and disk usage and battery level, "
        "plus the processes using the most CPU or memory. The system RAM figure is "
        "the answer to 'how much RAM is used'; process figures are each process's own share."
    )
    Input = SystemMonitorInput
    category = ToolCategory.SYSTEM
    keywords = (
        "ram",
        "memory",
        "cpu",
        "battery",
        "disk",
        "slow",
        "minne",
        "arbetsminne",
        "processor",
        "batteri",
        "hårddisk",
        "långsam",
    )

    def __init__(self, probe: SystemProbe | None = None) -> None:
        self._probe = probe or LiveSystemProbe()

    async def run(self, params: SystemMonitorInput, context: ToolContext) -> ToolResult:
        """Measure in a worker thread and phrase the result."""
        del context  # measurements need no injected services
        snapshot = await asyncio.to_thread(self._probe.snapshot, params.top_n, params.sort_by)
        return ToolResult.success(
            _spoken(snapshot, params.sort_by),
            detail=_detail(snapshot, params.sort_by),
            data=_data(snapshot, params.sort_by),
        )


def _gb(megabytes: float) -> float:
    return megabytes / _MB_PER_GB


def _spoken(snapshot: SystemSnapshot, sort_by: SortKey) -> str:
    parts = [
        f"System CPU is at {snapshot.cpu_percent:.0f} percent",
        f"system RAM is {_gb(snapshot.memory_used_mb):.1f} of "
        f"{_gb(snapshot.memory_total_mb):.1f} gigabytes used "
        f"({snapshot.memory_percent:.0f} percent)",
        f"disk at {snapshot.disk_percent:.0f} percent",
    ]
    if snapshot.battery_percent is not None:
        state = "charging" if snapshot.battery_plugged else "on battery"
        parts.append(f"battery at {snapshot.battery_percent:.0f} percent, {state}")
    if snapshot.top_processes:
        heaviest = snapshot.top_processes[0]
        parts.append(f"the process using the most {_key_noun(sort_by)} is {heaviest.name}")
    return ", ".join(parts) + "."


def _key_noun(sort_by: SortKey) -> str:
    return "CPU" if sort_by == "cpu" else "memory"


def _detail(snapshot: SystemSnapshot, sort_by: SortKey) -> str:
    lines = [
        "Whole computer (system totals):",
        f"  System CPU: {snapshot.cpu_percent:.1f}% of all cores",
        f"  System RAM: used {_gb(snapshot.memory_used_mb):.1f} of "
        f"{_gb(snapshot.memory_total_mb):.1f} GB ({snapshot.memory_percent:.0f}%)",
        f"  Disk: used {snapshot.disk_used_gb:.1f} of {snapshot.disk_total_gb:.1f} GB "
        f"({snapshot.disk_percent:.0f}%)",
    ]
    if snapshot.battery_percent is not None:
        plugged = "plugged in" if snapshot.battery_plugged else "on battery"
        lines.append(f"  Battery: {snapshot.battery_percent:.0f}% ({plugged})")
    if snapshot.top_processes:
        lines.append(
            f"Top processes by {_key_noun(sort_by)} "
            "(each process's own usage, not the system total):"
        )
        lines.extend(
            f"  {row.name} (pid {row.pid}): {row.memory_mb:.0f} MB RAM, {row.cpu_percent:.1f}% CPU"
            for row in snapshot.top_processes
        )
    return "\n".join(lines)


def _data(snapshot: SystemSnapshot, sort_by: SortKey) -> dict[str, JsonValue]:
    return {
        "system": {
            "cpu_percent": snapshot.cpu_percent,
            "ram_used_gb": round(_gb(snapshot.memory_used_mb), 1),
            "ram_total_gb": round(_gb(snapshot.memory_total_mb), 1),
            "ram_percent": snapshot.memory_percent,
            "disk_used_gb": round(snapshot.disk_used_gb, 1),
            "disk_total_gb": round(snapshot.disk_total_gb, 1),
            "disk_percent": snapshot.disk_percent,
            "battery_percent": snapshot.battery_percent,
            "battery_plugged": snapshot.battery_plugged,
        },
        "top_processes_sorted_by": sort_by,
        "top_processes": [
            {
                "pid": row.pid,
                "name": row.name,
                "process_ram_mb": round(row.memory_mb),
                "process_cpu_percent": round(row.cpu_percent, 1),
            }
            for row in snapshot.top_processes
        ],
    }


# Design notes:
# - psutil (the optional ``desktop`` extra) is imported at module level on
#   purpose: without it the registry records this plugin as a discovery
#   failure with the ModuleNotFoundError, which the dashboard shows, and the
#   rest of Jarvis starts normally.
# - No background sampler: one process table costs ~12 ms on Windows, so a
#   fresh measurement per request is fast enough and Jarvis spends no CPU
#   while nobody asks (ADR-0026).
# - Per-process CPU is divided by window length times core count so it is
#   on the same 0-100 scale as the machine-wide figure.
# - System RAM is reported in GB and process memory in MB, under separate
#   headings and ``system`` / ``process_*`` data keys: qwen2.5:7b used to
#   read the heaviest process's memory as the machine's RAM (ADR-0020).
# - The probe is injected through the constructor with a default so the
#   registry can still build the tool with no arguments.
