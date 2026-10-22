"""Tool: CPU, memory, disk, battery and the processes using the most of them.

Measurements come from psutil behind the :class:`SystemProbe` Protocol so
the tool is tested with a fake and psutil (an optional ``desktop`` extra)
is only imported when the real probe is used. Sampling blocks for a short
interval, so it runs in a worker thread.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol

import psutil
from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult

SortKey = Literal["cpu", "memory"]

DEFAULT_SAMPLE_SECONDS: Final = 0.3
"""CPU percentages need two samples; this is the gap between them."""

_BYTES_PER_MB: Final = 1024 * 1024
_BYTES_PER_GB: Final = 1024**3
_MAX_TOP_N: Final = 10


@dataclass(frozen=True, slots=True)
class ProcessInfo:
    """One row of the "top processes" table."""

    pid: int
    name: str
    cpu_percent: float
    memory_mb: float


@dataclass(frozen=True, slots=True)
class SystemSnapshot:
    """Everything the tool reports, measured at one moment.

    Attributes:
        cpu_percent: Whole-machine CPU load over the sample interval.
        memory_used_mb: Physical memory in use.
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


class PsutilProbe:
    """The real probe, backed by psutil."""

    def __init__(self, *, sample_seconds: float = DEFAULT_SAMPLE_SECONDS) -> None:
        self._sample_seconds = sample_seconds

    def snapshot(self, top_n: int, sort_by: SortKey) -> SystemSnapshot:
        """Sample CPU (blocking for ``sample_seconds``) and read the rest."""
        processes = [proc for proc in psutil.process_iter(["pid", "name"]) if _prime(proc)]
        cpu_percent = float(psutil.cpu_percent(interval=self._sample_seconds))
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
            top_processes=_top(processes, top_n, sort_by, psutil.cpu_count() or 1),
        )


def _prime(proc: psutil.Process) -> bool:
    """Take the first CPU sample for ``proc``; False if it vanished meanwhile."""
    try:
        proc.cpu_percent(None)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    return True


def _top(
    processes: list[psutil.Process], top_n: int, sort_by: SortKey, cpu_count: int
) -> tuple[ProcessInfo, ...]:
    rows: list[ProcessInfo] = []
    for proc in processes:
        try:
            rows.append(
                ProcessInfo(
                    pid=proc.pid,
                    name=proc.name(),
                    cpu_percent=proc.cpu_percent(None) / cpu_count,
                    memory_mb=proc.memory_info().rss / _BYTES_PER_MB,
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    key = (lambda row: row.cpu_percent) if sort_by == "cpu" else (lambda row: row.memory_mb)
    return tuple(sorted(rows, key=key, reverse=True)[:top_n])


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
    sort_by: SortKey = Field(default="cpu", description="Rank processes by 'cpu' or 'memory'.")


class SystemMonitorTool(Tool[SystemMonitorInput]):
    """Answers "how is the computer doing" with numbers the user can act on."""

    name = "system_monitor"
    description = (
        "Report CPU load, memory and disk usage, battery level and the processes "
        "using the most CPU or memory."
    )
    Input = SystemMonitorInput
    category = ToolCategory.SYSTEM

    def __init__(self, probe: SystemProbe | None = None) -> None:
        self._probe = probe or PsutilProbe()

    async def run(self, params: SystemMonitorInput, context: ToolContext) -> ToolResult:
        """Measure in a worker thread and phrase the result."""
        del context  # measurements need no injected services
        snapshot = await asyncio.to_thread(self._probe.snapshot, params.top_n, params.sort_by)
        return ToolResult.success(
            _spoken(snapshot),
            detail=_detail(snapshot, params.sort_by),
            data=_data(snapshot),
        )


def _spoken(snapshot: SystemSnapshot) -> str:
    parts = [
        f"CPU is at {snapshot.cpu_percent:.0f} percent",
        f"memory at {snapshot.memory_percent:.0f} percent",
        f"disk at {snapshot.disk_percent:.0f} percent",
    ]
    if snapshot.battery_percent is not None:
        state = "charging" if snapshot.battery_plugged else "on battery"
        parts.append(f"battery at {snapshot.battery_percent:.0f} percent, {state}")
    if snapshot.top_processes:
        heaviest = snapshot.top_processes[0]
        parts.append(f"the heaviest process is {heaviest.name}")
    return ", ".join(parts) + "."


def _detail(snapshot: SystemSnapshot, sort_by: SortKey) -> str:
    lines = [
        f"CPU: {snapshot.cpu_percent:.1f}%",
        f"Memory: {snapshot.memory_used_mb:.0f} / {snapshot.memory_total_mb:.0f} MB "
        f"({snapshot.memory_percent:.0f}%)",
        f"Disk: {snapshot.disk_used_gb:.1f} / {snapshot.disk_total_gb:.1f} GB "
        f"({snapshot.disk_percent:.0f}%)",
    ]
    if snapshot.battery_percent is not None:
        plugged = "plugged in" if snapshot.battery_plugged else "on battery"
        lines.append(f"Battery: {snapshot.battery_percent:.0f}% ({plugged})")
    if snapshot.top_processes:
        lines.append(f"Top processes by {sort_by}:")
        lines.extend(
            f"  {row.name} (pid {row.pid}): {row.cpu_percent:.1f}% CPU, {row.memory_mb:.0f} MB"
            for row in snapshot.top_processes
        )
    return "\n".join(lines)


def _data(snapshot: SystemSnapshot) -> dict[str, JsonValue]:
    return {
        "cpu_percent": snapshot.cpu_percent,
        "memory_used_mb": round(snapshot.memory_used_mb),
        "memory_total_mb": round(snapshot.memory_total_mb),
        "memory_percent": snapshot.memory_percent,
        "disk_used_gb": round(snapshot.disk_used_gb, 1),
        "disk_total_gb": round(snapshot.disk_total_gb, 1),
        "disk_percent": snapshot.disk_percent,
        "battery_percent": snapshot.battery_percent,
        "battery_plugged": snapshot.battery_plugged,
        "top_processes": [
            {
                "pid": row.pid,
                "name": row.name,
                "cpu_percent": round(row.cpu_percent, 1),
                "memory_mb": round(row.memory_mb),
            }
            for row in snapshot.top_processes
        ],
    }


# Design notes:
# - psutil (the optional ``desktop`` extra) is imported at module level on
#   purpose: without it the registry records this plugin as a discovery
#   failure with the ModuleNotFoundError, which the dashboard shows, and the
#   rest of Jarvis starts normally.
# - Per-process CPU is divided by the core count so it is on the same 0-100
#   scale as the machine-wide figure; psutil reports it per core.
# - The probe is injected through the constructor with a default so the
#   registry can still build the tool with no arguments.
