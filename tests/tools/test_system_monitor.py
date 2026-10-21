from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.tools.system_monitor import (
    ProcessInfo,
    PsutilProbe,
    SortKey,
    SystemMonitorTool,
    SystemSnapshot,
)
from tests.agent.fakes import make_context

SNAPSHOT = SystemSnapshot(
    cpu_percent=37.4,
    memory_used_mb=12_288.0,
    memory_total_mb=32_768.0,
    memory_percent=37.5,
    disk_used_gb=410.2,
    disk_total_gb=931.5,
    disk_percent=44.0,
    battery_percent=81.0,
    battery_plugged=False,
    top_processes=(
        ProcessInfo(pid=4242, name="chrome.exe", cpu_percent=22.5, memory_mb=1_800.0),
        ProcessInfo(pid=11, name="python.exe", cpu_percent=9.0, memory_mb=300.0),
    ),
)


@dataclass
class FakeProbe:
    snapshot_value: SystemSnapshot = SNAPSHOT
    calls: list[tuple[int, SortKey]] = field(default_factory=list)

    def snapshot(self, top_n: int, sort_by: SortKey) -> SystemSnapshot:
        self.calls.append((top_n, sort_by))
        return self.snapshot_value


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered_with_the_real_probe() -> None:
    tool = ToolRegistry.discover().get("system_monitor")

    assert isinstance(tool, SystemMonitorTool)
    assert isinstance(tool._probe, PsutilProbe)
    assert not tool.requires_confirmation


async def test_reports_the_snapshot_in_all_three_forms(settings: Settings) -> None:
    probe = FakeProbe()

    result = await SystemMonitorTool(probe).call({}, make_context(settings))

    assert result.ok
    assert probe.calls == [(3, "cpu")]
    assert result.spoken == (
        "CPU is at 37 percent, memory at 38 percent, disk at 44 percent, "
        "battery at 81 percent, on battery, the heaviest process is chrome.exe."
    )
    assert "Memory: 12288 / 32768 MB (38%)" in result.detail
    assert "Battery: 81% (on battery)" in result.detail
    assert "chrome.exe (pid 4242): 22.5% CPU, 1800 MB" in result.detail
    assert result.data is not None
    assert result.data["cpu_percent"] == 37.4
    assert result.data["top_processes"] == [
        {"pid": 4242, "name": "chrome.exe", "cpu_percent": 22.5, "memory_mb": 1800},
        {"pid": 11, "name": "python.exe", "cpu_percent": 9.0, "memory_mb": 300},
    ]


async def test_arguments_are_forwarded_to_the_probe(settings: Settings) -> None:
    probe = FakeProbe()

    await SystemMonitorTool(probe).call({"top_n": 5, "sort_by": "memory"}, make_context(settings))

    assert probe.calls == [(5, "memory")]


async def test_desktop_without_battery_or_processes(settings: Settings) -> None:
    snapshot = SystemSnapshot(
        cpu_percent=5.0,
        memory_used_mb=1.0,
        memory_total_mb=2.0,
        memory_percent=50.0,
        disk_used_gb=1.0,
        disk_total_gb=2.0,
        disk_percent=50.0,
        battery_percent=None,
        battery_plugged=None,
        top_processes=(),
    )

    result = await SystemMonitorTool(FakeProbe(snapshot)).call({"top_n": 0}, make_context(settings))

    assert result.spoken == "CPU is at 5 percent, memory at 50 percent, disk at 50 percent."
    assert "Battery" not in result.detail
    assert "Top processes" not in result.detail
    assert result.data is not None
    assert result.data["battery_percent"] is None


@pytest.mark.parametrize("raw", [{"top_n": 11}, {"top_n": -1}, {"sort_by": "disk"}])
async def test_invalid_arguments_are_rejected(settings: Settings, raw: dict[str, object]) -> None:
    result = await SystemMonitorTool(FakeProbe()).call(raw, make_context(settings))

    assert not result.ok
    assert next(iter(raw)) in result.detail


def test_psutil_probe_measures_a_real_machine() -> None:
    pytest.importorskip("psutil")

    snapshot = PsutilProbe(sample_seconds=0.05).snapshot(top_n=2, sort_by="memory")

    assert 0.0 <= snapshot.cpu_percent <= 100.0
    assert 0.0 < snapshot.memory_used_mb <= snapshot.memory_total_mb
    assert snapshot.disk_total_gb > 0.0
    assert len(snapshot.top_processes) <= 2
    memory = [row.memory_mb for row in snapshot.top_processes]
    assert memory == sorted(memory, reverse=True)


def test_psutil_probe_sorts_by_cpu_too() -> None:
    pytest.importorskip("psutil")

    snapshot = PsutilProbe(sample_seconds=0.05).snapshot(top_n=1, sort_by="cpu")

    assert len(snapshot.top_processes) <= 1
