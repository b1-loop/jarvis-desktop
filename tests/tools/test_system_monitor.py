import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import psutil
import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.tools._process_table import ProcessSample
from jarvis.tools.system_monitor import (
    LiveSystemProbe,
    ProcessInfo,
    SortKey,
    SystemMonitorTool,
    SystemSnapshot,
    rank_processes,
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
MB = 1024 * 1024


@dataclass
class FakeProbe:
    snapshot_value: SystemSnapshot = SNAPSHOT
    calls: list[tuple[int, SortKey]] = field(default_factory=list)

    def snapshot(self, top_n: int, sort_by: SortKey) -> SystemSnapshot:
        self.calls.append((top_n, sort_by))
        return self.snapshot_value


class FakeProcessSource:
    """Hands out prepared process tables in order, one per ``sample`` call."""

    def __init__(self, *tables: tuple[ProcessSample, ...]) -> None:
        self._tables: Iterator[tuple[ProcessSample, ...]] = iter(tables)
        self.calls = 0

    def sample(self) -> tuple[ProcessSample, ...]:
        self.calls += 1
        return next(self._tables)


def fake_clock(*readings: float) -> Callable[[], float]:
    return iter(readings).__next__


def sample(pid: int, name: str, cpu_seconds: float, memory_mb: int) -> ProcessSample:
    return ProcessSample(pid=pid, name=name, cpu_seconds=cpu_seconds, memory_bytes=memory_mb * MB)


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered_with_the_live_probe() -> None:
    tool = ToolRegistry.discover().get("system_monitor")

    assert isinstance(tool, SystemMonitorTool)
    assert isinstance(tool._probe, LiveSystemProbe)
    assert not tool.requires_confirmation


async def test_reports_the_snapshot_in_all_three_forms(settings: Settings) -> None:
    probe = FakeProbe()

    result = await SystemMonitorTool(probe).call({}, make_context(settings))

    assert result.ok
    assert probe.calls == [(3, "cpu")]
    assert result.spoken == (
        "System CPU is at 37 percent, system RAM is 12.0 of 32.0 gigabytes used (38 percent), "
        "disk at 44 percent, battery at 81 percent, on battery, "
        "the process using the most CPU is chrome.exe."
    )
    assert "System RAM: used 12.0 of 32.0 GB (38%)" in result.detail
    assert "Battery: 81% (on battery)" in result.detail
    assert "chrome.exe (pid 4242): 1800 MB RAM, 22.5% CPU" in result.detail
    assert result.data is not None
    assert result.data["system"] == {
        "cpu_percent": 37.4,
        "ram_used_gb": 12.0,
        "ram_total_gb": 32.0,
        "ram_percent": 37.5,
        "disk_used_gb": 410.2,
        "disk_total_gb": 931.5,
        "disk_percent": 44.0,
        "battery_percent": 81.0,
        "battery_plugged": False,
    }
    assert result.data["top_processes"] == [
        {"pid": 4242, "name": "chrome.exe", "process_ram_mb": 1800, "process_cpu_percent": 22.5},
        {"pid": 11, "name": "python.exe", "process_ram_mb": 300, "process_cpu_percent": 9.0},
    ]


async def test_process_memory_is_labelled_apart_from_system_ram(settings: Settings) -> None:
    result = await SystemMonitorTool(FakeProbe()).call(
        {"sort_by": "memory"}, make_context(settings)
    )

    system_block, process_block = result.detail.split("Top processes", maxsplit=1)
    assert system_block.startswith("Whole computer (system totals):")
    assert "System RAM" in system_block
    assert "chrome.exe" not in system_block
    assert process_block.startswith(" by memory (each process's own usage, not the system total)")
    assert "the process using the most memory is chrome.exe" in result.spoken
    assert result.data is not None
    assert result.data["top_processes_sorted_by"] == "memory"


async def test_arguments_are_forwarded_to_the_probe(settings: Settings) -> None:
    probe = FakeProbe()

    await SystemMonitorTool(probe).call({"top_n": 5, "sort_by": "memory"}, make_context(settings))

    assert probe.calls == [(5, "memory")]


async def test_desktop_without_battery_or_processes(settings: Settings) -> None:
    snapshot = SystemSnapshot(
        cpu_percent=5.0,
        memory_used_mb=1024.0,
        memory_total_mb=2048.0,
        memory_percent=50.0,
        disk_used_gb=1.0,
        disk_total_gb=2.0,
        disk_percent=50.0,
        battery_percent=None,
        battery_plugged=None,
        top_processes=(),
    )

    result = await SystemMonitorTool(FakeProbe(snapshot)).call({"top_n": 0}, make_context(settings))

    assert result.spoken == (
        "System CPU is at 5 percent, system RAM is 1.0 of 2.0 gigabytes used (50 percent), "
        "disk at 50 percent."
    )
    assert "Battery" not in result.detail
    assert "Top processes" not in result.detail
    assert result.data is not None
    assert result.data["top_processes"] == []


@pytest.mark.parametrize("raw", [{"top_n": 11}, {"top_n": -1}, {"sort_by": "disk"}])
async def test_invalid_arguments_are_rejected(settings: Settings, raw: dict[str, object]) -> None:
    result = await SystemMonitorTool(FakeProbe()).call(raw, make_context(settings))

    assert not result.ok
    assert next(iter(raw)) in result.detail


def test_ranking_uses_cpu_time_deltas_over_the_window() -> None:
    before = (sample(10, "busy.exe", 5.0, 100), sample(20, "quiet.exe", 50.0, 900))
    after = (sample(10, "busy.exe", 5.5, 100), sample(20, "quiet.exe", 50.1, 900))

    top = rank_processes(before, after, core_seconds=1.0, top_n=2, sort_by="cpu")

    assert [row.name for row in top] == ["busy.exe", "quiet.exe"]
    assert top[0].cpu_percent == pytest.approx(50.0)
    assert top[1].cpu_percent == pytest.approx(10.0)
    assert top[1].memory_mb == pytest.approx(900.0)


def test_ranking_by_memory_ignores_cpu() -> None:
    table = (sample(10, "small.exe", 1.0, 100), sample(20, "big.exe", 1.0, 900))

    top = rank_processes(table, table, core_seconds=1.0, top_n=1, sort_by="memory")

    assert [row.name for row in top] == ["big.exe"]


def test_ranking_skips_the_idle_process_and_treats_new_or_reused_pids_as_idle() -> None:
    before = (sample(0, "System Idle Process", 0.0, 0), sample(30, "old.exe", 0.0, 10))
    after = (
        sample(0, "System Idle Process", 9.0, 0),
        sample(30, "reused.exe", 400.0, 10),
        sample(40, "new.exe", 300.0, 10),
    )

    top = rank_processes(before, after, core_seconds=1.0, top_n=5, sort_by="cpu")

    assert {row.name for row in top} == {"reused.exe", "new.exe"}
    assert all(row.cpu_percent == 0.0 for row in top)


@pytest.mark.parametrize(
    ("used", "core_seconds", "expected"),
    [(5.0, 1.0, 100.0), (-1.0, 1.0, 0.0), (1.0, 0.0, 0.0)],
)
def test_cpu_share_is_clamped_to_a_percentage(
    used: float, core_seconds: float, expected: float
) -> None:
    before = (sample(1, "a.exe", 0.0, 1),)
    after = (sample(1, "a.exe", used, 1),)

    (row,) = rank_processes(before, after, core_seconds, top_n=1, sort_by="cpu")

    assert row.cpu_percent == expected


def test_live_probe_samples_the_table_around_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(psutil, "cpu_count", lambda: 2)
    source = FakeProcessSource(
        (sample(7, "worker.exe", 1.0, 50),), (sample(7, "worker.exe", 1.5, 60),)
    )
    probe = LiveSystemProbe(processes=source, sample_seconds=0.0, clock=fake_clock(0.0, 0.5))

    snapshot = probe.snapshot(top_n=1, sort_by="cpu")

    assert source.calls == 2
    (row,) = snapshot.top_processes
    assert row.cpu_percent == pytest.approx(50.0)  # 0.5 s of CPU over 0.5 s x 2 cores
    assert row.memory_mb == pytest.approx(60.0)
    assert 0.0 < snapshot.memory_used_mb <= snapshot.memory_total_mb


def test_live_probe_skips_the_process_table_when_no_rows_are_wanted() -> None:
    source = FakeProcessSource()
    probe = LiveSystemProbe(processes=source, sample_seconds=0.0, clock=fake_clock(0.0, 0.1))

    snapshot = probe.snapshot(top_n=0, sort_by="cpu")

    assert source.calls == 0
    assert snapshot.top_processes == ()


def test_live_probe_measures_a_real_machine_quickly() -> None:
    probe = LiveSystemProbe(sample_seconds=0.05)

    started = time.perf_counter()
    snapshot = probe.snapshot(top_n=2, sort_by="memory")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert 0.0 <= snapshot.cpu_percent <= 100.0
    assert 0.0 < snapshot.memory_used_mb <= snapshot.memory_total_mb
    assert snapshot.disk_total_gb > 0.0
    assert 0 < len(snapshot.top_processes) <= 2
    memory = [row.memory_mb for row in snapshot.top_processes]
    assert memory == sorted(memory, reverse=True)


def test_live_probe_ranks_a_real_machine_by_cpu() -> None:
    snapshot = LiveSystemProbe(sample_seconds=0.05).snapshot(top_n=3, sort_by="cpu")

    cpu = [row.cpu_percent for row in snapshot.top_processes]
    assert cpu == sorted(cpu, reverse=True)
    assert all(0.0 <= value <= 100.0 for value in cpu)
