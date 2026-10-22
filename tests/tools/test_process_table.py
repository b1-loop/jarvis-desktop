import ctypes
import sys
from types import SimpleNamespace

import psutil
import pytest

from jarvis.tools import _process_table
from jarvis.tools._process_table import (
    FallbackProcessSource,
    NtProcessSource,
    ProcessSample,
    PsutilProcessSource,
    SystemProcessInformation,
    default_process_source,
    parse_process_table,
)

ENTRY_BYTES = ctypes.sizeof(SystemProcessInformation)
NAME_AREA_BYTES = 64
TICKS_PER_SECOND = 10_000_000


def kernel_buffer(*entries: tuple[int, str, int, int, int]) -> ctypes.Array[ctypes.c_char]:
    """Lay out entries the way the kernel does: linked by ``next_entry_offset``.

    Each entry is ``(pid, name, user_ticks, kernel_ticks, working_set_bytes)``
    and is followed by its UTF-16 image name, which the entry points at.
    """
    stride = ENTRY_BYTES + NAME_AREA_BYTES
    buffer = ctypes.create_string_buffer(stride * len(entries))
    for index, (pid, name, user, kernel, working_set) in enumerate(entries):
        offset = index * stride
        entry = SystemProcessInformation.from_buffer(buffer, offset)
        entry.next_entry_offset = stride if index < len(entries) - 1 else 0
        entry.unique_process_id = pid
        entry.user_time, entry.kernel_time = user, kernel
        entry.working_set_size = working_set
        encoded = name.encode("utf-16-le")
        name_address = ctypes.addressof(buffer) + offset + ENTRY_BYTES
        ctypes.memmove(name_address, encoded, len(encoded))
        entry.image_name.length = len(encoded)
        entry.image_name.buffer = name_address if name else None
    return buffer


def test_parses_every_entry_in_the_linked_list() -> None:
    buffer = kernel_buffer(
        (0, "", 0, 0, 8192),
        (4242, "chrome.exe", 3 * TICKS_PER_SECOND, TICKS_PER_SECOND, 1800 * 1024 * 1024),
    )

    table = parse_process_table(buffer)

    assert table == (
        ProcessSample(pid=0, name="System Idle Process", cpu_seconds=0.0, memory_bytes=8192),
        ProcessSample(
            pid=4242, name="chrome.exe", cpu_seconds=4.0, memory_bytes=1800 * 1024 * 1024
        ),
    )


def test_a_nameless_non_idle_process_keeps_an_empty_name() -> None:
    (row,) = parse_process_table(kernel_buffer((99, "", 0, 0, 0)))

    assert row.name == ""


def test_psutil_source_reads_only_the_needed_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[list[str]] = []
    readable = SimpleNamespace(
        info={
            "pid": 7,
            "name": "worker.exe",
            "cpu_times": SimpleNamespace(user=1.25, system=0.75),
            "memory_info": SimpleNamespace(rss=4096),
        }
    )
    denied = SimpleNamespace(info={"pid": 8, "name": None, "cpu_times": None, "memory_info": None})

    def process_iter(attrs: list[str]) -> list[SimpleNamespace]:
        requested.append(attrs)
        return [readable, denied]

    monkeypatch.setattr(psutil, "process_iter", process_iter)

    table = PsutilProcessSource().sample()

    assert requested == [["pid", "name", "cpu_times", "memory_info"]]
    assert table == (ProcessSample(pid=7, name="worker.exe", cpu_seconds=2.0, memory_bytes=4096),)


def test_default_source_matches_the_platform() -> None:
    expected = FallbackProcessSource if sys.platform == "win32" else PsutilProcessSource

    assert isinstance(default_process_source(), expected)


class StaticSource:
    def __init__(self, rows: tuple[ProcessSample, ...]) -> None:
        self.rows = rows
        self.calls = 0

    def sample(self) -> tuple[ProcessSample, ...]:
        self.calls += 1
        return self.rows


class BrokenSource:
    def __init__(self) -> None:
        self.calls = 0

    def sample(self) -> tuple[ProcessSample, ...]:
        self.calls += 1
        raise _process_table.ProcessTableError("kernel said no", status=0xC0000022)


def test_fallback_source_prefers_the_primary() -> None:
    primary = StaticSource((ProcessSample(1, "a.exe", 0.0, 1),))
    fallback = StaticSource(())

    assert FallbackProcessSource(primary, fallback).sample() == primary.rows
    assert fallback.calls == 0


def test_fallback_source_switches_for_good_after_one_failure() -> None:
    primary = BrokenSource()
    fallback = StaticSource((ProcessSample(2, "b.exe", 0.0, 1),))
    source = FallbackProcessSource(primary, fallback)

    assert source.sample() == fallback.rows
    assert source.sample() == fallback.rows
    assert primary.calls == 1
    assert fallback.calls == 2


def test_nt_source_reports_an_unexpected_status(monkeypatch: pytest.MonkeyPatch) -> None:
    access_denied = -1073741790  # STATUS_ACCESS_DENIED (0xC0000022) as a signed int
    monkeypatch.setattr(
        _process_table, "_nt_query_system_information", lambda: lambda *_: access_denied
    )

    with pytest.raises(_process_table.ProcessTableError, match="0xC0000022") as raised:
        NtProcessSource().sample()
    assert raised.value.status == 0xC0000022


def test_nt_source_gives_up_when_the_table_keeps_growing(monkeypatch: pytest.MonkeyPatch) -> None:
    length_mismatch = -1073741820  # STATUS_INFO_LENGTH_MISMATCH (0xC0000004)
    monkeypatch.setattr(_process_table, "_INITIAL_BUFFER_BYTES", 16)
    monkeypatch.setattr(
        _process_table, "_nt_query_system_information", lambda: lambda *_: length_mismatch
    )

    with pytest.raises(_process_table.ProcessTableError, match="outgrowing"):
        NtProcessSource().sample()


@pytest.mark.skipif(sys.platform != "win32", reason="NtQuerySystemInformation is Windows-only")
def test_nt_source_agrees_with_psutil_on_this_process() -> None:
    me = psutil.Process()

    table = {row.pid: row for row in NtProcessSource().sample()}

    row = table[me.pid]
    assert row.name.lower() == me.name().lower()
    assert row.memory_bytes > 0
    times = me.cpu_times()
    assert row.cpu_seconds == pytest.approx(times.user + times.system, abs=0.5)
    assert len(table) > 1


@pytest.mark.skipif(sys.platform == "win32", reason="the guard only fires off Windows")
def test_nt_source_refuses_other_platforms() -> None:
    with pytest.raises(_process_table.ProcessTableError, match="only on Windows"):
        NtProcessSource().sample()
