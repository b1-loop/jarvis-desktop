"""Whole-machine process table: every process's CPU time and memory at once.

``system_monitor`` needs, for every running process, its name, cumulative
CPU time and resident memory. psutil answers that one process at a time,
and on Windows each process it may not open (services, antivirus, anything
elevated: about half of them on a normal desktop) costs a full system-wide
process snapshot of its own, roughly 10 ms each. Three passes over 350
processes took about 6 s. :class:`NtProcessSource` reads the same numbers
for every process from a single ``NtQuerySystemInformation`` call, which is
what psutil does internally for each of those processes, so one table costs
about as much as one of psutil's fallbacks. Other platforms use
:class:`PsutilProcessSource`, where per-process reads are cheap.

The leading underscore keeps the registry from treating this module as a
plugin.
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

import psutil

from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessSample:
    """Raw counters for one process at one moment.

    Attributes:
        pid: Process id.
        name: Executable name, e.g. ``chrome.exe``.
        cpu_seconds: CPU time used since the process started, user plus
            kernel, summed over all cores. Only differences are meaningful.
        memory_bytes: Resident memory (working set / RSS) of this process.
    """

    pid: int
    name: str
    cpu_seconds: float
    memory_bytes: int


class ProcessSource(Protocol):
    """Reads the process table; blocking, called from a worker thread."""

    def sample(self) -> tuple[ProcessSample, ...]:
        """Counters for every process the source can see, right now."""


class PsutilProcessSource:
    """Portable source: one ``psutil.process_iter`` pass with only the needed fields."""

    def sample(self) -> tuple[ProcessSample, ...]:
        """Read every process; ones that vanish or deny access are skipped."""
        rows: list[ProcessSample] = []
        for proc in psutil.process_iter(["pid", "name", "cpu_times", "memory_info"]):
            times, memory = proc.info["cpu_times"], proc.info["memory_info"]
            if times is None or memory is None:
                continue  # process_iter stores None for fields it was denied
            rows.append(
                ProcessSample(
                    pid=proc.info["pid"],
                    name=proc.info["name"] or "",
                    cpu_seconds=float(times.user + times.system),
                    memory_bytes=int(memory.rss),
                )
            )
        return tuple(rows)


_ON_WINDOWS: Final = sys.platform == "win32"
_SYSTEM_PROCESS_INFORMATION_CLASS: Final = 5
_STATUS_INFO_LENGTH_MISMATCH: Final = 0xC0000004
_NTSTATUS_MASK: Final = 0xFFFFFFFF
"""NTSTATUS comes back as a signed C int; masking gives the documented hex value."""
_INITIAL_BUFFER_BYTES: Final = 1024 * 1024
_MAX_ATTEMPTS: Final = 5
_SECONDS_PER_100NS_TICK: Final = 1e-7
_IDLE_PROCESS_NAME: Final = "System Idle Process"


class _UnicodeString(ctypes.Structure):
    _fields_ = (
        ("length", ctypes.c_ushort),
        ("maximum_length", ctypes.c_ushort),
        ("buffer", ctypes.c_void_p),
    )


class SystemProcessInformation(ctypes.Structure):
    """Leading fields of ``SYSTEM_PROCESS_INFORMATION`` (winternl.h), natural alignment.

    Only the fields up to ``working_set_size`` are declared; the entry is
    longer, and ``next_entry_offset`` is what walks the list, so the tail
    never needs a layout.
    """

    _fields_ = (
        ("next_entry_offset", ctypes.c_ulong),
        ("number_of_threads", ctypes.c_ulong),
        ("working_set_private_size", ctypes.c_longlong),
        ("hard_fault_count", ctypes.c_ulong),
        ("number_of_threads_high_watermark", ctypes.c_ulong),
        ("cycle_time", ctypes.c_ulonglong),
        ("create_time", ctypes.c_longlong),
        ("user_time", ctypes.c_longlong),
        ("kernel_time", ctypes.c_longlong),
        ("image_name", _UnicodeString),
        ("base_priority", ctypes.c_long),
        ("unique_process_id", ctypes.c_void_p),
        ("inherited_from_unique_process_id", ctypes.c_void_p),
        ("handle_count", ctypes.c_ulong),
        ("session_id", ctypes.c_ulong),
        ("unique_process_key", ctypes.c_size_t),
        ("peak_virtual_size", ctypes.c_size_t),
        ("virtual_size", ctypes.c_size_t),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
    )


class ProcessTableError(JarvisError):
    """The kernel refused to return the process table.

    Attributes:
        status: The NTSTATUS code, ``None`` when the call never failed outright.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        self.status = status
        super().__init__(message)


class NtProcessSource:
    """Windows source: the whole table from one ``NtQuerySystemInformation`` call."""

    def __init__(self) -> None:
        self._buffer_bytes = _INITIAL_BUFFER_BYTES

    def sample(self) -> tuple[ProcessSample, ...]:
        """Query the kernel once and decode every entry.

        Raises:
            ProcessTableError: The call failed for a reason other than a
                buffer that was too small.
        """
        buffer = self._query()
        return parse_process_table(buffer)

    def _query(self) -> ctypes.Array[ctypes.c_char]:
        query = _nt_query_system_information()
        for _ in range(_MAX_ATTEMPTS):
            buffer = ctypes.create_string_buffer(self._buffer_bytes)
            needed = ctypes.c_ulong(0)
            status = (
                query(
                    _SYSTEM_PROCESS_INFORMATION_CLASS,
                    buffer,
                    self._buffer_bytes,
                    ctypes.byref(needed),
                )
                & _NTSTATUS_MASK
            )
            if status == 0:
                return buffer
            if status != _STATUS_INFO_LENGTH_MISMATCH:
                raise ProcessTableError(f"NtQuerySystemInformation failed: 0x{status:08X}", status)
            # Processes start between calls, so ask for more than reported.
            self._buffer_bytes = max(needed.value, self._buffer_bytes) * 2
        raise ProcessTableError("process table kept outgrowing the buffer")


def _nt_query_system_information() -> Callable[..., int]:
    if sys.platform == "win32":
        function: Callable[..., int] = ctypes.WinDLL("ntdll").NtQuerySystemInformation
        return function
    raise ProcessTableError("NtQuerySystemInformation exists only on Windows")


def parse_process_table(buffer: ctypes.Array[ctypes.c_char]) -> tuple[ProcessSample, ...]:
    """Walk the ``SYSTEM_PROCESS_INFORMATION`` list filled in by the kernel.

    Args:
        buffer: The buffer passed to ``NtQuerySystemInformation``; image
            name pointers point into it, so it must stay alive meanwhile.
    """
    rows: list[ProcessSample] = []
    offset = 0
    while True:
        entry = SystemProcessInformation.from_buffer(buffer, offset)
        rows.append(_decode(entry))
        if entry.next_entry_offset == 0:
            return tuple(rows)
        offset += entry.next_entry_offset


def _decode(entry: SystemProcessInformation) -> ProcessSample:
    pid = int(entry.unique_process_id or 0)
    name = _image_name(entry.image_name) or (_IDLE_PROCESS_NAME if pid == 0 else "")
    ticks = entry.user_time + entry.kernel_time
    return ProcessSample(
        pid=pid,
        name=name,
        cpu_seconds=ticks * _SECONDS_PER_100NS_TICK,
        memory_bytes=int(entry.working_set_size),
    )


def _image_name(image_name: _UnicodeString) -> str:
    if not image_name.buffer or not image_name.length:
        return ""
    # Windows strings are UTF-16LE whatever the host's wchar_t is, so decode the
    # bytes explicitly; that also keeps the parser testable off Windows.
    raw = ctypes.string_at(image_name.buffer, image_name.length)
    return raw.decode("utf-16-le", errors="replace")


class FallbackProcessSource:
    """Uses ``primary`` until it fails once, then ``fallback`` for good.

    The kernel query is an undocumented-in-detail interface; if a future
    Windows build rejects it, the tool should get slower, not break.
    """

    def __init__(self, primary: ProcessSource, fallback: ProcessSource) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_failed = False

    def sample(self) -> tuple[ProcessSample, ...]:
        """Sample from the primary source, switching to the fallback on failure."""
        if not self._primary_failed:
            try:
                return self._primary.sample()
            except ProcessTableError as error:
                log.warning("process_table.fallback", error=str(error), status=error.status)
                self._primary_failed = True
        return self._fallback.sample()


def default_process_source() -> ProcessSource:
    """The fastest source for this platform, with psutil behind it on Windows."""
    portable = PsutilProcessSource()
    return FallbackProcessSource(NtProcessSource(), portable) if _ON_WINDOWS else portable


# Design notes:
# - ctypes rather than a C++ extension: this is one documented (winternl.h)
#   system call per sample with no hot loop around it; decoding ~350
#   entries in Python takes a few milliseconds. A pybind11 module would add
#   a build dependency to a pure-Python tool for no measurable gain, and
#   text-only development must keep working without the native build
#   (ADR-0002). ADR-0026 records the measurements.
# - ``working_set_size`` is what psutil reports as ``rss`` on Windows, so
#   both sources mean the same thing by "memory".
# - The fallback is logged once at warning level and then sticks, so a
#   broken kernel query costs one failed call, not one per request.
# - The buffer size is remembered between calls so a steady-state sample is
#   a single system call.
