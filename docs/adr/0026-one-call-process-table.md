# 0026. system_monitor reads the process table in one kernel call

Date: 2026-10-22
Status: Accepted

## Context

"Hur mycket RAM används?" and "vad drar mest CPU?" took about 10 s before
the first spoken token (ADR-0020's measurements). The model was not the
problem: `system_monitor` itself took about 6 s.

The tool asked psutil for every process's CPU time and memory, one process
at a time, in three passes (prime CPU, read CPU, read memory). Measured on
the development machine (Windows 11, 24 logical cores, ~355 processes):

| Step | Time |
|---|---|
| `process_iter(["pid", "name"])` | 2 ms |
| `cpu_percent()` on every process | 2.0 s |
| `memory_info()` on every process | 2.1 s |
| `virtual_memory()`, `disk_usage()`, `sensors_battery()` | < 1 ms together |

Half the processes (services, antivirus, anything elevated) cannot be
opened by a normal user. For each of those psutil falls back to
`NtQuerySystemInformation(SystemProcessInformation)`, which returns the
*whole* system's process table, and picks one entry out of it: about
10 ms per process, 178 processes, per pass. `oneshot()` only shares that
fallback between two fields of the same process, so it saves one pass of
three.

The same answer had a second defect: `qwen2.5:7b` read the heaviest
process's memory ("python.exe: 1800 MB") back to the user as "your RAM",
because the system total and the per-process rows used the same word and
the data keys (`memory_used_mb`, `memory_mb`) differed only by a suffix.

## Decision

Read the process table the way psutil's fallback does, but once for all
processes: `NtProcessSource` calls `NtQuerySystemInformation` through
ctypes and decodes every `SYSTEM_PROCESS_INFORMATION` entry (pid, image
name, user + kernel time, working set). One table costs about 12 ms.

A request now takes one table, the machine-wide CPU window
(`psutil.cpu_percent(interval=0.2)`), and a second table; per-process CPU
is the difference between the two tables over the window. Machine totals
still come from psutil. The table sits behind a `ProcessSource` Protocol,
so the ranking is unit-tested with a fake source; `PsutilProcessSource`
(one `process_iter` pass with only the four needed fields) serves other
platforms and is the fallback on Windows if the kernel call ever fails.

The output separates the two kinds of number everywhere the model looks:
"Whole computer (system totals): System RAM: used 21.1 of 31.1 GB" in GB,
then "Top processes by memory (each process's own usage, not the system
total)" in MB, and data keys `system.ram_used_gb` versus
`top_processes[].process_ram_mb`.

Results, median of repeated `SystemMonitorTool.run` calls on the same
machine:

| Request | Before (7 runs) | After (9 runs) |
|---|---|---|
| RAM (`sort_by=memory`, 3 processes) | 6394 ms | 224 ms |
| CPU / top processes (`sort_by=cpu`, 3 processes) | 5959 ms | 224 ms |
| Totals only (`top_n=0`) | 6159 ms | 202 ms |

The remaining 200 ms is the CPU sampling window, which runs in a worker
thread (`asyncio.to_thread`), so the event loop is never blocked.

## Alternatives considered

- **Background sampler** keeping a cached snapshot fresh. With psutil a
  pass costs about 2 s of CPU, so refreshing every 5 s would keep 40 % of
  a core busy while nobody asks; with the one-call table it is unnecessary,
  since a fresh measurement already fits the latency budget. No idle cost
  is better than a small one.
- **Skip processes psutil cannot open.** Fast, but those are exactly the
  ones that often top the list (Defender, `svchost.exe`, `dwm.exe`), so
  the answer would be wrong.
- **A C++ sampler in the native extension.** It would make the same system
  call. Decoding ~355 entries in Python takes a few milliseconds, so the
  native build would buy nothing measurable and would make a pure-Python
  tool depend on the compiled extension (ADR-0002).
- **Skip the CPU window for RAM-only questions.** Would save 200 ms, but
  Windows has no instantaneous CPU load figure; the alternative is "load
  since the last call", which can span minutes and reads as current. The
  tool reports CPU in every answer, so it measures it honestly.

## Consequences

- RAM and CPU questions answer as fast as the model can speak, and a
  regression would show in `test_live_probe_measures_a_real_machine_quickly`.
- The tool now depends on the layout of `SYSTEM_PROCESS_INFORMATION`. It
  is declared in the SDK's `winternl.h` and psutil relies on the same
  layout, but Microsoft reserves the right to change it; a failed call
  falls back to psutil (slower, never broken) and is logged once.
- `SystemSnapshot`'s data shape changed (`system`, `top_processes_sorted_by`,
  `process_ram_mb`, `process_cpu_percent`). Nothing else read the old keys.
- `PsutilProbe` is now `LiveSystemProbe`, because psutil no longer supplies
  the process table on Windows.
