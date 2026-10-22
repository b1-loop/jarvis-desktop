# 0017. Background processes for dev tools: spawn under the sandbox, track in memory

Date: 2026-10-22
Status: Accepted

## Context

"Start the dev server" is the first capability whose process must outlive
the tool call. Everything else in dev mode runs through
`utils/sandbox.run_sandboxed`, which waits for the process, enforces a
timeout and captures output through pipes. A dev server must not be
killed by that timeout, nobody reads its pipes (a full pipe would freeze
it), and when it exits (crashed, or the user stopped it) Jarvis should say
so. The proactive layer already has the consumer side:
`LongRunningCommandTrigger` reads finished commands from any
`CommandWatcher` (ADR-0007). Two rules must survive: no tool spawns a
process outside the sandbox policy, and the tool layer does not depend on
the proactive layer's internals or the other way round.

## Decision

Add `spawn_sandboxed(command, policy, *, log_path, cwd)` to
`utils/sandbox.py`. It shares one `_prepare` step with `run_sandboxed`
(denylist and operator refusal, working-root check, binary resolution) and
the same scrubbed environment, then starts the process with stdin closed
and stdout and stderr appended to a log file under `<data_dir>/logs`. It
returns immediately; the policy timeout does not apply.

Track such processes with `jarvis.tools._jobs.BackgroundJobs`, an in-memory
registry with one watcher task per process. It satisfies two Protocols
structurally: `ToolContext.jobs` (`BackgroundJobTracker.track`, defined in
`agent/tool.py`) and `proactive.triggers.CommandWatcher.pop_finished`. The
composition root creates one instance and hands it to both. When the
context has no tracker, the tool keeps a private fallback tracker so the
process is still reaped, and it does not promise to report the exit.

## Alternatives considered

- **Raw `subprocess.Popen` in the tool** - quickest, and a second execution
  path that skips the policy. Rejected by the "sandbox shell commands" rule.
- **`run_sandboxed` in a background task with a huge timeout** - keeps one
  function but holds the output in pipes and memory for hours, and returns
  no pid to the user.
- **Persist jobs in SQLite** - would survive a restart, but a restarted
  Jarvis has no handle on the old process and could only guess at its exit
  code. In-memory matches what can actually be observed.
- **Kill background jobs on shutdown** - tidy, but the server was started
  for the user and is theirs; closing Jarvis should not take down their
  work. `BackgroundJobs.close()` stops watching only.

## Consequences

- Every process Jarvis starts, short or long, passes the same policy code;
  auditing the sandbox covers both.
- A dev server's output is in a file the user (and a future "show the dev
  server log" tool) can read; nothing blocks on an unread pipe.
- Exit announcements depend on wiring: the composition root must pass the
  same `BackgroundJobs` to `ToolContext.jobs` and to
  `LongRunningCommandTrigger`. Without it the tool still works and says less.
- Jobs are forgotten on restart, by design.
