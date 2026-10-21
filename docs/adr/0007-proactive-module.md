# 0007. Proactive module: one scheduler, polled triggers, YAML routines

Date: 2026-10-21
Status: Accepted

## Context

Phase 3 asks Jarvis to speak first: warn before a meeting, when the battery
runs low, when a long command finishes, and run a "good morning" briefing at
07:30 or on request. Four properties shape the design:

- The sources are heterogeneous (calendar API, OS battery, SQLite reminders,
  a process watcher) and most of them do not exist yet; steps 6 to 8 are
  being built in parallel.
- Everything else in Jarvis runs on one asyncio loop with a single
  orchestrator that owns the speaker. Unsolicited speech must go through it,
  not around it.
- Wall-clock schedules must survive midnight, DST and a laptop that sleeps.
- The user must be able to add or change a routine without writing Python.

## Decision

`jarvis.proactive` is three cooperating pieces, all pure Python:

1. **One `Scheduler`** on the event loop with an injectable `Clock`
   (`now()` and `sleep()`), supporting one-shot, interval and daily jobs.
   Daily jobs recompute "next run" from calendar date + wall time in a
   `zoneinfo` zone on every plan, never by adding 24 hours, and all
   comparisons happen in UTC. Each run is its own task; exceptions are
   logged with a traceback, counted on the job and never stop the loop.
2. **A `TriggerRunner`** that registers every `Trigger` as an interval job
   on that scheduler, de-duplicates notifications per key within a cooldown
   and hands survivors to a `NotificationSink`. Concrete triggers depend on
   the smallest possible source Protocols (`CalendarSource`,
   `BatterySource`, `ReminderSource`, `CommandWatcher`) defined next to them.
3. **YAML routines** validated by pydantic (`extra="forbid"`, discriminated
   trigger union, template placeholder checks) and executed by a
   `RoutineRunner` through `ToolInvoker` and `Speaker` Protocols, with
   per-step and whole-routine timeouts and a continue-or-abort policy.

## Alternatives considered

- **A thread or task per source with its own loop** – each integration
  would re-implement polling, error handling, backoff and shutdown; failures
  would be invisible to each other and to the dashboard. One scheduler gives
  one place to see every pending job and one shutdown path. The cost, a
  polling delay of up to `proactive_poll_seconds`, is irrelevant for
  "meeting in ten minutes".
- **Push-based sources (webhooks, OS events)** – strictly better latency,
  but every source would need its own listener and most of ours (Google
  Calendar, psutil battery) are poll-only anyway. The `Trigger` Protocol does
  not forbid a push source: it can buffer events and return them from
  `check()`.
- **Importing tools and the agent directly** – the proactive module would
  depend on the tool registry, the orchestrator and the integrations, none
  of which exist in this step and all of which sit in other layers. Two
  tiny Protocols (`invoke(name, args) -> str`, `say(text)`) keep the
  dependency arrow pointing the right way and let the whole module be tested
  with fakes in milliseconds.
- **Routines in Python (decorated functions)** – flexible, but editing them
  means editing the codebase, and a mistake is a traceback at startup
  instead of "steps.1.tool: Field required" with a file name.
- **Cron syntax for schedules** – more expressive, but the only shapes the
  features need are "every N", "at HH:MM on these days" and "once at T".
  A cron parser would add a dependency to express those three.
- **`datetime + 24h` for daily jobs** – simpler, and wrong twice a year.

## Consequences

- The agent (step 6) implements `ToolInvoker` over its registry and raises
  `UnknownToolError` for unknown names; the orchestrator (step 7) implements
  `Speaker` and `NotificationSink`; integrations (step 8) implement the
  source Protocols. Each is a few lines.
- `tzdata` becomes a runtime dependency: Windows has no system zoneinfo
  database, and the DST guarantee depends on real zone rules.
- Routine YAML must quote times (`at: "07:30"`); PyYAML would otherwise read
  `07:30` as the integer 450. The loader explains this in its error.
- A routine with a tool that is not yet installed is reported out loud and
  skipped rather than failing, which is what lets `routines/morning.yaml`
  ship before the weather, calendar and email tools do.
- Unsolicited speech is routed through the sink, so the orchestrator decides
  how a `Priority.LOW` notification behaves while the user is mid-sentence.
