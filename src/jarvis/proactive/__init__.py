"""Scheduler, triggers and YAML-configured routines.

Lets Jarvis speak first: upcoming meetings, low battery, finished long
commands and the morning briefing all originate here. The package depends
only on small Protocols (``Clock``, ``NotificationSink``, ``ToolInvoker``,
``Speaker`` and the trigger sources) that the agent, orchestrator and
integrations implement; see ADR-0007.
"""

from jarvis.proactive.routines import (
    Routine,
    RoutineError,
    RoutineRegistry,
    RoutineReport,
    RoutineRunner,
    Speaker,
    ToolInvoker,
    UnknownToolError,
    load_routine,
    load_routines,
)
from jarvis.proactive.scheduler import Clock, Scheduler, SchedulerError, SystemClock
from jarvis.proactive.triggers import (
    Notification,
    NotificationSink,
    Priority,
    Trigger,
    TriggerRunner,
)

__all__ = [
    "Clock",
    "Notification",
    "NotificationSink",
    "Priority",
    "Routine",
    "RoutineError",
    "RoutineRegistry",
    "RoutineReport",
    "RoutineRunner",
    "Scheduler",
    "SchedulerError",
    "Speaker",
    "SystemClock",
    "ToolInvoker",
    "Trigger",
    "TriggerRunner",
    "UnknownToolError",
    "load_routine",
    "load_routines",
]
