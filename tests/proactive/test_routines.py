import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from jarvis.proactive.routines import (
    ManualTrigger,
    PhraseTrigger,
    Routine,
    RoutineError,
    RoutineRegistry,
    RoutineRunner,
    Step,
    StepStatus,
    TimeTrigger,
    load_routine,
    load_routines,
)
from jarvis.proactive.scheduler import Scheduler
from tests.proactive.fakes import FakeClock, FakeInvoker, FakeSpeaker

REPO_ROUTINES = Path(__file__).resolve().parents[2] / "routines"
SOFIA = ZoneInfo("Europe/Sofia")

VALID = """
name: evening
trigger:
  kind: time
  at: "19:00"
  days: [mon, fri]
steps:
  - tool: weather
    speak: "Tomorrow: {result}"
  - tool: lights_off
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def routine(*steps: Step, on_error: str = "continue", **extra: object) -> Routine:
    return Routine.model_validate(
        {
            "name": "test",
            "trigger": {"kind": "manual"},
            "steps": [s.model_dump() for s in steps],
            "on_error": on_error,
            **extra,
        }
    )


# --- loading ----------------------------------------------------------------------


def test_shipped_morning_routine_is_valid() -> None:
    routines = load_routines(REPO_ROUTINES)

    assert [r.name for r in routines] == ["morning"]
    morning = routines[0]
    assert isinstance(morning.trigger, TimeTrigger)
    assert morning.trigger.at == time(7, 30)
    assert morning.trigger.weekdays == frozenset(range(5))
    assert [s.tool for s in morning.steps] == [
        "weather",
        "calendar_agenda",
        "email_unread",
        "news_headlines",
    ]


def test_valid_file_loads_with_defaults(tmp_path: Path) -> None:
    loaded = load_routine(write(tmp_path, "evening.yaml", VALID))

    assert loaded.name == "evening"
    assert isinstance(loaded.trigger, TimeTrigger)
    assert loaded.trigger.weekdays == frozenset({0, 4})
    assert loaded.steps[1].speak is None
    assert loaded.steps[1].timeout_seconds == 30.0
    assert loaded.on_error == "continue"
    assert loaded.policy_for(loaded.steps[0]) == "continue"


def test_unquoted_time_gets_a_helpful_error(tmp_path: Path) -> None:
    path = write(tmp_path, "r.yaml", VALID.replace('"19:00"', "19:00"))

    with pytest.raises(RoutineError) as excinfo:
        load_routine(path)

    assert excinfo.value.path == path
    assert any("trigger.time.at" in p and "quote the time" in p for p in excinfo.value.problems)


def test_unknown_field_is_reported_with_its_path(tmp_path: Path) -> None:
    path = write(tmp_path, "r.yaml", VALID.replace("  - tool: lights_off", "  - tol: lights_off"))

    with pytest.raises(RoutineError) as excinfo:
        load_routine(path)

    problems = "\n".join(excinfo.value.problems)
    assert "steps.1.tol: Extra inputs are not permitted" in problems
    assert "steps.1.tool: Field required" in problems


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("name: x\ntrigger: {kind: manual}\nsteps: []\n", "steps: "),
        ("name: x\ntrigger: {kind: time, at: '07:00', days: []}\nsteps: [{tool: a}]\n", "days"),
        (
            "name: x\ntrigger: {kind: time, at: '07:00', days: [funday]}\nsteps: [{tool: a}]\n",
            "days.0",
        ),
        ("name: x\ntrigger: {kind: hotkey}\nsteps: [{tool: a}]\n", "trigger"),
        (
            "name: x\ntrigger: {kind: manual}\nsteps: [{tool: a, speak: '{weather}'}]\n",
            "placeholder",
        ),
        ("name: x\ntrigger: {kind: manual}\nsteps: [{tool: a, speak: '{result'}]\n", "malformed"),
        ("name: x\ntrigger: {kind: manual}\nsteps: [{tool: a, timeout_seconds: 0}]\n", "timeout"),
        ("name: x\ntrigger: {kind: manual}\nsteps: [{tool: a, on_error: retry}]\n", "on_error"),
        ("name: x\ntrigger: {kind: phrase, phrase: ''}\nsteps: [{tool: a}]\n", "phrase"),
        ("- just\n- a list\n", "top level must be a mapping"),
    ],
)
def test_validation_errors_name_the_field(tmp_path: Path, text: str, expected: str) -> None:
    with pytest.raises(RoutineError) as excinfo:
        load_routine(write(tmp_path, "r.yaml", text))

    assert expected in str(excinfo.value)


def test_yaml_syntax_error_reports_line_and_column(tmp_path: Path) -> None:
    path = write(tmp_path, "r.yaml", "name: x\nsteps:\n  - tool: a\n   bad: [\n")

    with pytest.raises(RoutineError, match=r"YAML syntax: .*line \d+, column \d+"):
        load_routine(path)


def test_missing_file_is_a_routine_error(tmp_path: Path) -> None:
    with pytest.raises(RoutineError, match="cannot read file"):
        load_routine(tmp_path / "nope.yaml")


def test_missing_directory_yields_no_routines(tmp_path: Path) -> None:
    assert load_routines(tmp_path / "absent") == []


def test_duplicate_names_across_files_are_rejected(tmp_path: Path) -> None:
    write(tmp_path, "a.yaml", VALID)
    write(tmp_path, "b.yml", VALID)
    write(tmp_path, "notes.txt", "ignored")

    with pytest.raises(RoutineError, match=r"already used by a\.yaml"):
        load_routines(tmp_path)


def test_phrase_is_normalised() -> None:
    trigger = PhraseTrigger.model_validate({"kind": "phrase", "phrase": "  Good Morning, Jarvis! "})
    assert trigger.phrase == "good morning jarvis"


# --- running ------------------------------------------------------------------------


@pytest.fixture
def speaker() -> FakeSpeaker:
    return FakeSpeaker()


async def test_happy_path_speaks_intro_results_and_outro(speaker: FakeSpeaker) -> None:
    invoker = FakeInvoker({"weather": "sunny, 18 degrees", "lights": "ok"})
    plan = routine(
        Step(tool="weather", args={"location": "home"}, speak="Weather: {result}"),
        Step(tool="lights", args={"state": "off"}),
        intro="Good evening.",
        outro="Done.",
    )

    report = await RoutineRunner(invoker, speaker).run(plan)

    assert report.ok
    assert [s.status for s in report.steps] == [StepStatus.OK, StepStatus.OK]
    assert report.steps[0].output == "sunny, 18 degrees"
    assert invoker.calls == [("weather", {"location": "home"}), ("lights", {"state": "off"})]
    assert speaker.said == ["Good evening.", "Weather: sunny, 18 degrees", "Done."]


async def test_unknown_tool_is_reported_and_skipped(speaker: FakeSpeaker) -> None:
    invoker = FakeInvoker({"after": "fine"})
    plan = routine(Step(tool="teleport", speak="{result}"), Step(tool="after", speak="{result}"))

    report = await RoutineRunner(invoker, speaker).run(plan)

    assert not report.ok
    assert report.steps[0].status is StepStatus.UNKNOWN_TOOL
    assert report.steps[0].error == "unknown tool 'teleport'"
    assert report.steps[1].ok
    assert speaker.said == ["I don't have a tool called teleport, skipping it.", "fine"]


async def test_step_timeout_is_reported_and_routine_continues(speaker: FakeSpeaker) -> None:
    invoker = FakeInvoker({"slow": "never", "fast": "quick"}, slow={"slow"})
    plan = routine(Step(tool="slow", timeout_seconds=0.01), Step(tool="fast", speak="{result}"))

    report = await RoutineRunner(invoker, speaker).run(plan)

    assert report.steps[0].status is StepStatus.TIMED_OUT
    assert report.steps[1].ok
    assert speaker.said == ["slow took too long, skipping it.", "quick"]


async def test_tool_exception_is_logged_and_reported(speaker: FakeSpeaker) -> None:
    invoker = FakeInvoker({"flaky": ConnectionError("offline")})
    plan = routine(Step(tool="flaky", speak="{result}"))

    report = await RoutineRunner(invoker, speaker).run(plan)

    assert report.steps[0].status is StepStatus.FAILED
    assert report.steps[0].error == "ConnectionError('offline')"
    assert speaker.said == ["flaky failed, skipping it."]


async def test_abort_policy_stops_after_first_failure(speaker: FakeSpeaker) -> None:
    invoker = FakeInvoker({"second": "x"})
    plan = routine(Step(tool="missing"), Step(tool="second"), on_error="abort", outro="Done.")

    report = await RoutineRunner(invoker, speaker).run(plan)

    assert report.aborted
    assert [s.tool for s in report.steps] == ["missing"]
    assert invoker.calls == [("missing", {})]
    assert "Done." not in speaker.said


async def test_step_policy_overrides_routine_policy(speaker: FakeSpeaker) -> None:
    invoker = FakeInvoker({"b": "x"})
    plan = routine(Step(tool="a", on_error="abort"), Step(tool="b"), on_error="continue")

    report = await RoutineRunner(invoker, speaker).run(plan)

    assert report.aborted
    assert len(report.steps) == 1


async def test_routine_budget_cuts_a_slow_routine_short(speaker: FakeSpeaker) -> None:
    invoker = FakeInvoker({"slow": "never"}, slow={"slow"})
    plan = routine(
        Step(tool="slow", timeout_seconds=10),
        max_duration_seconds=0.02,
        outro="Done.",
    )

    report = await RoutineRunner(invoker, speaker).run(plan)

    assert report.timed_out
    assert not report.ok
    assert speaker.said == ["test ran out of time."]


async def test_cancellation_propagates(speaker: FakeSpeaker) -> None:
    invoker = FakeInvoker({"slow": "never"}, slow={"slow"})
    plan = routine(Step(tool="slow", timeout_seconds=10))
    task = asyncio.create_task(RoutineRunner(invoker, speaker).run(plan))
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# --- registry -------------------------------------------------------------------------


def phrase_routine(name: str, phrase: str) -> Routine:
    return Routine.model_validate(
        {"name": name, "trigger": {"kind": "phrase", "phrase": phrase}, "steps": [{"tool": "x"}]}
    )


def test_registry_matches_phrases_on_whole_words() -> None:
    registry = RoutineRegistry([phrase_routine("morning", "good morning"), routine(Step(tool="x"))])

    assert registry.by_phrase("Hey, good morning Jarvis!") is registry.by_name("morning")
    assert registry.by_phrase("goodnight") is None
    assert registry.by_phrase("that was a good morningstar") is None
    assert registry.by_name("missing") is None
    assert isinstance(registry.by_name("test"), Routine)
    assert isinstance(registry.routines[1].trigger, ManualTrigger)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(datetime(2026, 10, 21, 7, 0, tzinfo=SOFIA))


@pytest.fixture
async def scheduler(clock: FakeClock) -> AsyncIterator[Scheduler]:
    async with Scheduler(clock) as scheduler:
        await clock.settle()
        yield scheduler


async def test_registry_schedules_timed_routines_only(
    scheduler: Scheduler, clock: FakeClock, speaker: FakeSpeaker
) -> None:
    timed = Routine.model_validate(
        {
            "name": "morning",
            "trigger": {"kind": "time", "at": "07:30", "days": ["wed"]},
            "steps": [{"tool": "weather", "speak": "{result}"}],
        }
    )
    registry = RoutineRegistry([timed, phrase_routine("hello", "hello"), routine(Step(tool="x"))])
    runner = RoutineRunner(FakeInvoker({"weather": "fog"}), speaker)

    job_ids = registry.schedule(scheduler, runner, tz=SOFIA)

    assert list(job_ids) == ["morning"]
    job = scheduler.job(job_ids["morning"])
    assert job is not None
    assert job.next_run == datetime(2026, 10, 21, 7, 30, tzinfo=SOFIA)

    await clock.advance(timedelta(minutes=30))
    assert speaker.said == ["fog"]
