import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from jarvis.agent.chaining import TurnReport
from jarvis.config import Settings
from jarvis.memory import InMemoryVectorStore, MemoryManager, SqliteStore
from tests.agent.fakes import make_context

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_models.py"


def load_script() -> ModuleType:
    # scripts/ is not a package, so the module is loaded from its file path.
    spec = importlib.util.spec_from_file_location("compare_models", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("compare_models", module)
    spec.loader.exec_module(module)
    return module


compare = load_script()
MOMENT = datetime(2026, 10, 22, 14, 30, tzinfo=datetime.now().astimezone().tzinfo)


def test_numbers_accept_swedish_decimal_commas() -> None:
    assert compare.numbers_in("Du använder 7,9 GB av 31.7 GB (25 %).") == [7.9, 31.7, 25.0]


def test_percentages_and_memory_amounts_are_read_with_their_units() -> None:
    answer = "Du använder 20683 MB (65 %) och 2,5 GB swap, 70 procent totalt."

    assert compare.percents_in(answer) == [65.0, 70.0]
    assert compare.amounts_in_gb(answer) == [pytest.approx(20.2, abs=0.05), 2.5]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Du använder 20683 MB av 31865 MB, 64 procent.", True),
        ("Du använder 1694 MB för MemCompression av 20755 MB, 75%.", False),
        ("Du använder 20,1 GB.", True),
        ("Minnet ser bra ut.", False),
    ],
)
async def test_the_ram_check_trusts_only_the_first_figure(
    monkeypatch: pytest.MonkeyPatch, answer: str, expected: bool
) -> None:
    class Memory:
        percent = 65.0
        used = 20.3 * 1024**3

    monkeypatch.setattr(compare.psutil, "virtual_memory", Memory)
    probe = compare.Probe(answer=answer, memory=None, asked_at=MOMENT)

    assert await compare.check_ram(probe) is expected


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Klockan är 14:31.", True),
        ("Den är 14.29 just nu.", True),
        ("Klockan är 09:11.", False),
        ("Det är eftermiddag.", False),
    ],
)
def test_a_clock_time_must_be_close_to_now(answer: str, expected: bool) -> None:
    assert compare.mentions_time(answer, MOMENT) is expected


@pytest.mark.parametrize(
    ("spec", "parsed"),
    [
        ("qwen2.5:7b", ("qwen2.5:7b", False)),
        ("qwen3:8b@on", ("qwen3:8b", True)),
        ("qwen3:8b@default", ("qwen3:8b", None)),
    ],
)
def test_model_specs_carry_the_think_setting(spec: str, parsed: tuple[str, bool | None]) -> None:
    assert compare.parse_spec(spec) == parsed


def test_an_unknown_think_flag_is_refused() -> None:
    with pytest.raises(SystemExit):
        compare.parse_spec("qwen3:8b@maybe")


async def test_unsafe_tools_are_offered_but_refuse_to_run(
    make_settings: Any,
) -> None:
    registry = compare.sandboxed_registry()
    context = make_context(make_settings())
    lock = registry.get("lock_screen")

    result = await lock.run(lock.Input(), context)

    assert "lock_screen" in registry.names
    assert not result.ok
    assert type(registry.get("current_time")).__name__ == "CurrentTimeTool"


async def test_the_reminder_check_reads_the_store(tmp_path: Path) -> None:
    async with SqliteStore(tmp_path / "memory.sqlite3") as store:
        memory = MemoryManager(store, InMemoryVectorStore())
        await memory.add_reminder("tvätten", MOMENT + timedelta(minutes=5, seconds=20))
        probe = compare.Probe(answer="Okej.", memory=memory, asked_at=MOMENT)

        assert await compare.reminder_check(compare.LAUNDRY)(probe)
        assert not await compare.reminder_check(compare.MUM)(probe)  # wrong reminder
        assert not await compare.check_todo_added(probe)


async def test_the_relisting_case_changes_the_list_behind_the_models_back(
    tmp_path: Path,
) -> None:
    (case,) = [case for case in compare.CASES if case.label == "list todos again"]
    async with SqliteStore(tmp_path / "memory.sqlite3") as store:
        memory = MemoryManager(store, InMemoryVectorStore())
        await memory.add_todo("mjölk")
        assert case.setup is not None
        await case.setup(memory)

        stale = compare.Probe(answer="Mjölk.", memory=memory, asked_at=MOMENT)
        fresh = compare.Probe(answer="Mjölk och bröd.", memory=memory, asked_at=MOMENT)

        assert [todo.text for todo in await memory.list_todos()] == ["mjölk", "bröd"]
        assert not await case.check(stale)
        assert await case.check(fresh)


def test_the_report_has_a_row_per_request_and_totals(make_settings: Any) -> None:
    del make_settings
    (case,) = [case for case in compare.CASES if case.label == "calculation"]
    report = TurnReport(user_text=case.request, reply="Det blir 391.")
    outcome = compare.Outcome(case, report, True, 0.8, 1.5)
    run = compare.RunResult(label="qwen3:8b (think off)", outcomes=[outcome])

    table = compare.markdown([run])

    assert "| qwen3:8b (think off) | calculation | - | yes | yes | 0.8 | 1.5 |" in table
    assert "| qwen3:8b (think off) | 1/1 | 0.8 | 1.5 |" in table


def test_settings_use_the_swedish_voice_and_the_temp_dir(tmp_path: Path) -> None:
    args = compare.parse_args(["qwen3:8b"])

    settings: Settings = compare.comparison_settings(tmp_path, args, False)

    assert settings.data_dir == tmp_path
    assert settings.tts_voice == "sv-SE-MattiasNeural"
    assert settings.ollama_think is False


async def test_the_name_case_plants_a_misheard_name_and_wants_the_profile_one(
    tmp_path: Path,
) -> None:
    (case,) = [case for case in compare.CASES if case.label == "name from profile"]
    async with SqliteStore(tmp_path / "memory.sqlite3") as store:
        memory = MemoryManager(store, InMemoryVectorStore())
        assert case.setup is not None
        await case.setup(memory)

        context = await memory.recall(case.request)

        assert context.user_name == "Bozhi"
        assert "Ravel" in context.render()
        for answer, expected in (
            ("Du heter Bozhi.", True),
            ("Du heter Ravel.", False),
            ("Bozhi, eller Ravel?", False),
        ):
            probe = compare.Probe(answer=answer, memory=memory, asked_at=MOMENT)
            assert await case.check(probe) is expected


async def test_the_greeting_case_only_forbids_the_misheard_name() -> None:
    (case,) = [case for case in compare.CASES if case.label == "greeting by name"]

    def probe(answer: str) -> Any:
        return compare.Probe(answer=answer, memory=None, asked_at=MOMENT)

    assert await case.check(probe("Jag mår bra, tack!"))
    assert await case.check(probe("Hej Bozhi!"))
    assert not await case.check(probe("Hej Ravel!"))


def test_case_filters_keep_the_usual_order() -> None:
    args = compare.parse_args(["qwen3:8b", "--case", "calculation", "--case", "time"])

    assert [case.label for case in compare.selected_cases(args)] == ["time", "calculation"]
    assert compare.selected_cases(compare.parse_args(["qwen3:8b"])) == compare.CASES
