from datetime import datetime, timedelta, timezone

from jarvis.agent.persona import Persona, clock_section

PERSONA = Persona(name="Jarvis", tone="calm", language="Swedish", native_language="svenska")
STOCKHOLM_SUMMER = timezone(timedelta(hours=2), "CEST")
NOW = datetime(2026, 10, 22, 16, 43, 31, tzinfo=STOCKHOLM_SUMMER)


def test_the_clock_section_states_the_moment_to_the_minute() -> None:
    text = clock_section(NOW)

    assert "Thursday 2026-10-22, 16:43 local time (UTC+02:00, CEST)" in text
    assert "16:43:31" not in text  # minutes only, so the prompt changes once a minute


def test_a_naive_moment_leaves_the_offset_blank_instead_of_guessing() -> None:
    text = clock_section(datetime(2026, 10, 22, 9, 5))

    assert "09:05 local time (zone unknown)" in text


def test_the_prompt_carries_the_clock_before_the_memory() -> None:
    prompt = PERSONA.system_prompt("User name: Bozhidar", now=NOW)

    assert prompt.index("Right now it is Thursday") < prompt.index("User name: Bozhidar")
    assert "never as the answer to a question about live data" in prompt


def test_without_a_moment_there_is_no_clock_section() -> None:
    assert "Right now it is" not in PERSONA.system_prompt()


def test_live_data_must_come_from_a_tool_call_in_this_turn() -> None:
    prompt = PERSONA.system_prompt()

    assert "call the tool in this turn" in prompt
    for kind in ("todos", "reminders", "calendar", "email", "battery"):
        assert kind in prompt
