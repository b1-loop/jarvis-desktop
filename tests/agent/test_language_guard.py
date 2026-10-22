from collections.abc import Callable

import pytest

from jarvis.agent.language_guard import LanguageGuard, contains_cjk
from jarvis.agent.persona import Persona, languages_of_voice
from jarvis.config import Settings
from jarvis.llm.base import SystemMessage
from tests.agent.fakes import EchoTool
from tests.agent.llm_fakes import (
    ScriptedLLM,
    SentenceCollector,
    call,
    make_agent,
    text_round,
    tool_round,
)

MakeSettings = Callable[..., Settings]

DRIFTED = "Var god och的具体任务：解释“See you in the sea.”这句英文的含义。"  # noqa: RUF001
"""qwen2.5:7b's real answer to a misheard line with a Swedish persona."""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (DRIFTED, True),
        ("今日は。", True),
        ("안녕하세요", True),
        ("Kan du vara mer specifik?", False),
        ("Hörlurar, smörgås och Ölands väder.", False),
        ("Привет, как дела?", False),
    ],
)
def test_cjk_detection(text: str, expected: bool) -> None:
    assert contains_cjk(text) is expected


def test_cjk_languages_may_speak_cjk() -> None:
    chinese = Persona(name="Jarvis", tone="calm", language="Chinese", native_language="中文")
    swedish = Persona(name="Jarvis", tone="calm", language="Swedish", native_language="svenska")

    assert not LanguageGuard(chinese).rejects("你好。")
    assert LanguageGuard(swedish).rejects("你好。")


def test_the_prompt_names_the_language_in_english_and_in_itself(
    make_settings: MakeSettings,
) -> None:
    prompt = Persona.from_settings(make_settings(tts_voice="sv-SE-MattiasNeural")).system_prompt()

    assert "Always reply in Swedish (svenska)" in prompt
    assert "never switch to Chinese" in prompt
    assert languages_of_voice("zh-CN-YunxiNeural") == ("Chinese", "中文")
    assert languages_of_voice("xx-YY-Unknown") == ("English", "English")


async def test_a_drifted_reply_is_never_spoken_and_is_regenerated_once(
    make_settings: MakeSettings,
) -> None:
    llm = ScriptedLLM(rounds=[text_round(DRIFTED), text_round("Vi ses i havet, sir.")])
    sink = SentenceCollector()

    report = await make_agent(make_settings(tts_voice="sv-SE-MattiasNeural"), llm).respond(
        "See you in the sea.", sink
    )

    assert sink.sentences == ["Vi ses i havet, sir."]
    assert report.llm_rounds == 2
    retry_system = llm.requests[1][0]
    assert isinstance(retry_system, SystemMessage)
    assert "only in Swedish (svenska)" in retry_system.text
    assert all(DRIFTED not in str(message) for message in llm.requests[1])


async def test_a_second_drift_is_dropped_without_another_retry(
    make_settings: MakeSettings,
) -> None:
    llm = ScriptedLLM(rounds=[text_round(DRIFTED), text_round("好的。")])
    sink = SentenceCollector()

    report = await make_agent(make_settings(tts_voice="sv-SE-MattiasNeural"), llm).respond(
        "?", sink
    )

    assert report.llm_rounds == 2
    assert sink.sentences  # the fallback, never the drifted text
    assert not any(contains_cjk(sentence) for sentence in sink.sentences)


async def test_sentences_spoken_before_the_drift_are_not_repeated(
    make_settings: MakeSettings,
) -> None:
    llm = ScriptedLLM(
        rounds=[
            text_round("Javisst, det kan jag berätta. ", "这是中文。"),
            text_round("Javisst, det kan jag berätta. ", "Det är soligt i dag."),
        ]
    )
    sink = SentenceCollector()

    await make_agent(make_settings(tts_voice="sv-SE-MattiasNeural"), llm).respond("Väder?", sink)

    assert sink.sentences == ["Javisst, det kan jag berätta.", "Det är soligt i dag."]


async def test_a_drifted_reply_with_tool_calls_keeps_its_calls(
    make_settings: MakeSettings,
) -> None:

    llm = ScriptedLLM(
        rounds=[
            tool_round(call("c1", "echo", text="hej"), text="我来查一下。"),
            text_round("Klart."),
        ]
    )
    sink = SentenceCollector()

    report = await make_agent(
        make_settings(tts_voice="sv-SE-MattiasNeural"), llm, EchoTool()
    ).respond("Eka hej", sink)

    assert [record.name for record in report.calls] == ["echo"]
    assert not any(contains_cjk(sentence) for sentence in sink.sentences)


async def test_a_chinese_persona_speaks_chinese(make_settings: MakeSettings) -> None:
    llm = ScriptedLLM(rounds=[text_round("你好。")])
    sink = SentenceCollector()

    report = await make_agent(make_settings(tts_voice="zh-CN-YunxiNeural"), llm).respond("hi", sink)

    assert sink.sentences == ["你好。"]
    assert report.llm_rounds == 1
