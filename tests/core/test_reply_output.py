"""The "speak replies" switch."""

from __future__ import annotations

from jarvis.core.reply_output import SwitchableOutput, TextReplies


class RecordingSink:
    def __init__(self, name: str) -> None:
        self.name = name
        self.said: list[str] = []
        self.notices: list[str] = []
        self.flushes = 0

    async def say(self, sentence: str) -> None:
        self.said.append(sentence)

    async def notice(self, text: str) -> None:
        self.notices.append(text)

    async def flush(self) -> bool:
        self.flushes += 1
        return True


async def test_answers_are_spoken_while_the_switch_is_on() -> None:
    speaking, silent = RecordingSink("voice"), RecordingSink("text")
    output = SwitchableOutput(speaking, silent, speak=True)

    await output.say("Hello.")
    assert await output.flush()

    assert speaking.said == ["Hello."]
    assert silent.said == []


async def test_turning_it_off_mid_answer_keeps_the_answer_in_one_sink() -> None:
    speaking, silent = RecordingSink("voice"), RecordingSink("text")
    output = SwitchableOutput(speaking, silent, speak=True)

    await output.say("First sentence.")
    output.set_speak(False)
    await output.say("Second sentence.")
    await output.flush()
    await output.say("Next answer.")
    await output.flush()

    assert speaking.said == ["First sentence.", "Second sentence."]
    assert silent.said == ["Next answer."]
    assert speaking.flushes == 1 and silent.flushes == 1


async def test_notices_are_never_spoken() -> None:
    speaking, silent = RecordingSink("voice"), RecordingSink("text")
    output = SwitchableOutput(speaking, silent)

    await output.notice("Saved.")

    assert silent.notices == ["Saved."]
    assert speaking.notices == []


async def test_text_replies_print_and_show_in_the_overlay() -> None:
    terminal = RecordingSink("terminal")
    shown: list[str] = []
    replies = TextReplies(terminal, shown.append)

    await replies.say("It is ten.")
    await replies.notice("note")

    assert terminal.said == ["It is ten."]
    assert shown == ["It is ten."]
    assert terminal.notices == ["note"]
    assert await replies.flush()
