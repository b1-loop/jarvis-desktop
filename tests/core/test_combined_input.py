"""Typed lines and voice turns sharing one conversation."""

from __future__ import annotations

import asyncio

import pytest

from jarvis.core.combined_input import CombinedAnswers, CombinedInput


class FakeVoice:
    """A voice source whose next request the test releases by hand."""

    def __init__(self) -> None:
        self.reads = 0
        self.cancelled = 0
        self._next: asyncio.Future[str | None] | None = None

    async def read_line(self, prompt: str) -> str | None:
        del prompt
        self.reads += 1
        self._next = asyncio.get_running_loop().create_future()
        try:
            return await self._next
        except asyncio.CancelledError:
            self.cancelled += 1
            raise

    def say(self, text: str | None) -> None:
        assert self._next is not None
        self._next.set_result(text)

    async def listen_for_answer(self) -> str:
        self._next = asyncio.get_running_loop().create_future()
        try:
            return await self._next or ""
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


class FakeTurns:
    def __init__(self, *, running: bool = False) -> None:
        self.running = running
        self.cancels = 0

    @property
    def turn_in_progress(self) -> bool:
        return self.running

    def cancel_current_turn(self) -> bool:
        self.cancels += 1
        self.running = False
        return True


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def test_a_spoken_request_passes_through() -> None:
    voice = FakeVoice()
    combined = CombinedInput(voice)
    read = asyncio.create_task(combined.read_line("you> "))
    await settle()

    voice.say("what time is it")

    assert await read == "what time is it"


async def test_typed_while_idle_starts_a_turn_and_cancels_the_voice_wait() -> None:
    voice = FakeVoice()
    combined = CombinedInput(voice)
    combined.attach(FakeTurns(running=False))
    read = asyncio.create_task(combined.read_line("you> "))
    await settle()

    combined.submit("vad är klockan?")

    assert await read == "vad är klockan?"
    assert voice.cancelled == 1


async def test_typed_while_speaking_barges_in_and_silences_before_the_next_turn() -> None:
    voice = FakeVoice()
    silenced: list[int] = []
    turns = FakeTurns(running=True)
    combined = CombinedInput(voice, silence=lambda: silenced.append(len(silenced)))
    combined.attach(turns)

    combined.submit("stop, tell me the weather instead")

    assert turns.cancels == 1
    assert silenced == []  # not yet: the cancelled turn is still unwinding
    assert await combined.read_line("you> ") == "stop, tell me the weather instead"
    assert silenced == [0]


async def test_typed_lines_keep_their_order() -> None:
    combined = CombinedInput(FakeVoice())
    combined.submit("first")
    combined.submit("second")

    assert await combined.read_line("") == "first"
    assert await combined.read_line("") == "second"


async def test_a_voice_read_that_finished_first_wins_the_tie() -> None:
    voice = FakeVoice()
    combined = CombinedInput(voice)
    read = asyncio.create_task(combined.read_line(""))
    await settle()

    voice.say("spoken")
    combined.submit("typed")

    assert await read == "spoken"
    assert await combined.read_line("") == "typed"


async def test_the_pending_voice_read_is_reused_not_restarted() -> None:
    voice = FakeVoice()
    combined = CombinedInput(voice)
    first = asyncio.create_task(combined.read_line(""))
    await settle()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await combined.aclose()

    assert voice.cancelled == 1


async def test_typed_echo_is_written_like_a_spoken_line() -> None:
    lines: list[str] = []
    combined = CombinedInput(FakeVoice(), echo=lines.append)
    combined.submit("hej")

    await combined.read_line("")

    assert lines == ["you (typed)> hej\n"]


async def test_a_typed_line_answers_a_pending_confirmation() -> None:
    voice = FakeVoice()
    turns = FakeTurns(running=True)
    combined = CombinedInput(voice)
    combined.attach(turns)
    answers = CombinedAnswers(voice, combined)
    answer = asyncio.create_task(answers.listen_for_answer())
    await settle()

    combined.submit("yes")

    assert await answer == "yes"
    assert turns.cancels == 0  # an answer is not a barge-in
    assert voice.cancelled == 1  # the spoken wait was cancelled


async def test_a_spoken_answer_wins_when_it_comes_first() -> None:
    voice = FakeVoice()
    combined = CombinedInput(voice)
    answers = CombinedAnswers(voice, combined)
    answer = asyncio.create_task(answers.listen_for_answer())
    await settle()

    voice.say("no")

    assert await answer == "no"
    combined.submit("later request")
    assert await combined.read_line("") == "later request"
