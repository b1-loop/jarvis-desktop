"""The agent loop: one user utterance in, tool calls and a spoken answer out.

For each request the :class:`Agent` builds the conversation (persona system
prompt with recalled memory, the user's words, the registry's tool
schemas), streams the model's reply and forwards it sentence by sentence to
a sink so speech can start while the model is still writing. When the
model asks for tools, they run through :class:`~jarvis.agent.chaining.ToolChain`
(and therefore through the central confirmation policy), their results go
back to the model, and the loop repeats until the model stops or the
tool-round limit is reached. A turn that used tools always ends with a
spoken summary.

Two guards keep small local models honest. A :class:`ToolSelector` trims the
tool list each round to what the request needs (ADR-0020), and when every
tool call of the turn failed or was declined, the closing words come from
the tools' own results instead of the model, which has been seen to report
"Done" for a command the user had just refused.

The turn is plain ``async`` code with no shielding, so cancelling the task
that runs it (barge-in, step 7) stops streaming and tool execution at the
next await.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Iterable
from dataclasses import dataclass
from typing import Final, Protocol

from jarvis.agent.chaining import (
    DEFAULT_MAX_PARALLEL_TOOLS,
    DEFAULT_MAX_TOOL_CALLS,
    ToolChain,
    TurnReport,
)
from jarvis.agent.confirmation import ConfirmationPolicy
from jarvis.agent.persona import Persona
from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool import ToolContext
from jarvis.agent.tool_selection import RelevanceToolSelector, ToolSelector
from jarvis.core.events import AssistantSaid, LatencySample
from jarvis.llm.base import (
    AssistantMessage,
    Done,
    LLMClient,
    LLMError,
    LLMEvent,
    Message,
    SystemMessage,
    TextDelta,
    ToolDefinition,
    UserMessage,
)
from jarvis.llm.streaming import SentenceSplitter
from jarvis.memory.models import JsonObject
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger
from jarvis.utils.timing import Stopwatch

log = get_logger(__name__)

SentenceSink = Callable[[str], Awaitable[None]]
"""Receives each complete sentence as soon as it exists (TTS, or the terminal)."""

DEFAULT_MAX_TOOL_ROUNDS: Final = 6
"""Model/tool round trips per request; mirrors ``Settings.max_tool_rounds``."""

SUMMARY_REQUEST: Final = (
    "Now tell me in one short spoken sentence what you did. Do not call any more tools."
)
ROUND_LIMIT_DETAIL: Final = (
    "Not run: the limit of tool rounds for this request was reached. "
    "Summarise what was done and what is left."
)
APOLOGY: Final = "Sorry, I couldn't get an answer from the language model."
CLOUD_HINT: Final = (
    "If the cloud stays unreachable, set JARVIS_LLM_PROVIDER to ollama, "
    "or start me with --mode local, to keep working offline."
)
NOTHING_TO_SAY: Final = "Sorry, I don't have an answer to that."
NOTHING_DONE: Final = "I wasn't able to finish that."


class Recollection(Protocol):
    """What memory returns before a turn; only its prompt rendering is used."""

    def render(self) -> str:
        """Prompt-ready text, empty when nothing is known."""


class ConversationMemory(Protocol):
    """The slice of ``jarvis.memory.MemoryManager`` the loop needs."""

    async def recall(self, user_text: str) -> Recollection:
        """Context worth knowing before answering ``user_text``."""

    async def record_turn(
        self, user_text: str, assistant_text: str, tool_calls: Iterable[JsonObject] = ()
    ) -> object:
        """Persist one exchange."""


@dataclass(frozen=True, slots=True)
class AgentLimits:
    """Per-turn bounds that keep a confused model from running away.

    Attributes:
        max_tool_rounds: Model requests that may end in tool calls.
        max_tool_calls: Tool executions across all rounds.
        max_parallel_tools: Calls from one message run concurrently.
    """

    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_parallel_tools: int = DEFAULT_MAX_PARALLEL_TOOLS


class Agent:
    """Turns one user request into a spoken answer, using tools as needed.

    Args:
        llm: The model, behind the provider-neutral interface.
        registry: Tools the model may call.
        policy: Central confirmation policy; every call goes through it.
        persona: Name, tone and the system prompt.
        context: Services handed to tools, including how to ask the user.
        memory: Recall before and recording after each turn; optional so
            the agent also runs without persistence.
        limits: Tool-round, tool-call and parallelism bounds.
        cloud_provider: Whether ``llm`` is a hosted service, which decides
            if a failure apology suggests switching to local mode.
        selector: Which tools each request offers the model; defaults to
            ``RelevanceToolSelector.from_settings(context.settings)``.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        registry: ToolRegistry,
        policy: ConfirmationPolicy,
        persona: Persona,
        context: ToolContext,
        memory: ConversationMemory | None = None,
        limits: AgentLimits | None = None,
        cloud_provider: bool = False,
        selector: ToolSelector | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.policy = policy
        self.persona = persona
        self.context = context
        self.memory = memory
        self.limits = limits or AgentLimits()
        self.cloud_provider = cloud_provider
        self.selector = selector or RelevanceToolSelector.from_settings(context.settings)

    async def respond(self, user_text: str, sink: SentenceSink) -> TurnReport:
        """Answer ``user_text``, streaming sentences into ``sink``.

        Never raises for model or tool failures: an LLM failure is spoken
        as an apology and recorded in the report. ``CancelledError``
        propagates so the caller can abandon the turn.
        """
        return await _Turn(self, user_text, sink).run()

    def tool_definitions(self, request: str, *, keep: Collection[str] = ()) -> list[ToolDefinition]:
        """The tools offered for ``request`` in the provider-neutral shape.

        Args:
            request: The user's words, which the selector ranks tools against.
            keep: Tools already used this turn; they stay offered.
        """
        tools = self.selector.select(request, self.registry.tools(), keep=keep)
        return [
            ToolDefinition.from_model(tool.name, tool.description, tool.Input) for tool in tools
        ]


class _Turn:
    """State of one request; lives exactly as long as ``Agent.respond``."""

    def __init__(self, agent: Agent, user_text: str, sink: SentenceSink) -> None:
        self.agent = agent
        self.user_text = user_text
        self.sink = sink
        self.bus = agent.context.bus
        self.clock = agent.context.clock
        self.watch = Stopwatch(self.clock)
        self.chain = ToolChain(
            agent.registry,
            agent.policy,
            agent.context,
            clock=self.clock,
            max_calls=agent.limits.max_tool_calls,
            max_parallel=agent.limits.max_parallel_tools,
        )
        self.messages: list[Message] = []
        self.spoken: list[str] = []
        self.held: list[str] | None = None
        self.spoken_since_tools = False
        self.llm_rounds = 0
        self.round_limit_hit = False
        self.error: str | None = None

    async def run(self) -> TurnReport:
        try:
            await self._converse()
        except LLMError as error:
            await self._apologise(error)
        if not self.spoken or (self.chain.records and not self.spoken_since_tools):
            await self._emit(self._fallback_summary())
        report = self._report()
        await self._remember(report)
        self.bus.publish(AssistantSaid(text=report.reply))
        self.bus.publish(LatencySample(stage="agent.turn", ms=report.duration_ms))
        log.info("agent.turn.finished", tools=report.tools_used, rounds=report.llm_rounds)
        return report

    async def _converse(self) -> None:
        self.messages = [SystemMessage(await self._system_prompt()), UserMessage(self.user_text)]
        tool_rounds = 0
        while calls := (await self._stream_reply()).message.tool_calls:
            if tool_rounds >= self.agent.limits.max_tool_rounds:
                self.round_limit_hit = True
                log.warning("agent.round_limit", rounds=tool_rounds)
                decline = self.chain.decline(
                    calls, round_index=tool_rounds + 1, detail=ROUND_LIMIT_DETAIL
                )
                self.messages.append(decline)
                break
            tool_rounds += 1
            self.spoken_since_tools = False
            self.messages.append(await self.chain.run_round(calls, round_index=tool_rounds))
        if tool_rounds and not self.spoken_since_tools and not self._every_call_failed():
            await self._request_summary()

    async def _system_prompt(self) -> str:
        memory = self.agent.memory
        if memory is None:
            return self.agent.persona.system_prompt()
        try:
            recollection = await memory.recall(self.user_text)
        except JarvisError:
            log.exception("agent.memory.recall_failed")
            return self.agent.persona.system_prompt()
        return self.agent.persona.system_prompt(recollection.render())

    async def _stream_reply(self) -> Done:
        """One model request; sentences go to the sink as they complete."""
        self.llm_rounds += 1
        watch = Stopwatch(self.clock)
        splitter = SentenceSplitter()
        done: Done | None = None
        self.held = [] if self._every_call_failed() else None
        stream = self.agent.llm.stream(self.messages, self._offered_tools())
        try:
            async for event in stream:
                done = await self._handle(event, splitter) or done
        finally:
            await _close(stream)
        if (tail := splitter.flush()) is not None:
            await self._emit(tail)
        self.bus.publish(LatencySample(stage="agent.llm_round", ms=watch.elapsed_ms))
        if done is None:
            raise LLMError(f"{self.agent.llm.model} stream ended without a final event")
        self.messages.append(done.message)
        await self._release_held(done)
        return done

    def _offered_tools(self) -> list[ToolDefinition]:
        """This round's tools: re-selected so tools used earlier stay offered."""
        used = {record.name for record in self.chain.records}
        tools = self.agent.tool_definitions(self.user_text, keep=used)
        log.debug("agent.tools.offered", count=len(tools), tools=[tool.name for tool in tools])
        return tools

    def _every_call_failed(self) -> bool:
        """True once tools were called this turn and not one of them succeeded."""
        records = self.chain.records
        return bool(records) and not any(record.ok for record in records)

    async def _release_held(self, done: Done) -> None:
        """Speak or drop what the model said while every tool had failed.

        A reply that calls more tools is the model trying again, so its words
        are spoken. A final reply is replaced by the tools' own results in
        ``run``: after nothing succeeded, nothing the model adds is grounded.
        """
        held, self.held = self.held, None
        if not held:
            return
        if done.message.tool_calls:
            for sentence in held:
                await self._emit(sentence)
            return
        log.warning("agent.grounding.reply_replaced", dropped=" ".join(held))

    async def _handle(self, event: LLMEvent, splitter: SentenceSplitter) -> Done | None:
        if isinstance(event, TextDelta):
            for sentence in splitter.push(event.text):
                await self._emit(sentence)
        elif isinstance(event, Done):
            return event
        return None

    async def _request_summary(self) -> None:
        """Ask once for a one-sentence summary when tools ran and nothing was said after."""
        if _is_empty(self.messages[-1]):
            self.messages.pop()  # providers reject an empty assistant turn mid-history
        self.messages.append(UserMessage(SUMMARY_REQUEST))
        done = await self._stream_reply()
        if done.message.tool_calls:
            log.warning("agent.summary.ignored_tool_calls", count=len(done.message.tool_calls))

    async def _apologise(self, error: LLMError) -> None:
        self.held = None
        self.error = str(error)
        log.error("agent.llm_failed", error=self.error, cloud=self.agent.cloud_provider)
        await self._emit(APOLOGY)
        await self._emit(_as_sentence(self.error))
        if self.agent.cloud_provider:
            await self._emit(CLOUD_HINT)

    async def _emit(self, sentence: str) -> None:
        if self.held is not None:
            self.held.append(sentence)
            return
        if not self.spoken:
            self.bus.publish(LatencySample(stage="agent.first_sentence", ms=self.watch.elapsed_ms))
        self.spoken.append(sentence)
        self.spoken_since_tools = True
        await self.sink(sentence)

    def _fallback_summary(self) -> str:
        """Built from the tools' own spoken results when the model said nothing.

        Successes are reported when there are any; when every call failed,
        the failures' own sentences ("Okay, I won't do that.") are the summary.
        """
        records = self.chain.records
        succeeded = [record.spoken for record in records if record.ok and record.spoken]
        reported = succeeded or [record.spoken for record in records if record.spoken]
        if reported:
            return " ".join(dict.fromkeys(reported))
        return NOTHING_DONE if records else NOTHING_TO_SAY

    def _report(self) -> TurnReport:
        return TurnReport(
            user_text=self.user_text,
            reply=" ".join(self.spoken),
            calls=self.chain.records,
            llm_rounds=self.llm_rounds,
            duration_ms=self.watch.elapsed_ms,
            round_limit_hit=self.round_limit_hit,
            call_budget_hit=self.chain.call_budget_hit,
            error=self.error,
        )

    async def _remember(self, report: TurnReport) -> None:
        if self.agent.memory is None:
            return
        try:
            await self.agent.memory.record_turn(
                report.user_text, report.reply, report.memory_payload()
            )
        except JarvisError:
            log.exception("agent.memory.record_failed")


def _is_empty(message: Message) -> bool:
    return isinstance(message, AssistantMessage) and not message.text and not message.tool_calls


def _as_sentence(text: str) -> str:
    text = text.strip()
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


async def _close(stream: AsyncIterator[LLMEvent]) -> None:
    """Close an async-generator stream now, so a cancelled turn frees its connection."""
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        await aclose()


# Design notes:
# - Sentences, not tokens and not whole answers, leave the loop: a sentence
#   is the smallest unit TTS speaks naturally, and handing it over as soon
#   as it is complete bounds perceived latency by first-sentence time
#   (ADR-0005, ADR-0014). The same sink feeds the terminal in text mode.
# - Memory is recalled into the system prompt instead of replaying an
#   in-process history, so a new session continues where the last one
#   ended and the agent itself stays stateless between turns.
# - The tool-round limit answers any still-pending calls with an error
#   result before the summary request; every tool_use must be answered or
#   the provider rejects the next request. The summary request keeps the
#   tool definitions for the same reason (Anthropic refuses tool blocks in
#   a request without tools) and simply ignores calls made in it.
# - If the model ends with no text at all, the fallback summary is built
#   from the tools' own ``spoken`` sentences: "always end with a spoken
#   summary" holds even when the model misbehaves, without inventing
#   anything a tool did not report.
# - Grounding guard: once every call of the turn has failed or been
#   declined, the model's next reply is held back. If it calls tools again
#   it is spoken; if it is the final answer it is dropped and the turn ends
#   with the failed tools' own ``spoken`` sentences, deduplicated. Replacing
#   rather than checking the text is deliberate: deciding whether free text
#   "claims success" needs a classifier, and a 7B model's "Done. It printed:
#   ansel" after a declined command (a real run) shows the model cannot be
#   the judge. The cost is losing a correct paraphrase or a follow-up
#   question in that case; the tools' sentences are short, true and say
#   what did not happen. No summary request is made then, saving a round.
# - The tool list is re-selected every round with the tools already used
#   pinned, so a chain keeps its tools while each round stays short.
# - Memory failures are logged with a traceback and the turn continues: a
#   broken database must not make Jarvis mute (graceful degradation).
#   LLM failures become an apology that includes the provider's own remedy
#   text, plus the LOCAL-mode hint for cloud providers.
# - CancelledError is never caught here. ``_close`` runs in ``finally`` so
#   the HTTP stream is released immediately on cancellation; a cancelled
#   turn is not recorded in memory because it has no final answer.
