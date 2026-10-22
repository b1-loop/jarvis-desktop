"""The single memory entry point for the agent.

The agent should never talk to SQLite or ChromaDB directly. It records a
turn, asks for context before answering, remembers facts, and manages
notes, todos and reminders through ``MemoryManager``. The manager decides
what goes where: every turn and note is written to SQLite (truth) and
embedded into the vector store (recall); facts live in SQLite only.

``recall`` returns a :class:`RecalledContext` trimmed to a character
budget so that memory can never crowd the user's actual question out of
the prompt.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final

from jarvis.memory.models import Conversation, Fact, JsonObject, Message, Note, Reminder, Role, Todo
from jarvis.memory.sqlite_store import SqliteStore
from jarvis.memory.user_profile import PROFILE_PREFIX, UserProfile
from jarvis.memory.vector_store import VectorHit, VectorStore
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

Summariser = Callable[[Sequence[Message]], Awaitable[str]]
"""Turns a conversation into a short "what was discussed" text.

The composition root does not wire an LLM-backed summariser (it would cost
a model call at every shutdown), so ``truncating_summariser``, which keeps
the first words of the exchange, is what runs today.
"""

DEFAULT_CONTEXT_BUDGET_CHARS: Final = 4_000
DEFAULT_SUMMARY_CHARS: Final = 300
_TURN_KIND: Final = "turn"
_NOTE_KIND: Final = "note"
_ELLIPSIS: Final = "…"


async def truncating_summariser(messages: Sequence[Message]) -> str:
    """Placeholder summariser: the conversation's text cut to a fixed length."""
    text = " ".join(f"{message.role.value}: {message.content}" for message in messages)
    return _truncate(text, DEFAULT_SUMMARY_CHARS)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(limit - len(_ELLIPSIS), 0)] + _ELLIPSIS


def _render_recent(message: Message) -> str:
    """One line of the recent conversation as the model should read it.

    An assistant answer that came from tools (the time, RAM in use, unread
    mail) is stale by the next turn. Showing its text tempted small models
    to repeat it instead of calling the tool again, so only the fact that a
    tool answered, and when, is kept.
    """
    stamp = message.created_at.astimezone().strftime("%H:%M")
    if message.role is Role.ASSISTANT and message.tool_calls:
        tools = ", ".join(sorted({str(call.get("name", "tool")) for call in message.tool_calls}))
        return (
            f"[{stamp}] {message.role.value}: (answered using {tools}; that result is "
            "out of date now, call the tool again for current values)"
        )
    return f"[{stamp}] {message.role.value}: {message.content}"


@dataclass(frozen=True, slots=True)
class RecalledContext:
    """What the agent should know before answering one user message.

    Attributes:
        profile: Rendered user profile; empty when nothing is known.
        facts: Non-profile facts, most recently updated first.
        recent: The latest messages in chronological order.
        semantic: Older turns and notes similar to the new message.
    """

    profile: str
    facts: tuple[Fact, ...]
    recent: tuple[Message, ...]
    semantic: tuple[VectorHit, ...]

    def render(self) -> str:
        """Prompt-ready text with one section per non-empty part."""
        sections: list[str] = []
        if self.profile:
            sections.append(self.profile)
        if self.facts:
            facts = "\n".join(f"- {fact.key}: {fact.value}" for fact in self.facts)
            sections.append(f"Known facts:\n{facts}")
        if self.semantic:
            hits = "\n".join(f"- {hit.text}" for hit in self.semantic)
            sections.append(f"Related memories:\n{hits}")
        if self.recent:
            turns = "\n".join(_render_recent(message) for message in self.recent)
            sections.append(f"Recent conversation:\n{turns}")
        return "\n\n".join(sections)

    def _shrunk(self) -> RecalledContext | None:
        """Drop the least valuable remaining item, or ``None`` if nothing is left."""
        if self.semantic:
            return replace(self, semantic=self.semantic[:-1])
        if self.facts:
            return replace(self, facts=self.facts[:-1])
        if self.recent:
            return replace(self, recent=self.recent[1:])
        if self.profile:
            return replace(self, profile="")
        return None

    def fit(self, budget_chars: int) -> RecalledContext:
        """Shrink until ``render()`` fits in ``budget_chars``.

        Items go in order of least value: semantic hits, then loose facts,
        then the oldest recent messages, and the profile last.
        """
        context = self
        while len(context.render()) > budget_chars:
            smaller = context._shrunk()
            if smaller is None:
                break
            context = smaller
        return context


class MemoryManager:
    """Coordinates the relational and semantic stores for the agent.

    Args:
        store: Open relational store (truth).
        vectors: Vector store (recall).
        recent_turns: How many user/assistant exchanges count as "recent".
        semantic_hits: How many similar memories to recall.
        context_budget_chars: Upper bound on ``RecalledContext.render()``.
        summariser: Hook for the end-of-conversation summary.
    """

    def __init__(
        self,
        store: SqliteStore,
        vectors: VectorStore,
        *,
        recent_turns: int = 10,
        semantic_hits: int = 5,
        context_budget_chars: int = DEFAULT_CONTEXT_BUDGET_CHARS,
        summariser: Summariser = truncating_summariser,
    ) -> None:
        self._store = store
        self._vectors = vectors
        self._recent_turns = recent_turns
        self._semantic_hits = semantic_hits
        self._budget = context_budget_chars
        self._summarise = summariser
        self._conversation: Conversation | None = None

    @property
    def conversation(self) -> Conversation | None:
        """The conversation turns are currently recorded into, if any."""
        return self._conversation

    # -- conversations ------------------------------------------------------

    async def start_conversation(self) -> Conversation:
        """Begin a new conversation; ends the current one first if needed."""
        if self._conversation is not None:
            await self.end_conversation()
        self._conversation = await self._store.start_conversation()
        log.info("memory.conversation.started", conversation_id=self._conversation.id)
        return self._conversation

    async def end_conversation(self) -> str:
        """Summarise and close the current conversation.

        Returns:
            The stored summary; empty if no conversation was open.
        """
        if self._conversation is None:
            return ""
        conversation_id = self._conversation.id
        messages = await self._store.recent_messages(
            2 * self._recent_turns, conversation_id=conversation_id
        )
        summary = await self._summarise(messages) if messages else ""
        await self._store.set_summary(conversation_id, summary)
        self._conversation = None
        log.info("memory.conversation.ended", conversation_id=conversation_id)
        return summary

    async def record_turn(
        self, user_text: str, assistant_text: str, tool_calls: Iterable[JsonObject] = ()
    ) -> tuple[Message, Message]:
        """Persist one exchange and embed it for later recall.

        A conversation is started implicitly if none is open.
        """
        conversation = self._conversation or await self.start_conversation()
        user = await self._store.add_message(conversation.id, Role.USER, user_text)
        assistant = await self._store.add_message(
            conversation.id, Role.ASSISTANT, assistant_text, tool_calls
        )
        await self._vectors.add(
            f"turn:{assistant.id}",
            f"User: {user_text}\nJarvis: {assistant_text}",
            {"kind": _TURN_KIND, "conversation_id": conversation.id, "message_id": assistant.id},
        )
        log.debug("memory.turn.recorded", conversation_id=conversation.id, message_id=assistant.id)
        return user, assistant

    # -- recall -----------------------------------------------------------------

    async def recall(self, user_text: str) -> RecalledContext:
        """Everything worth knowing before answering ``user_text``, within budget."""
        profile = await UserProfile.load(self._store)
        facts = [f for f in await self._store.list_facts() if not f.key.startswith(PROFILE_PREFIX)]
        facts.sort(key=lambda fact: fact.updated_at, reverse=True)
        recent = await self._store.recent_messages(2 * self._recent_turns)
        semantic = await self._semantic_hits_for(user_text, exclude={m.id for m in recent})
        context = RecalledContext(
            profile=profile.render(),
            facts=tuple(facts),
            recent=tuple(recent),
            semantic=tuple(semantic),
        ).fit(self._budget)
        log.debug(
            "memory.recalled",
            recent=len(context.recent),
            semantic=len(context.semantic),
            facts=len(context.facts),
            chars=len(context.render()),
        )
        return context

    async def _semantic_hits_for(self, text: str, *, exclude: set[int]) -> list[VectorHit]:
        if self._semantic_hits <= 0:
            return []
        hits = await self._vectors.query(text, k=self._semantic_hits + len(exclude))
        fresh = [hit for hit in hits if hit.metadata.get("message_id") not in exclude]
        return fresh[: self._semantic_hits]

    # -- facts ------------------------------------------------------------------

    async def remember(
        self, key: str, value: str, *, source: str = "user", confidence: float = 1.0
    ) -> Fact:
        """Store or update a fact."""
        fact = await self._store.upsert_fact(key, value, source=source, confidence=confidence)
        log.info("memory.fact.remembered", key=key, source=source)
        return fact

    async def forget(self, key: str) -> bool:
        """Delete a fact; returns whether it existed."""
        return await self._store.delete_fact(key)

    async def profile(self) -> UserProfile:
        """The current user profile."""
        return await UserProfile.load(self._store)

    # -- notes ------------------------------------------------------------------

    async def add_note(self, content: str, title: str = "") -> Note:
        """Store a note and embed it for recall."""
        note = await self._store.add_note(content, title)
        text = f"{title}\n{content}" if title else content
        await self._vectors.add(f"note:{note.id}", text, {"kind": _NOTE_KIND, "note_id": note.id})
        return note

    async def search_notes(self, query: str, limit: int = 10) -> list[Note]:
        """Full-text search over notes."""
        return await self._store.search_notes(query, limit)

    async def delete_note(self, note_id: int) -> None:
        """Remove a note from both stores."""
        await self._store.delete_note(note_id)
        await self._vectors.delete(f"note:{note_id}")

    # -- todos ------------------------------------------------------------------

    async def add_todo(self, text: str) -> Todo:
        """Add an open todo."""
        return await self._store.add_todo(text)

    async def list_todos(self, *, include_done: bool = False) -> list[Todo]:
        """Open todos, or all of them."""
        return await self._store.list_todos(include_done=include_done)

    async def complete_todo(self, todo_id: int) -> Todo:
        """Tick a todo off."""
        return await self._store.complete_todo(todo_id)

    # -- reminders --------------------------------------------------------------

    async def add_reminder(self, text: str, due_at: datetime) -> Reminder:
        """Schedule a reminder."""
        return await self._store.add_reminder(text, due_at)

    async def list_reminders(self, *, include_fired: bool = False) -> list[Reminder]:
        """Pending reminders by due time, or all of them."""
        return await self._store.list_reminders(include_fired=include_fired)

    async def pop_due_reminders(self, now: datetime | None = None) -> list[Reminder]:
        """Reminders that are due; each is returned exactly once."""
        return await self._store.pop_due_reminders(now)


# Design notes:
# - Recent messages span conversations rather than only the current one so
#   that a fresh session still starts with "what we talked about last time",
#   which is the behaviour the Phase 1 feature list promises.
# - Semantic hits that are already in the recent window are filtered out by
#   message id; the query asks for extra results to compensate so recall
#   still returns ``semantic_hits`` useful items.
# - Budget is measured in characters, not tokens, because the tokenizer
#   depends on the LLM provider (Claude vs llama.cpp). 4 characters per
#   token is a safe approximation for English; the agent passes a budget
#   derived from its model's context window.
# - The summariser is a plain async callable rather than a Protocol class
#   because it has exactly one operation and the LLM version will close
#   over an LLMClient.
# - MemoryManager depends on the concrete SqliteStore within its own
#   package; the Protocol boundary that matters for the rest of Jarvis is
#   MemoryManager itself (the agent only sees it) and VectorStore (optional
#   dependency).
