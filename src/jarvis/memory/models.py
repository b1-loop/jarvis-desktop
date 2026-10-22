"""Row types for the relational memory store.

Frozen dataclasses rather than pydantic models: rows come from our own
schema, not from untrusted input, so validation would only add overhead.
Every timestamp is timezone-aware UTC; the store normalises on the way in
so that comparisons and ordering never depend on the machine's local zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

JsonObject = dict[str, Any]
"""A JSON-serialisable mapping, e.g. one tool call as the LLM emitted it."""


class Role(StrEnum):
    """Who produced a message. Mirrors the roles the LLM API understands."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


def utc_now() -> datetime:
    """Current time as an aware UTC datetime; the default store clock."""
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Conversation:
    """One session of back-and-forth, from wake word (or first keystroke) to idle.

    Attributes:
        id: Row id.
        started_at: When the first message arrived.
        summary: Short "what was discussed" text, filled in when the
            conversation ends; empty until then.
    """

    id: int
    started_at: datetime
    summary: str = ""


@dataclass(frozen=True, slots=True)
class Message:
    """One utterance inside a conversation.

    Attributes:
        id: Row id.
        conversation_id: Owning conversation.
        role: Who said it.
        content: The text as spoken or generated.
        tool_calls: Tool invocations the assistant made in this turn, stored
            verbatim as JSON so the dashboard can replay them.
        created_at: When it was recorded.
    """

    id: int
    conversation_id: int
    role: Role
    content: str
    tool_calls: tuple[JsonObject, ...]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Note:
    """A dictated note.

    Attributes:
        id: Row id.
        title: Optional heading; empty when the user just said "note that...".
        content: Body text.
        created_at: When it was recorded.
    """

    id: int
    title: str
    content: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Reminder:
    """Something to say to the user at a given time.

    Attributes:
        id: Row id.
        text: What to say.
        due_at: When to say it.
        fired: True once the proactive layer has delivered it.
        created_at: When it was set.
    """

    id: int
    text: str
    due_at: datetime
    fired: bool
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Todo:
    """A todo item.

    Attributes:
        id: Row id.
        text: The task.
        done: True once ticked off.
        created_at: When it was added.
    """

    id: int
    text: str
    done: bool
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Fact:
    """A key/value statement about the user or their world.

    Attributes:
        key: Dotted, unique name such as ``profile.name`` or
            ``project.jarvis.language``.
        value: The statement.
        source: Where it came from: ``"user"`` when stated explicitly,
            ``"llm"`` when inferred, ``"tool"`` when observed.
        confidence: 0..1, so inferred facts can be ranked below stated ones.
        created_at: First time the key was written.
        updated_at: Last time the value changed.
    """

    key: str
    value: str
    source: str
    confidence: float
    created_at: datetime
    updated_at: datetime


# Design notes:
# - Facts are keyed by name rather than by row id because the agent reasons
#   in terms of "what do I know about X"; a second statement about the same
#   key is an update, not a new row. See ADR-0008.
# - tool_calls is a tuple of plain dicts, not a typed model, because the
#   exact shape belongs to the LLM provider; the store only
#   promises to give back what it was given.
