"""Prompt-level tool calling for models without a native tool API.

Our llama.cpp wrapper generates plain text; it has no structured tool-use
channel like Claude's or Ollama's. Small local models can still call tools
if the protocol is spelled out in the system prompt and the reply is parsed
defensively:

1. :func:`render_tool_instructions` lists every tool with its JSON schema
   and asks for exactly one JSON object ``{"tool": name, "arguments": {...}}``
   when a tool is wanted, plain sentences otherwise.
2. :class:`ToolCallParser` classifies the streamed reply *while it streams*:
   ordinary text is released immediately (so TTS can start), a reply that
   opens with ``{`` (optionally after a Markdown fence or a ``<tool_call>``
   tag, which small models emit out of habit) is held back until the object
   closes, then parsed and validated against the tool's schema.
3. Anything that does not validate falls back to text: the user hears what
   the model wrote instead of Jarvis failing the turn.

The parsed object is only ever *data*: a tool name looked up in the offered
tools and an argument dict that the tool validates again with pydantic.
Nothing from the model is evaluated or executed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Final

from jarvis.llm.base import ToolDefinition
from jarvis.utils.logging import get_logger

TOOL_KEY: Final = "tool"
NAME_KEY: Final = "name"
"""Accepted alias for ``tool``: Qwen and OpenAI-trained models default to it."""
ARGUMENTS_KEY: Final = "arguments"

OPENERS: Final[tuple[str, ...]] = ("```json", "```", "<tool_call>")
"""Wrappers tolerated before the JSON object; longest first so ``json`` is consumed."""

MAX_TOOL_CALL_CHARS: Final = 4_000
"""A held-back candidate longer than this is treated as text: no tool call
offered to Jarvis needs arguments this large, and holding more would delay
speech for a reply that is almost certainly prose or data."""

_JSON_TYPES: Final[Mapping[str, tuple[type, ...]]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}

log = get_logger(__name__)


def render_tool_instructions(tools: Sequence[ToolDefinition]) -> str:
    """The system-prompt section that teaches the model the protocol."""
    listing = "\n".join(_describe_tool(tool) for tool in tools)
    return (
        "You can use tools. To use one, reply with ONLY a single JSON object and "
        "nothing else, in exactly this form:\n"
        f'{{"{TOOL_KEY}": "<tool name>", "{ARGUMENTS_KEY}": {{<arguments>}}}}\n'
        "The arguments must match the tool's JSON schema. Only use a tool when the "
        "request cannot be answered without it, and at most one tool per reply. You "
        "will then receive the tool's result and can answer or use another tool. "
        "Otherwise answer in plain sentences with no JSON.\n\n"
        f"Available tools:\n{listing}"
    )


def render_tool_call(name: str, arguments: Mapping[str, Any]) -> str:
    """A past call in protocol form, so replayed history shows the model its own format."""
    return json.dumps({TOOL_KEY: name, ARGUMENTS_KEY: dict(arguments)}, ensure_ascii=False)


def _describe_tool(tool: ToolDefinition) -> str:
    schema = json.dumps(_without_titles(tool.input_schema), separators=(",", ":"))
    return f"- {tool.name}: {tool.description}\n  arguments schema: {schema}"


def _without_titles(value: Any) -> Any:
    # pydantic adds a "title" to every property; it costs prompt tokens a
    # small context window cannot spare and tells the model nothing new.
    if isinstance(value, Mapping):
        return {key: _without_titles(item) for key, item in value.items() if key != "title"}
    if isinstance(value, list):
        return [_without_titles(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    """A validated request to run ``name`` with ``arguments``."""

    name: str
    arguments: dict[str, Any]


class _Phase(Enum):
    UNDECIDED = auto()
    CANDIDATE = auto()
    TEXT = auto()
    CALL = auto()


class ToolCallParser:
    """Streaming classifier for one reply: plain text or one JSON tool call.

    Feed every generated piece to :meth:`push` and emit what it returns as
    text. Once :attr:`call` is set the reply is a tool call and generation
    can be stopped. :meth:`finish` returns any text still held back when
    the stream ends (an unfinished or invalid candidate).
    """

    def __init__(self, tools: Sequence[ToolDefinition]) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._phase = _Phase.UNDECIDED
        self._held = ""
        self._scanner = JsonObjectScanner()
        self._call: ParsedToolCall | None = None

    @property
    def call(self) -> ParsedToolCall | None:
        """The validated tool call, once the JSON object has closed."""
        return self._call

    def push(self, piece: str) -> str:
        """Add ``piece``; return the text that is safe to emit now (maybe empty)."""
        if self._phase is _Phase.TEXT:
            return piece
        if self._phase is _Phase.CALL:
            return ""
        self._held += piece
        if self._phase is _Phase.UNDECIDED:
            return self._classify()
        return self._scan(piece)

    def finish(self) -> str:
        """End of stream: release any held-back text (fallback for a broken call)."""
        if self._phase in (_Phase.UNDECIDED, _Phase.CANDIDATE) and self._held.strip():
            log.warning("llm.local.tool_call_unfinished", chars=len(self._held))
            return self._release()
        return ""

    def _classify(self) -> str:
        start = json_start(self._held)
        if start is None:
            return ""
        if start < 0:
            return self._release()
        self._phase = _Phase.CANDIDATE
        return self._scan(self._held[start:])

    def _scan(self, fresh: str) -> str:
        end = self._scanner.feed(fresh)
        if end is None:
            return self._release() if len(self._held) > MAX_TOOL_CALL_CHARS else ""
        consumed = len(self._held) - len(fresh) + end
        candidate = self._held[self._held.index("{") : consumed]
        self._call = self._validate(candidate)
        if self._call is None:
            return self._release()
        self._phase = _Phase.CALL
        return ""

    def _validate(self, candidate: str) -> ParsedToolCall | None:
        try:
            return parse_tool_call(candidate, self._tools)
        except ToolCallRejectedError as error:
            log.warning("llm.local.tool_call_rejected", reason=str(error))
            return None

    def _release(self) -> str:
        text, self._held = self._held, ""
        self._phase = _Phase.TEXT
        return text


def json_start(text: str) -> int | None:
    """Where the JSON object in ``text`` starts, if the reply opens with one.

    Returns:
        The index of ``{`` when the reply (after whitespace and tolerated
        openers) starts with an object; ``-1`` when it is definitely text;
        ``None`` when it is too early to tell.
    """
    position = _skip_whitespace(text, 0)
    while position < len(text):
        rest = text[position:].lower()
        if rest.startswith("{"):
            return position
        if any(opener.startswith(rest) and len(rest) < len(opener) for opener in OPENERS):
            return None
        opener = next((opener for opener in OPENERS if rest.startswith(opener)), None)
        if opener is None:
            return -1
        position = _skip_whitespace(text, position + len(opener))
    return None


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


class JsonObjectScanner:
    """Finds where a streamed JSON object closes, without parsing it.

    Tracks brace depth outside string literals (honouring escapes), so
    braces inside argument strings do not end the object early.
    """

    def __init__(self) -> None:
        self._depth = 0
        self._in_string = False
        self._escaped = False

    def feed(self, text: str) -> int | None:
        """Consume ``text``; return the index just past the closing brace, if reached."""
        for index, char in enumerate(text):
            if self._advance(char):
                return index + 1
        return None

    def _advance(self, char: str) -> bool:
        if self._in_string:
            if self._escaped:
                self._escaped = False
            elif char == "\\":
                self._escaped = True
            elif char == '"':
                self._in_string = False
            return False
        if char == '"':
            self._in_string = True
        elif char == "{":
            self._depth += 1
        elif char == "}":
            self._depth -= 1
            return self._depth == 0
        return False


class ToolCallRejectedError(ValueError):
    """A JSON candidate is not a valid call of an offered tool (internal to the parser)."""


def parse_tool_call(candidate: str, tools: Mapping[str, ToolDefinition]) -> ParsedToolCall:
    """Parse and validate one protocol object.

    Raises:
        ToolCallRejectedError: If it is not JSON, names no offered tool, or
            its arguments do not satisfy the tool's schema.
    """
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ToolCallRejectedError(f"not valid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ToolCallRejectedError("not a JSON object")
    name = payload.get(TOOL_KEY, payload.get(NAME_KEY))
    if not isinstance(name, str) or name not in tools:
        raise ToolCallRejectedError(f"unknown tool {name!r}")
    arguments = _arguments_of(payload)
    problems = schema_problems(arguments, tools[name].input_schema)
    if problems:
        raise ToolCallRejectedError(f"{name}: {'; '.join(problems)}")
    return ParsedToolCall(name=name, arguments=arguments)


def _arguments_of(payload: dict[str, Any]) -> dict[str, Any]:
    if ARGUMENTS_KEY in payload:
        arguments = payload[ARGUMENTS_KEY]
        if arguments is None:
            return {}
        if not isinstance(arguments, dict):
            raise ToolCallRejectedError("arguments is not an object")
        return arguments
    # Small models sometimes flatten the arguments next to the tool name.
    return {key: value for key, value in payload.items() if key not in (TOOL_KEY, NAME_KEY)}


def schema_problems(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> list[str]:
    """Check the top level of ``arguments`` against an object JSON schema.

    Deliberately shallow (required keys, declared primitive types, enums,
    ``additionalProperties: false``): its job is to tell a real tool call
    from a stray JSON reply. Full validation is the tool's pydantic model,
    which runs again before anything executes.
    """
    properties: Mapping[str, Any] = schema.get("properties") or {}
    problems = [f"missing '{key}'" for key in schema.get("required") or () if key not in arguments]
    if schema.get("additionalProperties") is False:
        problems += [f"unexpected '{key}'" for key in arguments if key not in properties]
    for key, value in arguments.items():
        declared = properties.get(key)
        if isinstance(declared, Mapping):
            problems += _value_problems(key, value, declared)
    return problems


def _value_problems(key: str, value: Any, declared: Mapping[str, Any]) -> list[str]:
    enum = declared.get("enum")
    if isinstance(enum, list) and value not in enum:
        return [f"'{key}' must be one of {enum}"]
    expected = declared.get("type")
    names = [expected] if isinstance(expected, str) else expected
    if not isinstance(names, list):
        return []
    if any(_has_json_type(value, name) for name in names):
        return []
    return [f"'{key}' must be {' or '.join(str(name) for name in names)}"]


def _has_json_type(value: Any, name: object) -> bool:
    python_types = _JSON_TYPES.get(str(name))
    if python_types is None:
        return True
    if isinstance(value, bool) and name in ("integer", "number"):
        return False
    return isinstance(value, python_types)


# Design notes:
# - Classification is decided by the first non-whitespace characters, the
#   same signal a human would use: prose starts with a letter, a protocol
#   call with "{". Prose is therefore never delayed by more than one token,
#   which keeps time-to-first-audio identical to a text-only model.
# - Openers (```json, ```, <tool_call>) are tolerated because 0.5B-7B models
#   emit them from their training format even when told not to; Qwen2.5 in
#   particular wraps calls in <tool_call> tags. "name" is accepted for
#   "tool" for the same reason, and flattened arguments are tolerated.
# - Only the first object of a reply is considered and the caller stops
#   generation once it validates: the protocol is one call per turn, and
#   the agent loop asks again after the result, which is how the cloud
#   providers chain tools too.
# - Validation is shallow on purpose (see schema_problems). Writing a full
#   JSON Schema validator here would duplicate pydantic, and depending on
#   `jsonschema` for a heuristic would add a package to every install.
# - Rejections are logged with the reason rather than raised: an invalid
#   call is an expected outcome with a small model, and the fallback (speak
#   the text) is the graceful degradation the project rules ask for.
