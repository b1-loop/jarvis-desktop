"""Tool: parse a pasted stack trace into an exception and frames for the LLM.

"Explain this stack trace" with a Python traceback, a Node.js/browser error
or a C++ backtrace (gdb/lldb ``#N`` frames, WinDbg/Visual Studio
``module!function`` frames) on the clipboard. The tool is pure text
processing: it never runs, imports or opens anything named in the trace.
It normalises frames to innermost-first, marks library frames and points
at the innermost frame in the user's own code, which is where an
explanation should start. The explanation itself is the LLM's next turn.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import Field, JsonValue

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult

Language = Literal["python", "javascript", "cpp", "unknown"]

MAX_TRACE_CHARS: Final = 50_000
MAX_FRAMES: Final = 40
"""Frames passed on to the LLM; deep recursion adds length, not insight."""

_PY_FRAME: Final = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>.+)$')
_PY_HEADER: Final = "Traceback"
_PY_EXCEPTION: Final = re.compile(r"^(?P<type>[A-Za-z_][\w.]*)(?::\s*(?P<msg>.*))?$")
_JS_FRAME: Final = re.compile(
    r"^\s*at (?:(?P<func>.+?) \()?(?P<file>[^()\s]+?):(?P<line>\d+):(?P<col>\d+)\)?\s*$"
)
_JS_EXCEPTION: Final = re.compile(
    r"^(?:Uncaught )?(?P<type>[A-Z]\w*(?:Error|Exception))(?::\s*(?P<msg>.*))?$"
)
_GDB_FRAME: Final = re.compile(
    r"^#\d+\s+(?:0x[0-9a-fA-F]+ in )?(?P<func>[^(]+?)\s*(?:\(.*?\))?"
    r"(?: at (?P<file>.+?):(?P<line>\d+)| from (?P<library>\S+))?\s*$"
)
_WINDBG_FRAME: Final = re.compile(
    r"^\s*(?:[0-9a-fA-F`]+\s+)*(?P<module>[\w.]+)!(?P<func>[^+\s\[]+)(?:\+0x[0-9a-fA-F]+)?"
    r"(?:\s+\[(?P<file>.+?)\s*@\s*(?P<line>\d+)\])?\s*$"
)
_CPP_EXCEPTION: Final = re.compile(r"throwing an instance of '(?P<type>[^']+)'")
_CPP_WHAT: Final = re.compile(r"what\(\):\s*(?P<msg>.*)$", re.MULTILINE)
_CPP_SIGNAL: Final = re.compile(
    r"(?P<type>Segmentation fault|SIGSEGV|SIGABRT|Access violation|Aborted)", re.IGNORECASE
)
_LIBRARY_MARKERS: Final = (
    "site-packages",
    "dist-packages",
    "node_modules",
    "node:internal",
    "<frozen",
    "/usr/lib",
    "/usr/include",
    "lib/python",
    "\\lib\\",
    "/lib/",
    "msvcp",
    "ucrtbase",
    "ntdll",
    "kernel32",
    "libc.so",
    "libstdc++",
)


@dataclass(frozen=True, slots=True)
class Frame:
    """One stack frame, as far as the trace tells us."""

    function: str
    file: str
    line: int | None
    column: int | None = None

    @property
    def is_library(self) -> bool:
        """Heuristic: frame lives in a dependency, the runtime or the OS."""
        where = f"{self.file} {self.function}".lower()
        return any(marker in where for marker in _LIBRARY_MARKERS)


@dataclass(frozen=True, slots=True)
class ParsedTrace:
    """Language, exception and frames (innermost first)."""

    language: Language
    exception_type: str | None
    message: str | None
    frames: tuple[Frame, ...]

    @property
    def culprit(self) -> Frame | None:
        """Innermost frame in the user's own code, else the innermost frame."""
        for frame in self.frames:
            if not frame.is_library:
                return frame
        return self.frames[0] if self.frames else None


def _int(value: str | None) -> int | None:
    return int(value) if value else None


def _parse_python(lines: list[str]) -> ParsedTrace:
    frames = [
        Frame(m["func"].strip(), m["file"], _int(m["line"]))
        for line in lines
        if (m := _PY_FRAME.match(line))
    ]
    exception_type = message = None
    if not frames and not any(line.startswith(_PY_HEADER) for line in lines):
        return ParsedTrace("python", None, None, ())
    for line in reversed(lines):
        if not line.startswith((" ", "\t")) and (m := _PY_EXCEPTION.match(line.strip())):
            exception_type, message = m["type"], m["msg"]
            break
    return ParsedTrace("python", exception_type, message, tuple(reversed(frames)))


def _parse_javascript(lines: list[str]) -> ParsedTrace:
    frames = [
        Frame((m["func"] or "<anonymous>").strip(), m["file"], _int(m["line"]), _int(m["col"]))
        for line in lines
        if (m := _JS_FRAME.match(line))
    ]
    exception_type = message = None
    for line in lines:
        if m := _JS_EXCEPTION.match(line.strip()):
            exception_type, message = m["type"], m["msg"]
            break
    return ParsedTrace("javascript", exception_type, message, tuple(frames))


def _parse_cpp(lines: list[str]) -> ParsedTrace:
    frames: list[Frame] = []
    for line in lines:
        if m := _GDB_FRAME.match(line):
            where = m["file"] or m["library"] or ""
            frames.append(Frame(m["func"].strip(), where, _int(m["line"])))
        elif m := _WINDBG_FRAME.match(line):
            frames.append(Frame(m["func"], m["file"] or m["module"], _int(m["line"])))
    text = "\n".join(lines)
    thrown = _CPP_EXCEPTION.search(text)
    what = _CPP_WHAT.search(text)
    signal = _CPP_SIGNAL.search(text)
    exception_type = thrown["type"] if thrown else (signal["type"] if signal else None)
    return ParsedTrace("cpp", exception_type, what["msg"] if what else None, tuple(frames))


_PARSERS: Final[tuple[Callable[[list[str]], ParsedTrace], ...]] = (
    _parse_python,
    _parse_javascript,
    _parse_cpp,
)


def parse_stack_trace(trace: str) -> ParsedTrace:
    """Parse with every dialect and keep the one that recognised most frames."""
    lines = trace.replace("\r\n", "\n").splitlines()
    candidates = [parser(lines) for parser in _PARSERS]
    best = max(
        candidates, key=lambda parsed: (len(parsed.frames), parsed.exception_type is not None)
    )
    if not best.frames and best.exception_type is None:
        return ParsedTrace("unknown", None, None, ())
    return best


class ExplainStackTraceInput(ToolInput):
    """Arguments for :class:`ExplainStackTraceTool`."""

    trace: str = Field(
        min_length=1,
        max_length=MAX_TRACE_CHARS,
        description="The full stack trace text exactly as pasted (Python, JavaScript or C++).",
    )


class ExplainStackTraceTool(Tool[ExplainStackTraceInput]):
    """Structures a stack trace so the LLM can explain it precisely."""

    name = "explain_stack_trace"
    description = (
        "Parse a Python traceback, JavaScript/Node error stack or C++ backtrace into the "
        "exception type, message and frames (innermost first, library frames marked) and "
        "point at the most likely culprit frame in the user's code. Use before explaining "
        "a stack trace; get it from clipboard_get if the user says they copied it."
    )
    Input = ExplainStackTraceInput
    category = ToolCategory.DEVELOPER

    async def run(self, params: ExplainStackTraceInput, context: ToolContext) -> ToolResult:
        """Parse and render; pure CPU work on bounded input, fine on the loop."""
        del context
        parsed = parse_stack_trace(params.trace)
        if parsed.language == "unknown":
            return ToolResult.failure(
                "That doesn't look like a stack trace I can read.",
                detail="no Python, JavaScript or C++ frames or exception line recognised",
            )
        return ToolResult.success(_spoken(parsed), detail=_detail(parsed), data=_data(parsed))


def _spoken(parsed: ParsedTrace) -> str:
    error = parsed.exception_type or "a crash"
    culprit = parsed.culprit
    if culprit is None:
        return f"It's a {parsed.language} {error}, with no frames to point at."
    line = f" line {culprit.line} of" if culprit.line else ""
    return f"It's a {error}, most likely from {culprit.function} at{line} {_short(culprit.file)}."


def _short(path: str) -> str:
    return re.split(r"[\\/]", path)[-1] or path


def _detail(parsed: ParsedTrace) -> str:
    lines = [
        f"language: {parsed.language}",
        f"exception: {parsed.exception_type or 'unknown'}",
        f"message: {parsed.message or '(none)'}",
        "frames (innermost first, * = library):",
    ]
    for frame in parsed.frames[:MAX_FRAMES]:
        marker = "*" if frame.is_library else " "
        location = f"{frame.file}:{frame.line}" if frame.line else frame.file
        lines.append(f" {marker} {frame.function} ({location})")
    if parsed.culprit is not None:
        lines.append(f"likely culprit: {parsed.culprit.function} in {parsed.culprit.file}")
    return "\n".join(lines)


def _data(parsed: ParsedTrace) -> dict[str, JsonValue]:
    frames: list[JsonValue] = [
        {
            "function": frame.function,
            "file": frame.file,
            "line": frame.line,
            "column": frame.column,
            "library": frame.is_library,
        }
        for frame in parsed.frames[:MAX_FRAMES]
    ]
    return {
        "language": parsed.language,
        "exception_type": parsed.exception_type,
        "message": parsed.message,
        "frames": frames,
        "frame_count": len(parsed.frames),
    }


# Design notes:
# - Every dialect parser runs and the one with the most frames wins; that
#   is more robust than sniffing a header, because pasted traces are often
#   truncated at the top or wrapped in log prefixes.
# - Frames are normalised to innermost first (Python prints outermost first)
#   so "the culprit" is always the first non-library frame.
# - A Python exception line ("Name: message") is only trusted next to
#   frames or a "Traceback" header; on its own the pattern would match any
#   single word.
# - Library detection is a path heuristic and is labelled as such; it is a
#   hint for the LLM, not a claim.
