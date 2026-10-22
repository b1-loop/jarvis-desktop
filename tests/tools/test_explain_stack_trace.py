from collections.abc import Callable

import pytest

from jarvis.config import Settings
from jarvis.tools.explain_stack_trace import ExplainStackTraceTool, parse_stack_trace
from tests.tools.local_fakes import local_context

PYTHON_TRACE = """\
Traceback (most recent call last):
  File "/home/me/app/main.py", line 10, in <module>
    run()
  File "/home/me/app/service.py", line 42, in run
    data = json.loads(raw)
  File "/usr/lib/python3.12/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
"""

JS_TRACE = """\
TypeError: Cannot read properties of undefined (reading 'map')
    at renderList (/srv/web/src/List.js:12:19)
    at Object.<anonymous> (/srv/web/node_modules/react-dom/cjs/react-dom.js:100:3)
    at /srv/web/src/index.js:5:1
"""

GDB_TRACE = """\
terminate called after throwing an instance of 'std::out_of_range'
  what():  vector::_M_range_check: __n (which is 5) >= this->size() (which is 3)
#0  0x00007ffff7a42e97 in raise () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x0000555555555abc in std::vector<int>::at (__n=5) at /usr/include/c++/12/stl_vector.h:1125
#2  0x0000555555555def in compute (values=...) at /home/me/engine/compute.cpp:27
#3  0x0000555555556123 in main () at /home/me/engine/main.cpp:9
"""

WINDBG_TRACE = """\
Access violation - code c0000005
00 jarvis_audio!RingBuffer::push+0x1a [C:/src/audio/ring_buffer.cpp @ 88]
01 jarvis_audio!Engine::on_frames+0x44 [C:/src/audio/engine.cpp @ 140]
02 KERNEL32!BaseThreadInitThunk+0x14
"""


def test_python_frames_are_innermost_first_with_the_culprit_in_user_code() -> None:
    parsed = parse_stack_trace(PYTHON_TRACE)

    assert parsed.language == "python"
    assert parsed.exception_type == "json.decoder.JSONDecodeError"
    assert parsed.message == "Expecting value: line 1 column 1 (char 0)"
    assert [frame.function for frame in parsed.frames] == ["loads", "run", "<module>"]
    assert parsed.frames[0].is_library
    assert parsed.culprit is not None
    assert (parsed.culprit.function, parsed.culprit.line) == ("run", 42)


def test_javascript_frames_with_and_without_function_names() -> None:
    parsed = parse_stack_trace(JS_TRACE)

    assert parsed.language == "javascript"
    assert parsed.exception_type == "TypeError"
    assert [(f.function, f.line, f.column) for f in parsed.frames] == [
        ("renderList", 12, 19),
        ("Object.<anonymous>", 100, 3),
        ("<anonymous>", 5, 1),
    ]
    assert parsed.culprit is not None
    assert parsed.culprit.function == "renderList"


def test_gdb_backtrace_with_an_uncaught_exception() -> None:
    parsed = parse_stack_trace(GDB_TRACE)

    assert parsed.language == "cpp"
    assert parsed.exception_type == "std::out_of_range"
    assert parsed.message is not None
    assert parsed.message.startswith("vector::_M_range_check")
    assert parsed.culprit is not None
    assert (parsed.culprit.function, parsed.culprit.line) == ("compute", 27)


def test_windbg_stack_with_an_access_violation() -> None:
    parsed = parse_stack_trace(WINDBG_TRACE)

    assert parsed.language == "cpp"
    assert parsed.exception_type == "Access violation"
    assert [frame.function for frame in parsed.frames] == [
        "RingBuffer::push",
        "Engine::on_frames",
        "BaseThreadInitThunk",
    ]
    assert parsed.frames[0].line == 88
    assert parsed.frames[2].is_library


async def test_tool_speaks_the_error_and_the_culprit(
    make_settings: Callable[..., Settings],
) -> None:
    result = await ExplainStackTraceTool().call(
        {"trace": PYTHON_TRACE}, local_context(make_settings())
    )

    assert result.ok
    assert result.spoken == (
        "It's a json.decoder.JSONDecodeError, most likely from run at line 42 of service.py."
    )
    assert "likely culprit: run in /home/me/app/service.py" in result.detail
    assert " * loads (/usr/lib/python3.12/json/__init__.py:346)" in result.detail
    assert result.data is not None
    assert result.data["frame_count"] == 3


async def test_an_exception_line_without_frames_is_still_explained(
    make_settings: Callable[..., Settings],
) -> None:
    result = await ExplainStackTraceTool().call(
        {"trace": "Segmentation fault (core dumped)"}, local_context(make_settings())
    )

    assert result.ok
    assert result.spoken == "It's a cpp Segmentation fault, with no frames to point at."


@pytest.mark.parametrize("text", ["hello there, how are you today?", "   "])
async def test_non_traces_are_rejected(make_settings: Callable[..., Settings], text: str) -> None:
    result = await ExplainStackTraceTool().call({"trace": text}, local_context(make_settings()))

    assert not result.ok
