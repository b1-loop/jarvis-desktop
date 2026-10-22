from collections.abc import Callable

from jarvis.config import Settings
from jarvis.os_integration import StubPlatform
from jarvis.tools.clipboard import MAX_CLIPBOARD_CHARS, ClipboardGetTool, ClipboardSetTool
from tests.tools.local_fakes import local_context


async def test_get_returns_the_text_and_a_spoken_preview(
    make_settings: Callable[..., Settings],
) -> None:
    platform = StubPlatform(clipboard="Meeting moved\nto Thursday")

    result = await ClipboardGetTool().call({}, local_context(make_settings(), platform=platform))

    assert result.spoken == "The clipboard says: Meeting moved to Thursday"
    assert result.detail == "Meeting moved\nto Thursday"
    assert result.data == {
        "text": "Meeting moved\nto Thursday",
        "characters": 25,
        "truncated": False,
    }


async def test_get_caps_huge_clipboards(make_settings: Callable[..., Settings]) -> None:
    platform = StubPlatform(clipboard="x" * (MAX_CLIPBOARD_CHARS + 10))

    result = await ClipboardGetTool().call({}, local_context(make_settings(), platform=platform))

    assert result.data is not None
    assert result.data["truncated"] is True
    assert len(str(result.data["text"])) == MAX_CLIPBOARD_CHARS


async def test_get_on_an_empty_clipboard(make_settings: Callable[..., Settings]) -> None:
    result = await ClipboardGetTool().call(
        {}, local_context(make_settings(), platform=StubPlatform())
    )

    assert result.ok
    assert result.spoken == "The clipboard has no text on it."


async def test_set_replaces_the_clipboard(make_settings: Callable[..., Settings]) -> None:
    platform = StubPlatform(clipboard="old")

    result = await ClipboardSetTool().call(
        {"text": "new text"}, local_context(make_settings(), platform=platform)
    )

    assert result.spoken == "Copied to the clipboard."
    assert platform.clipboard == "new text"
