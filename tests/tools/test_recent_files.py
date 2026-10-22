from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.os_integration import StubPlatform
from jarvis.tools.recent_files import RecentFilesTool
from tests.tools.local_fakes import local_context
from tests.tools.test_find_files import write


@pytest.fixture
def settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    write(tmp_path / "older.md", "x", age_days=3)
    write(tmp_path / "newest.docx.txt", "y", age_days=0.1)
    return make_settings(document_roots=[tmp_path])


async def test_lists_recent_documents_without_opening(settings: Settings) -> None:
    platform = StubPlatform()

    result = await RecentFilesTool().call({}, local_context(settings, platform=platform))

    assert result.ok
    assert result.spoken.startswith("The most recent is newest.docx.txt, changed ")
    assert result.data is not None
    assert result.data["opened"] is None
    assert platform.calls == []


async def test_opens_the_last_thing_worked_on(settings: Settings, tmp_path: Path) -> None:
    platform = StubPlatform()

    result = await RecentFilesTool().call(
        {"open_most_recent": True}, local_context(settings, platform=platform)
    )

    newest = str((tmp_path / "newest.docx.txt").resolve())
    assert result.spoken == "Opening newest.docx.txt."
    assert platform.calls == [("launch", (newest, ""))]
    assert result.data is not None
    assert result.data["opened"] == newest


async def test_listing_works_without_a_platform(settings: Settings) -> None:
    result = await RecentFilesTool().call({"limit": 1}, local_context(settings))

    assert result.ok
    assert result.data is not None
    assert isinstance(result.data["matches"], list)
    assert len(result.data["matches"]) == 1


async def test_says_so_when_there_is_nothing(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    result = await RecentFilesTool().call(
        {"open_most_recent": True}, local_context(make_settings(document_roots=[empty]))
    )

    assert result.ok
    assert result.spoken == "I couldn't find any recent documents."
