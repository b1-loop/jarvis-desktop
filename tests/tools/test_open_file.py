from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.os_integration import StubPlatform
from jarvis.tools.open_file import OpenFileTool
from tests.tools.local_fakes import local_context


async def test_opens_an_existing_document(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF-1.4")
    platform = StubPlatform()

    result = await OpenFileTool().call(
        {"path": str(document)}, local_context(make_settings(), platform=platform)
    )

    assert result.ok
    assert result.spoken == "Opening report.pdf."
    assert platform.calls == [("launch", (str(document), ""))]


@pytest.mark.parametrize("name", ["setup.exe", "run.BAT", "deploy.ps1", "link.lnk", "x.sh"])
async def test_refuses_anything_that_would_execute(
    make_settings: Callable[..., Settings], tmp_path: Path, name: str
) -> None:
    target = tmp_path / name
    target.write_bytes(b"")
    platform = StubPlatform()

    result = await OpenFileTool().call(
        {"path": str(target)}, local_context(make_settings(), platform=platform)
    )

    assert not result.ok
    assert "not a document" in result.spoken
    assert platform.calls == []


async def test_reports_a_missing_file(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    result = await OpenFileTool().call(
        {"path": str(tmp_path / "gone.docx")},
        local_context(make_settings(), platform=StubPlatform()),
    )

    assert not result.ok
    assert result.spoken == "I can't find the file gone.docx."
