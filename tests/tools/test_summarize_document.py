from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.tools.summarize_document import SummarizeDocumentTool
from tests.tools.local_fakes import local_context

MARKDOWN = """# Plan

## Goals
Ship the voice pipeline.

## Risks
{risks}

## Conclusion
On track.
"""


@pytest.fixture
def context(make_settings: Callable[..., Settings]) -> ToolContext:
    return local_context(make_settings())


async def test_returns_the_text_with_headings_for_the_llm(
    context: ToolContext, tmp_path: Path
) -> None:
    document = tmp_path / "plan.md"
    document.write_text(MARKDOWN.format(risks="Latency."), encoding="utf-8")

    result = await SummarizeDocumentTool().call({"path": str(document)}, context)

    assert result.ok
    assert result.spoken == "I've read plan.md."
    assert "## Goals" in result.detail
    assert "Ship the voice pipeline." in result.detail
    assert result.data is not None
    assert result.data["truncated"] is False
    assert result.data["sections"] == 4


async def test_long_documents_are_trimmed_but_keep_every_heading(
    context: ToolContext, tmp_path: Path
) -> None:
    document = tmp_path / "plan.md"
    document.write_text(MARKDOWN.format(risks="risk " * 2000), encoding="utf-8")

    result = await SummarizeDocumentTool().call({"path": str(document), "max_chars": 1000}, context)

    assert len(result.detail) <= 1000
    assert "## Conclusion" in result.detail
    assert "On track." in result.detail
    assert result.data is not None
    assert result.data["truncated"] is True


@pytest.mark.parametrize(
    ("name", "reason"), [("missing.md", "not found"), ("x.exe", "unsupported")]
)
async def test_unreadable_files_fail_with_the_reason(
    context: ToolContext, tmp_path: Path, name: str, reason: str
) -> None:
    (tmp_path / "x.exe").write_bytes(b"MZ")

    result = await SummarizeDocumentTool().call({"path": str(tmp_path / name)}, context)

    assert not result.ok
    assert result.spoken == f"I couldn't read {name}."
    assert reason in result.detail
