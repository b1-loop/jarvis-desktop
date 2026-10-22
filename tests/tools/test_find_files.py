import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.tools.find_files import FindFilesTool
from tests.tools.local_fakes import local_context

DAY = 24 * 3600


def write(path: Path, text: str, *, age_days: float = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    stamp = time.time() - age_days * DAY
    os.utime(path, (stamp, stamp))
    return path


@pytest.fixture
def context(make_settings: Callable[..., Settings], tmp_path: Path) -> ToolContext:
    write(tmp_path / "Hiring Plan 2026.md", "# Hiring\nTwo engineers.", age_days=2)
    write(tmp_path / "old" / "hiring plan draft.txt", "first draft", age_days=40)
    write(tmp_path / "budget.txt", "The hiring budget is fixed.", age_days=1)
    write(tmp_path / "setup.exe", "binary", age_days=0)
    return local_context(make_settings(document_roots=[tmp_path]))


async def test_finds_documents_by_name_newest_first(context: ToolContext) -> None:
    result = await FindFilesTool().call({"query": "hiring plan"}, context)

    assert result.ok
    assert result.spoken == "I found 2 documents: Hiring Plan 2026.md, hiring plan draft.txt."
    assert result.data is not None
    rows = result.data["matches"]
    assert isinstance(rows, list)
    assert [row["name"] for row in rows if isinstance(row, dict)] == [
        "Hiring Plan 2026.md",
        "hiring plan draft.txt",
    ]
    assert "Hiring Plan 2026.md" in result.detail


async def test_content_search_and_age_filter(context: ToolContext) -> None:
    result = await FindFilesTool().call(
        {"query": "hiring", "search_content": True, "modified_within_days": 7}, context
    )

    assert result.spoken == "I found 2 documents: budget.txt, Hiring Plan 2026.md."


async def test_single_match_is_phrased_in_the_singular(context: ToolContext) -> None:
    result = await FindFilesTool().call({"query": "budget"}, context)

    assert result.spoken == "I found one document: budget.txt."


async def test_reports_no_matches_and_where_it_looked(context: ToolContext, tmp_path: Path) -> None:
    result = await FindFilesTool().call({"query": "setup"}, context)

    assert result.ok
    assert result.spoken == "I couldn't find any documents matching setup."
    assert str(tmp_path) in result.detail
    assert result.data == {"matches": []}
