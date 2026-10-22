import asyncio
import sys
from pathlib import Path

import pytest

from jarvis.utils.sandbox import (
    CommandNotFoundError,
    CommandRefusedError,
    SandboxPolicy,
    spawn_sandboxed,
)

PYTHON = f'"{sys.executable}"'


@pytest.fixture
def policy(tmp_path: Path) -> SandboxPolicy:
    return SandboxPolicy(allowed_binaries=frozenset({"python"}), working_root=tmp_path)


async def test_spawn_returns_immediately_and_appends_both_streams_to_the_log(
    policy: SandboxPolicy, tmp_path: Path
) -> None:
    log = tmp_path / "logs" / "server.log"
    code = "import sys; print('out'); print('err', file=sys.stderr)"

    process = await spawn_sandboxed(f'{PYTHON} -c "{code}"', policy, log_path=log)

    assert await process.wait() == 0
    text = await asyncio.to_thread(log.read_text, encoding="utf-8")
    assert "out" in text
    assert "err" in text


async def test_spawn_withholds_secrets_and_honours_the_directory(
    policy: SandboxPolicy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    child = tmp_path / "child"
    child.mkdir()
    log = tmp_path / "env.log"
    code = "import os; print(os.getcwd()); print(os.environ.get('ANTHROPIC_API_KEY'))"

    process = await spawn_sandboxed(f'{PYTHON} -c "{code}"', policy, log_path=log, cwd=child)
    await process.wait()

    lines = (await asyncio.to_thread(log.read_text, encoding="utf-8")).splitlines()
    assert await asyncio.to_thread(Path(lines[0]).samefile, child)
    assert lines[1] == "None"


async def test_spawn_applies_the_same_refusals(policy: SandboxPolicy, tmp_path: Path) -> None:
    log = tmp_path / "never.log"

    with pytest.raises(CommandRefusedError):
        await spawn_sandboxed("python -c 1 && rm -rf /", policy, log_path=log)
    with pytest.raises(CommandRefusedError):
        await spawn_sandboxed(f"{PYTHON} -V", policy, log_path=log, cwd=tmp_path.parent)
    with pytest.raises(CommandNotFoundError):
        await spawn_sandboxed("definitely-not-a-real-binary", policy, log_path=log)
    assert not log.exists()
