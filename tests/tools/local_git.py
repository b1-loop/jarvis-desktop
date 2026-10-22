"""Throwaway git repositories for the dev-mode tool tests, built through the sandbox."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from jarvis.utils.sandbox import SandboxPolicy, run_sandboxed

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

_IDENTITY = "-c user.name=Jarvis -c user.email=jarvis@example.com -c commit.gpgsign=false"


async def git(root: Path, cwd: Path, arguments: str) -> str:
    """Run ``git <arguments>`` in ``cwd`` (under ``root``) and return stdout; fail loudly."""
    policy = SandboxPolicy(allowed_binaries=frozenset({"git"}), working_root=root)
    result = await run_sandboxed(f"git {_IDENTITY} {arguments}", policy, cwd=cwd)
    assert result.ok, result.stderr
    return result.stdout


async def make_repo(root: Path, name: str = "repo") -> Path:
    """A repository with one commit on ``main`` containing ``README.md``."""
    repo = root / name
    repo.mkdir()
    await git(root, repo, "init -q -b main")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    await git(root, repo, "add README.md")
    await git(root, repo, "commit -q -m initial")
    return repo
