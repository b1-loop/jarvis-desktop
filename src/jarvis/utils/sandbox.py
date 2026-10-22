"""Shell command policy and sandboxed execution.

Jarvis runs commands the LLM proposes, so the policy comes first and the
process second. A command is parsed into an argv (never handed to a shell),
checked against a denylist of destructive patterns and an allowlist of
binaries, and then run with a timeout, an output cap, a working directory
pinned under a configured root and an environment stripped of everything
but the handful of variables a build tool needs. The policy decides between
three verdicts: run it, ask the user first, or refuse outright.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import time
from asyncio.subprocess import DEVNULL, PIPE, STDOUT, Process
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, BinaryIO, Final

from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import Settings

log = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_MAX_OUTPUT_BYTES: Final = 64 * 1024
_READ_CHUNK_BYTES: Final = 4096
_MS_PER_SECOND: Final = 1000.0

_EXECUTABLE_SUFFIXES: Final = (".exe", ".cmd", ".bat", ".com", ".ps1")
_SHELL_OPERATORS: Final = frozenset({"&&", "||", "|", ";", "&", ">", ">>", "<", "2>", "2>&1"})
_REDIRECTION_PREFIXES: Final = (">", "<", "2>")

REFUSED_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern)
    for pattern in (
        r"^(rm|del|erase)\s+(.*\s)?-[a-z]*(rf|fr)[a-z]*(\s|$)",  # rm -rf, rm -fr, rm -Rf
        r"^(rm|del|erase)\s+(-[a-z]+\s+)*[/\\~]\s*$",  # rm / , del \
        r"^rm\s+(-[a-z]+\s+)*(-r|--recursive)\s+(/|~)(\s|$)",  # rm -r /
        r"^(del|erase|rmdir|rd)\s+.*(/s|/q)",  # del /s, rmdir /s
        r"^format(\.com)?(\s|$)",
        r"^diskpart(\s|$)",
        r"^mkfs",
        r"^dd\s+.*\bof=",
        r"^(shutdown|restart-computer|stop-computer|logoff)(\s|$)",
        r"^reg(\.exe)?\s+(delete|import|restore)(\s|$)",
        r"^bcdedit(\s|$)",
        r"^cipher\s+.*/w",
        r"^(net|net1)\s+(user|localgroup|share)(\s|$)",
        r"^(takeown|icacls|cacls)(\s|$)",
        r"^wmic\s+.*\b(delete|terminate|call)\b",
        r"^(sudo|runas|su)(\s|$)",
        r"^(cmd|powershell|pwsh|bash|sh|zsh|wsl)(\.exe)?(\s|$)",  # a shell escapes the policy
        r"^remove-item\s+.*-recurse",
        r"^git\s+push\s+.*(--force|-f)(\s|$).*\b(main|master)\b",
        r":\(\)\s*\{",  # fork bomb
    )
)
"""Never run, no matter what the user says; the LLM gets a refusal instead."""

CONFIRM_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = tuple(
    (re.compile(pattern), reason)
    for pattern, reason in (
        (r"^(rm|rmdir|rd|del|erase|remove-item)(\s|$)", "deletes files"),
        (
            r"^git\s+(push\s+.*(--force|-f)|reset\s+--hard|clean\s+-[a-z]*[fx]|branch\s+-D)",
            "rewrites git history or discards work",
        ),
        (r"^git\s+checkout\s+(--\s+)?\.(\s|$)", "discards uncommitted changes"),
        (r"^(pip|uv|npm|pnpm|yarn|cargo)\s+(uninstall|remove|rm)(\s|$)", "removes packages"),
        (r"^(kill|taskkill|pkill|stop-process)(\s|$)", "terminates processes"),
        (r"^(mv|move|ren|rename)(\s|$)", "moves or renames files"),
        (r"^(chmod|chown|attrib)(\s|$)", "changes permissions"),
    )
)
"""Run only after spoken confirmation."""

SAFE_ENVIRONMENT_KEYS: Final = frozenset(
    key.upper()
    for key in (
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "HOME",
        "USERNAME",
        "USER",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "LANG",
        "LC_ALL",
        "TERM",
        "SHELL",
    )
)
"""The only inherited variables; everything else (API keys, tokens) is withheld."""

_FORCED_ENVIRONMENT: Final = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


class SandboxError(JarvisError):
    """Base class for sandbox failures; the message is safe to speak aloud."""


class CommandRefusedError(SandboxError):
    """The policy refused to run the command at all.

    Attributes:
        command: The command line as given.
        reason: Why, in words the user can hear.
    """

    def __init__(self, command: str, reason: str) -> None:
        self.command = command
        self.reason = reason
        super().__init__(f"refused to run {command!r}: {reason}")


class CommandNotFoundError(SandboxError):
    """The binary could not be found on ``PATH``."""

    def __init__(self, binary: str) -> None:
        self.binary = binary
        super().__init__(f"command not found: {binary}")


class Verdict(StrEnum):
    """What the policy says about a command."""

    ALLOWED = "allowed"
    NEEDS_CONFIRMATION = "needs_confirmation"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class CommandCheck:
    """Outcome of :func:`check_command`.

    Attributes:
        verdict: Run, ask or refuse.
        reason: Why, phrased for the user.
        argv: The parsed command; empty when refused for being unparsable.
        binary: Normalised program name (lower-case, no path, no ``.exe``).
    """

    verdict: Verdict
    reason: str
    argv: tuple[str, ...]
    binary: str


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    """Limits applied to every sandboxed command.

    Attributes:
        allowed_binaries: Normalised program names that run without asking.
        working_root: Commands may only run in this directory or below it.
        timeout_seconds: Wall-clock limit before the process is killed.
        max_output_bytes: Cap per stream; the rest is read and discarded.
    """

    allowed_binaries: frozenset[str]
    working_root: Path
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES

    @classmethod
    def from_settings(cls, settings: Settings) -> SandboxPolicy:
        """Build the policy from the ``JARVIS_SHELL_*`` settings."""
        return cls(
            allowed_binaries=frozenset(normalize_binary(name) for name in settings.shell_allowlist),
            working_root=settings.shell_working_root or Path.home(),
            timeout_seconds=float(settings.shell_timeout_seconds),
        )


@dataclass(frozen=True, slots=True)
class CommandResult:
    """What a sandboxed command produced.

    Attributes:
        argv: Exactly what was executed.
        exit_code: Process exit status; ``None`` when it was killed.
        stdout: Decoded standard output, possibly truncated.
        stderr: Decoded standard error, possibly truncated.
        duration_ms: Wall time from spawn to exit.
        timed_out: True if the policy timeout killed the process.
        truncated: True if either stream hit the output cap.
    """

    argv: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool
    truncated: bool

    @property
    def ok(self) -> bool:
        """True when the process exited with status zero in time."""
        return self.exit_code == 0 and not self.timed_out


def normalize_binary(name: str) -> str:
    """Reduce ``C:/Python312/python.EXE`` to ``python`` for allowlist lookup."""
    # PureWindowsPath understands both separators and drive letters on every
    # host, so Linux CI judges Windows command lines the same way Windows does.
    base = PureWindowsPath(name.strip().strip('"')).name.lower()
    for suffix in _EXECUTABLE_SUFFIXES:
        if base.endswith(suffix):
            return base.removesuffix(suffix)
    return base


def parse_command(command: str) -> tuple[str, ...]:
    """Split a command line into argv the way the target OS would.

    POSIX quoting rules on POSIX; on Windows backslashes are path separators,
    so quotes are honoured but backslashes are left alone.
    """
    if os.name == "nt":
        return tuple(token.strip('"') for token in shlex.split(command, posix=False))
    return tuple(shlex.split(command))


def check_command(command: str, policy: SandboxPolicy) -> CommandCheck:
    """Classify ``command`` without running it."""
    try:
        argv = parse_command(command)
    except ValueError as error:
        return CommandCheck(Verdict.REFUSED, f"unbalanced quotes ({error})", (), "")
    if not argv:
        return CommandCheck(Verdict.REFUSED, "the command is empty", (), "")
    binary = normalize_binary(argv[0])
    verdict, reason = _classify(argv, binary, policy)
    return CommandCheck(verdict, reason, argv, binary)


def _classify(argv: tuple[str, ...], binary: str, policy: SandboxPolicy) -> tuple[Verdict, str]:
    """Apply the rules in order of severity: operators, denylist, confirm list, allowlist."""
    normalized = " ".join([binary, *argv[1:]]).lower()
    if any(_is_shell_operator(token) for token in argv):
        return Verdict.REFUSED, "shell operators are not allowed"
    if any(pattern.search(normalized) for pattern in REFUSED_PATTERNS):
        return Verdict.REFUSED, "matches a destructive pattern"
    for pattern, reason in CONFIRM_PATTERNS:
        if pattern.search(normalized):
            return Verdict.NEEDS_CONFIRMATION, reason
    if binary not in policy.allowed_binaries:
        return Verdict.NEEDS_CONFIRMATION, f"{binary} is not on the allowlist"
    return Verdict.ALLOWED, f"{binary} is on the allowlist"


def _is_shell_operator(token: str) -> bool:
    return token in _SHELL_OPERATORS or token.startswith(_REDIRECTION_PREFIXES)


def scrub_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Keep only the variables a build tool needs; never pass secrets through."""
    kept = {key: value for key, value in source.items() if key.upper() in SAFE_ENVIRONMENT_KEYS}
    kept.update(_FORCED_ENVIRONMENT)
    return kept


def resolve_working_directory(requested: Path | None, policy: SandboxPolicy) -> Path:
    """Return the directory to run in, which must be at or below the root.

    Raises:
        CommandRefusedError: If it escapes the root or does not exist.
    """
    root = policy.working_root.resolve()
    target = (requested or root).resolve()
    if not target.is_relative_to(root):
        raise CommandRefusedError(str(requested), f"working directory is outside {root}")
    if not target.is_dir():
        raise CommandRefusedError(str(requested), f"working directory {target} does not exist")
    return target


class _Capture:
    """Reads one pipe to the end, keeping at most ``limit`` bytes."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self.data = bytearray()
        self.truncated = False

    async def pump(self, stream: asyncio.StreamReader | None) -> None:
        """Drain ``stream`` so the child never blocks on a full pipe."""
        if stream is None:
            return
        while chunk := await stream.read(_READ_CHUNK_BYTES):
            room = self._limit - len(self.data)
            if room > 0:
                self.data.extend(chunk[:room])
            if len(chunk) > room:
                self.truncated = True

    def text(self) -> str:
        """Decoded output; undecodable bytes are replaced, never raised."""
        return self.data.decode("utf-8", errors="replace")


async def run_sandboxed(
    command: str, policy: SandboxPolicy, *, cwd: Path | None = None
) -> CommandResult:
    """Run ``command`` under ``policy`` and collect its output.

    Confirmation is the caller's job: a command whose verdict is
    ``NEEDS_CONFIRMATION`` runs here, because the agent only calls this after
    the confirmation policy has asked the user. Refused commands never run.

    Raises:
        CommandRefusedError: If the policy refuses the command or the
            working directory.
        CommandNotFoundError: If the binary is not on ``PATH``.
    """
    check, argv, workdir = _prepare(command, policy, cwd)
    log.info("sandbox.run", binary=check.binary, cwd=str(workdir), verdict=check.verdict.value)
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=workdir,
        env=scrub_environment(os.environ),
        stdin=DEVNULL,
        stdout=PIPE,
        stderr=PIPE,
    )
    stdout, stderr, timed_out = await _collect(process, policy)
    duration_ms = (time.perf_counter() - started) * _MS_PER_SECOND
    result = CommandResult(
        argv=argv,
        exit_code=None if timed_out else process.returncode,
        stdout=stdout.text(),
        stderr=stderr.text(),
        duration_ms=duration_ms,
        timed_out=timed_out,
        truncated=stdout.truncated or stderr.truncated,
    )
    log.info("sandbox.done", binary=check.binary, exit_code=result.exit_code, timed_out=timed_out)
    return result


def _prepare(
    command: str, policy: SandboxPolicy, cwd: Path | None
) -> tuple[CommandCheck, tuple[str, ...], Path]:
    """Apply the policy and resolve binary and directory, before anything spawns.

    Raises:
        CommandRefusedError: If the policy refuses the command or directory.
        CommandNotFoundError: If the binary is not on ``PATH``.
    """
    check = check_command(command, policy)
    if check.verdict is Verdict.REFUSED:
        raise CommandRefusedError(command, check.reason)
    workdir = resolve_working_directory(cwd, policy)
    executable = shutil.which(check.argv[0])
    if executable is None:
        raise CommandNotFoundError(check.argv[0])
    return check, (executable, *check.argv[1:]), workdir


async def spawn_sandboxed(
    command: str, policy: SandboxPolicy, *, log_path: Path, cwd: Path | None = None
) -> Process:
    """Start ``command`` under ``policy`` and return without waiting for it.

    For processes that are meant to keep running, such as a dev server. The
    same refusal, directory, binary and environment rules as
    :func:`run_sandboxed` apply; the timeout does not, because outliving it
    is the point. Both output streams go to ``log_path`` (appended), never to
    an unread pipe that would eventually block the child.

    Raises:
        CommandRefusedError: If the policy refuses the command or directory.
        CommandNotFoundError: If the binary is not on ``PATH``.
    """
    check, argv, workdir = _prepare(command, policy, cwd)
    log_file = await asyncio.to_thread(_open_log, log_path)
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=workdir,
            env=scrub_environment(os.environ),
            stdin=DEVNULL,
            stdout=log_file,
            stderr=STDOUT,
        )
    finally:
        log_file.close()  # the child holds its own handle
    log.info("sandbox.spawned", binary=check.binary, cwd=str(workdir), pid=process.pid)
    return process


def _open_log(path: Path) -> BinaryIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("ab")


async def _collect(process: Process, policy: SandboxPolicy) -> tuple[_Capture, _Capture, bool]:
    stdout = _Capture(policy.max_output_bytes)
    stderr = _Capture(policy.max_output_bytes)
    try:
        async with asyncio.timeout(policy.timeout_seconds):
            await asyncio.gather(stdout.pump(process.stdout), stderr.pump(process.stderr))
            await process.wait()
    except TimeoutError:
        log.warning("sandbox.timeout", limit_s=policy.timeout_seconds)
        process.kill()
        await process.wait()
        return stdout, stderr, True
    return stdout, stderr, False


# Design notes:
# - Policy first, process second: the verdict is computed from the parsed
#   argv before anything is spawned, and the same function backs the
#   shell tool's "needs confirmation" decision so the user is asked about
#   exactly what will run.
# - No shell, ever. ``create_subprocess_exec`` with an argv means ``&&``,
#   pipes and redirects cannot do anything; they are refused anyway so the
#   LLM gets a clear message instead of a program that received "|" as an
#   argument. The denylist also refuses ``cmd``, ``powershell`` and ``bash``
#   because ``cmd /c <anything>`` would bypass every other rule.
# - Environment allowlist instead of a secret denylist: a new credential name
#   is safe by default. Python children get UTF-8 forced so their output
#   decodes on Windows.
# - spawn_sandboxed shares _prepare with run_sandboxed so a background
#   process can never take a path around the policy. Its output goes to a
#   log file rather than a pipe: nobody reads a dev server's pipe, and a
#   full pipe would freeze it.
# - Output beyond the cap is drained and discarded rather than left in the
#   pipe, otherwise a chatty child would block on a full pipe and hit the
#   timeout instead of finishing.
