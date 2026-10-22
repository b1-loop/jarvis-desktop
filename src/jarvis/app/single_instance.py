"""One assistant per user: an OS file lock plus a loopback hand-over.

Two copies of Jarvis would both open the microphone, both answer "Hey
Jarvis" and both write the memory database. The first launch therefore
takes an exclusive lock on ``data_dir/jarvis.lock`` and keeps it for the
life of the process. A later launch that cannot take the lock connects to
the first one over ``127.0.0.1`` (port and a random token in
``data_dir/jarvis.instance``), asks it to show its dashboard, and exits.

The lock is held by the operating system (``msvcrt.locking`` on Windows,
``flock`` elsewhere), not by the file's existence, so a crashed instance
never leaves a stale lock behind: the OS drops it with the process, and
the leftover file is simply locked again by the next launch.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Final

from jarvis.utils.logging import get_logger

if sys.platform == "win32":
    import msvcrt

    def _try_lock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


log = get_logger(__name__)

LOCK_FILENAME: Final = "jarvis.lock"
HANDOFF_FILENAME: Final = "jarvis.instance"
LOOPBACK: Final = "127.0.0.1"
SHOW_COMMAND: Final = "show"
ACK: Final = b"ok\n"
EXCHANGE_TIMEOUT_S: Final = 2.0
"""A local round trip takes milliseconds; two seconds means the first instance is stuck."""
_TOKEN_BYTES: Final = 16


class InstanceLock:
    """Exclusive, process-lifetime lock on one file.

    Args:
        path: The lock file; created (with its directory) if missing.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: IO[bytes] | None = None

    @property
    def held(self) -> bool:
        """Whether this object currently holds the lock."""
        return self._file is not None

    def acquire(self) -> bool:
        """Take the lock without waiting.

        Returns:
            True when this process now owns the instance, False when another
            process (or another ``InstanceLock`` on the same file) holds it.

        Raises:
            OSError: If the file cannot be created or opened at all.
        """
        if self._file is not None:
            return True
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+b")
        try:
            _try_lock(handle.fileno())
        except OSError:
            handle.close()
            return False
        self._file = handle
        log.info("instance.lock.acquired", path=str(self._path), pid=os.getpid())
        return True

    def release(self) -> None:
        """Give the lock up. Idempotent; the file stays for the next launch."""
        handle, self._file = self._file, None
        if handle is None:
            return
        try:
            _unlock(handle.fileno())
        finally:
            handle.close()
        log.info("instance.lock.released", path=str(self._path))


@dataclass(frozen=True, slots=True)
class Handoff:
    """How a second launch reaches the running instance.

    Attributes:
        port: Loopback TCP port the running instance listens on.
        token: Random secret the request must carry, so only someone who
            can read this user's data directory can poke the instance.
        pid: Process id of the running instance, for the log.
    """

    port: int
    token: str
    pid: int


def write_handoff(path: Path, handoff: Handoff) -> None:
    """Publish ``handoff`` for later launches."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(handoff)), encoding="utf-8")


def read_handoff(path: Path) -> Handoff | None:
    """The published hand-over, or ``None`` when absent or unreadable.

    A missing file is normal (the first instance is still starting); a
    garbled one is logged, since it can only come from a crash mid-write.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Handoff(port=int(raw["port"]), token=str(raw["token"]), pid=int(raw["pid"]))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as error:
        log.warning("instance.handoff.unreadable", path=str(path), error=str(error))
        return None


class InstanceServer:
    """Loopback listener in the running instance that answers "show" requests.

    Args:
        on_show: Called on the loop when a valid request arrives.
        handoff_path: Where the port and token are published.
    """

    def __init__(self, on_show: Callable[[], None], handoff_path: Path) -> None:
        self._on_show = on_show
        self._handoff_path = handoff_path
        self._token = secrets.token_urlsafe(_TOKEN_BYTES)
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        """Listen on an ephemeral loopback port and publish it."""
        self._server = await asyncio.start_server(self._handle, LOOPBACK, 0)
        port = int(self._server.sockets[0].getsockname()[1])
        handoff = Handoff(port=port, token=self._token, pid=os.getpid())
        await asyncio.to_thread(write_handoff, self._handoff_path, handoff)
        log.info("instance.server.listening", port=port)

    async def aclose(self) -> None:
        """Stop listening and withdraw the hand-over file. Idempotent."""
        server, self._server = self._server, None
        if server is None:
            return
        server.close()
        await server.wait_closed()
        await asyncio.to_thread(self._handoff_path.unlink, missing_ok=True)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), EXCHANGE_TIMEOUT_S)
            if line.decode("utf-8", "replace").strip() == f"{SHOW_COMMAND} {self._token}":
                log.info("instance.show_requested")
                self._on_show()
                writer.write(ACK)
                await writer.drain()
            else:
                log.warning("instance.request.rejected")
        except (TimeoutError, OSError) as error:
            log.warning("instance.request.failed", error=str(error))
        finally:
            await _close(writer)


async def ask_running_instance_to_show(
    handoff_path: Path, reply_within: float = EXCHANGE_TIMEOUT_S
) -> bool:
    """Ask the instance that holds the lock to bring its dashboard up.

    Returns:
        True when it acknowledged; False when it has not published a port
        yet or did not answer, in which case the caller tells the user
        another way (a notification).
    """
    handoff = read_handoff(handoff_path)
    if handoff is None:
        return False
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(LOOPBACK, handoff.port), reply_within
        )
        try:
            writer.write(f"{SHOW_COMMAND} {handoff.token}\n".encode())
            await writer.drain()
            reply = await asyncio.wait_for(reader.readline(), reply_within)
        finally:
            await _close(writer)
    except (TimeoutError, OSError) as error:
        log.warning("instance.ping.failed", port=handoff.port, error=str(error))
        return False
    return reply == ACK


async def _close(writer: asyncio.StreamWriter) -> None:
    """Close a connection and wait for it, so no transport is left to the GC."""
    writer.close()
    try:
        await writer.wait_closed()
    except OSError as error:  # the peer hung up first; the socket is closed either way
        log.debug("instance.connection.reset", error=str(error))


# Design notes:
# - A lock file over a named mutex (CreateMutexW): both die with the
#   process, but the file lock behaves the same on Windows and Linux, so
#   the single-instance logic is tested on the Linux CI runner instead of
#   being Windows-only code behind a skip.
# - Only the first byte is locked on Windows (msvcrt.locking locks a byte
#   range); the hand-over lives in a separate file because a locked range
#   cannot be read by the second process.
# - An ephemeral port published in a file beats a fixed port: nothing else
#   on the machine can already own it, and the token in the same file
#   means another local user's process cannot trigger the "show" action.
# - The server never trusts the request beyond the exact command line; the
#   only thing a request can do is call on_show (show the dashboard).
# - A failed ping is not an error worth raising: the second launch still
#   exits, and the caller notifies "Jarvis is already running" instead.
