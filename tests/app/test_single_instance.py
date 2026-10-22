import asyncio
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from jarvis.app.single_instance import (
    LOOPBACK,
    Handoff,
    InstanceLock,
    InstanceServer,
    ask_running_instance_to_show,
    read_handoff,
    write_handoff,
)


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "jarvis.lock"


@pytest.fixture
def first(lock_path: Path) -> Iterator[InstanceLock]:
    lock = InstanceLock(lock_path)
    yield lock
    lock.release()


# -- lock ---------------------------------------------------------------------------


def test_first_acquire_wins_and_creates_the_directory(first: InstanceLock, lock_path: Path) -> None:
    assert first.acquire()
    assert first.held
    assert lock_path.is_file()


def test_second_acquire_fails_while_the_first_holds_it(
    first: InstanceLock, lock_path: Path
) -> None:
    second = InstanceLock(lock_path)
    assert first.acquire()

    assert not second.acquire()
    assert not second.held


def test_acquire_is_idempotent_for_the_holder(first: InstanceLock) -> None:
    assert first.acquire()
    assert first.acquire()


def test_release_lets_the_next_launch_in(first: InstanceLock, lock_path: Path) -> None:
    second = InstanceLock(lock_path)
    first.acquire()

    first.release()
    first.release()  # idempotent

    assert second.acquire()
    second.release()


def test_stale_file_from_a_crashed_instance_does_not_block(lock_path: Path) -> None:
    # A crash leaves the file behind but the OS has dropped the lock with the process.
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(b"12345 left over by a dead process")

    lock = InstanceLock(lock_path)
    assert lock.acquire()
    lock.release()


# -- hand-over file -------------------------------------------------------------------


def test_handoff_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "jarvis.instance"
    handoff = Handoff(port=50123, token="t0k", pid=42)

    write_handoff(path, handoff)

    assert read_handoff(path) == handoff


def test_missing_handoff_is_none(tmp_path: Path) -> None:
    assert read_handoff(tmp_path / "absent") is None


@pytest.mark.parametrize("content", ["not json", "{}", '{"port": "x", "token": 1, "pid": 2}'])
def test_garbled_handoff_is_none(tmp_path: Path, content: str) -> None:
    path = tmp_path / "jarvis.instance"
    path.write_text(content, encoding="utf-8")

    assert read_handoff(path) is None


# -- loopback server --------------------------------------------------------------------


async def test_second_launch_asks_the_running_one_to_show(tmp_path: Path) -> None:
    path = tmp_path / "jarvis.instance"
    shown: list[bool] = []
    server = InstanceServer(lambda: shown.append(True), path)
    await server.start()
    try:
        published = read_handoff(path)
        assert published is not None
        assert published.pid == os.getpid()

        assert await ask_running_instance_to_show(path)
        assert shown == [True]
    finally:
        await server.aclose()
    assert not path.exists()


async def test_a_request_without_the_token_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "jarvis.instance"
    shown: list[bool] = []
    server = InstanceServer(lambda: shown.append(True), path)
    await server.start()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        write_handoff(path, Handoff(port=raw["port"], token="wrong", pid=raw["pid"]))

        assert not await ask_running_instance_to_show(path)
        assert shown == []
    finally:
        await server.aclose()


async def test_ping_without_a_published_instance_is_false(tmp_path: Path) -> None:
    assert not await ask_running_instance_to_show(tmp_path / "jarvis.instance")


async def test_ping_to_a_dead_port_is_false(tmp_path: Path) -> None:
    probe = await asyncio.start_server(lambda _r, _w: None, LOOPBACK, 0)
    port = probe.sockets[0].getsockname()[1]
    probe.close()
    await probe.wait_closed()
    path = tmp_path / "jarvis.instance"
    write_handoff(path, Handoff(port=port, token="t", pid=1))

    assert not await ask_running_instance_to_show(path, reply_within=0.5)


async def test_server_close_is_idempotent(tmp_path: Path) -> None:
    server = InstanceServer(lambda: None, tmp_path / "jarvis.instance")
    await server.aclose()
    await server.start()
    await server.aclose()
    await server.aclose()
