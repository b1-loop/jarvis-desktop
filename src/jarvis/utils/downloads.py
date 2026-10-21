"""Verified, atomic file downloads.

Model files are fetched once and then trusted by the native engine, so the
download is the one place where integrity is checked: the content hash must
match before the file is moved into place, and a partial or corrupt download
never leaves a file behind that a later run would mistake for a good one.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Final

import httpx

from jarvis.utils.errors import JarvisError

_CHUNK_BYTES: Final = 64 * 1024


class DownloadError(JarvisError):
    """A download failed or its content did not match the expected hash."""


def sha256_of(path: Path) -> str:
    """Hex SHA-256 digest of a file, read in chunks so large models fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def is_verified(path: Path, sha256: str) -> bool:
    """True if ``path`` exists and has exactly the expected content."""
    return path.exists() and sha256_of(path) == sha256


async def download_verified(
    client: httpx.AsyncClient, url: str, destination: Path, sha256: str
) -> None:
    """Fetch ``url`` into ``destination`` and verify it.

    The body is streamed and hashed as it arrives, written to a sibling
    ``.part`` file in a worker thread, and renamed into place only after the
    hash matches.

    Raises:
        DownloadError: On any HTTP failure or hash mismatch; nothing is left
            at ``destination`` in that case.
    """
    try:
        body = await _fetch(client, url)
    except httpx.HTTPError as error:
        raise DownloadError(f"failed to download {url}: {error}") from error
    actual = hashlib.sha256(body).hexdigest()
    if actual != sha256:
        raise DownloadError(
            f"checksum mismatch for {url}: expected {sha256}, got {actual}. "
            "The upstream file may have changed; refusing to use it."
        )
    await asyncio.to_thread(_write_atomically, destination, body)


async def _fetch(client: httpx.AsyncClient, url: str) -> bytes:
    chunks = bytearray()
    async with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes(_CHUNK_BYTES):
            chunks.extend(chunk)
    return bytes(chunks)


def _write_atomically(destination: Path, body: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(body)
    partial.replace(destination)


# Design notes:
# - The whole body is held in memory before writing. Audio models are a few
#   megabytes; if a future local-LLM download needs gigabytes, switch to
#   hashing on the fly and streaming to the .part file in a thread.
# - File I/O goes through asyncio.to_thread so the event loop (and any audio
#   polling on it) never blocks on the disk.
