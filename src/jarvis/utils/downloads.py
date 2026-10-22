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
from typing import IO, Final

import httpx

from jarvis.utils.errors import JarvisError

_CHUNK_BYTES: Final = 1024 * 1024


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

    The body is streamed to a sibling ``.part`` file and hashed as it
    arrives, so a 500 MB GGUF never sits in memory, and the file is renamed
    into place only after the hash matches.

    Raises:
        DownloadError: On any HTTP failure or hash mismatch; nothing is left
            at ``destination`` in that case.
    """
    partial = destination.with_name(destination.name + ".part")
    await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
    try:
        actual = await _stream_to_file(client, url, partial)
    except httpx.HTTPError as error:
        await asyncio.to_thread(partial.unlink, missing_ok=True)
        raise DownloadError(f"failed to download {url}: {error}") from error
    if actual != sha256:
        await asyncio.to_thread(partial.unlink, missing_ok=True)
        raise DownloadError(
            f"checksum mismatch for {url}: expected {sha256}, got {actual}. "
            "The upstream file may have changed; refusing to use it."
        )
    await asyncio.to_thread(partial.replace, destination)


async def _stream_to_file(client: httpx.AsyncClient, url: str, partial: Path) -> str:
    """Write the response body to ``partial``; return its SHA-256 hex digest."""
    digest = hashlib.sha256()
    handle: IO[bytes] = await asyncio.to_thread(partial.open, "wb")
    try:
        async with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(_CHUNK_BYTES):
                digest.update(chunk)
                await asyncio.to_thread(handle.write, chunk)
    finally:
        await asyncio.to_thread(handle.close)
    return digest.hexdigest()


# Design notes:
# - Streaming to disk replaced the earlier buffer-everything approach when
#   the LOCAL-mode models (hundreds of megabytes) joined the manifest.
# - File I/O goes through asyncio.to_thread so the event loop (and any audio
#   polling on it) never blocks on the disk; one megabyte per hop keeps the
#   thread hand-offs negligible next to the network time.
