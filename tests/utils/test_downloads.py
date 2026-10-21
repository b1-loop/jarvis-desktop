import hashlib
from pathlib import Path

import httpx
import pytest

from jarvis.utils.downloads import DownloadError, download_verified, is_verified, sha256_of

BODY = b"model bytes " * 1000
BODY_SHA256 = hashlib.sha256(BODY).hexdigest()


def client_returning(status: int, body: bytes = BODY) -> httpx.AsyncClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(BODY)

    assert sha256_of(path) == BODY_SHA256


def test_is_verified_requires_existence_and_matching_hash(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    assert not is_verified(path, BODY_SHA256)

    path.write_bytes(BODY)
    assert is_verified(path, BODY_SHA256)
    assert not is_verified(path, "0" * 64)


async def test_download_writes_verified_file_into_new_directory(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "model.onnx"

    async with client_returning(200) as client:
        await download_verified(client, "https://example.test/model", destination, BODY_SHA256)

    assert destination.read_bytes() == BODY
    assert not destination.with_name("model.onnx.part").exists()


async def test_checksum_mismatch_raises_and_leaves_nothing_behind(tmp_path: Path) -> None:
    destination = tmp_path / "model.onnx"

    async with client_returning(200) as client:
        with pytest.raises(DownloadError, match="checksum mismatch"):
            await download_verified(client, "https://example.test/model", destination, "0" * 64)

    assert not destination.exists()


async def test_http_errors_become_download_errors(tmp_path: Path) -> None:
    destination = tmp_path / "model.onnx"

    async with client_returning(404) as client:
        with pytest.raises(DownloadError, match="failed to download"):
            await download_verified(client, "https://example.test/model", destination, BODY_SHA256)

    assert not destination.exists()
