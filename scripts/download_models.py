"""Download the ONNX models the audio engine needs, verifying each by SHA-256.

Usage::

    uv run python scripts/download_models.py            # into JARVIS_MODELS_DIR
    uv run python scripts/download_models.py --force    # re-download everything
    uv run python scripts/download_models.py --models-dir C:/models

Files that already exist with the right hash are skipped, so this is safe to
run on every setup. The model list lives in ``jarvis.audio.models`` next to
the code that consumes it.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from jarvis.audio.models import AUDIO_MODELS, ModelFile
from jarvis.config import Settings
from jarvis.utils.downloads import DownloadError, download_verified, is_verified


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Settings().models_dir,
        help="where to put the files (default: JARVIS_MODELS_DIR)",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if verified")
    return parser.parse_args(argv)


async def fetch(
    client: httpx.AsyncClient, model: ModelFile, models_dir: Path, *, force: bool
) -> None:
    """Download one model unless a verified copy is already present."""
    destination = model.path(models_dir)
    if not force and is_verified(destination, model.sha256):
        print(f"  {model.name}: already present and verified")
        return
    print(f"  {model.name}: downloading {model.size_bytes / 1e6:.1f} MB")
    await download_verified(client, model.url, destination, model.sha256)
    print(f"  {model.name}: ok")


async def run(models_dir: Path, *, force: bool) -> int:
    """Download every audio model; return a process exit code."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        try:
            for model in AUDIO_MODELS:
                await fetch(client, model, models_dir, force=force)
        except DownloadError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    print(f"models directory: {args.models_dir.resolve()}")
    return asyncio.run(run(args.models_dir, force=args.force))


if __name__ == "__main__":
    sys.exit(main())
