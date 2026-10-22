"""Download the models Jarvis needs, verifying each by SHA-256.

Usage::

    uv run python scripts/download_models.py            # audio models into JARVIS_MODELS_DIR
    uv run python scripts/download_models.py --local    # plus whisper.cpp, llama.cpp, Piper models
    uv run python scripts/download_models.py --piper-sv # plus the Swedish Piper voice (63 MB)
    uv run python scripts/download_models.py --force    # re-download everything
    uv run python scripts/download_models.py --models-dir C:/models

Files that already exist with the right hash are skipped, so this is safe to
run on every setup. The audio model list lives in ``jarvis.audio.models`` and
the LOCAL-mode list in ``jarvis.inference.models``, next to the code that
consumes them. ``--local`` is opt-in because those files total 630 MB.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from jarvis.audio.models import AUDIO_MODELS, ModelFile
from jarvis.config import Settings
from jarvis.inference.models import INFERENCE_MODELS, SWEDISH_VOICE_MODELS
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
    parser.add_argument(
        "--local",
        action="store_true",
        help="also fetch the whisper.cpp, llama.cpp and Piper models for LOCAL mode (630 MB)",
    )
    parser.add_argument(
        "--piper-sv",
        action="store_true",
        help="also fetch the Swedish Piper voice (sv_SE-nst-medium, 63 MB)",
    )
    return parser.parse_args(argv)


def selected_models(*, local: bool, swedish_voice: bool = False) -> tuple[ModelFile, ...]:
    """The manifest entries a run should fetch."""
    models = AUDIO_MODELS + INFERENCE_MODELS if local else AUDIO_MODELS
    return models + SWEDISH_VOICE_MODELS if swedish_voice else models


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


async def run(models_dir: Path, *, force: bool, local: bool, swedish_voice: bool = False) -> int:
    """Download every selected model; return a process exit code."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        try:
            for model in selected_models(local=local, swedish_voice=swedish_voice):
                await fetch(client, model, models_dir, force=force)
        except DownloadError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    print(f"models directory: {args.models_dir.resolve()}")
    return asyncio.run(
        run(args.models_dir, force=args.force, local=args.local, swedish_voice=args.piper_sv)
    )


if __name__ == "__main__":
    sys.exit(main())
