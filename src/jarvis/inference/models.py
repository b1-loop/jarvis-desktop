"""The whisper.cpp and llama.cpp model files used in LOCAL mode and in tests.

Both are pinned by Hugging Face revision and SHA-256, like the audio models,
so an upstream re-upload fails verification instead of changing what the
native tests transcribe or generate. These are the *small* models that the
test-suite and a first LOCAL-mode run use; a user who wants better quality
points ``JARVIS_LOCAL_STT_MODEL`` / ``JARVIS_LOCAL_LLM_MODEL`` at any other
ggml / GGUF file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from jarvis.audio.models import ModelFile

WHISPER_TINY: Final = ModelFile(
    name="ggml-tiny.bin",
    url=(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/"
        "5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.bin"
    ),
    sha256="be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
    size_bytes=77_691_713,
)
"""Whisper tiny (multilingual, ggml format, 75 MB): fast enough for tests
on a CI-class CPU; the real LOCAL mode default is a larger model."""

QWEN2_5_0_5B_INSTRUCT: Final = ModelFile(
    name="qwen2.5-0.5b-instruct-q4_k_m.gguf",
    url=(
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/"
        "9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    ),
    sha256="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
    size_bytes=491_400_032,
)
"""Qwen2.5 0.5B Instruct, Q4_K_M (Apache-2.0, 490 MB): a chat model small
enough to generate a few tokens in a test, with a ChatML template that
llama.cpp renders natively."""

INFERENCE_MODELS: Final[tuple[ModelFile, ...]] = (WHISPER_TINY, QWEN2_5_0_5B_INSTRUCT)
"""Everything ``scripts/download_models.py --local`` fetches."""


@dataclass(frozen=True, slots=True)
class InferenceModelPaths:
    """Resolved paths handed to the native wrappers."""

    whisper: Path
    llama: Path

    @classmethod
    def in_directory(cls, models_dir: Path) -> InferenceModelPaths:
        """Standard layout under ``models_dir``."""
        return cls(
            whisper=WHISPER_TINY.path(models_dir),
            llama=QWEN2_5_0_5B_INSTRUCT.path(models_dir),
        )

    def missing(self) -> tuple[Path, ...]:
        """Paths that do not exist yet, for a precise error message."""
        return tuple(path for path in (self.whisper, self.llama) if not path.exists())


# Design notes:
# - ModelFile is reused from jarvis.audio.models: it is a value type with no
#   behaviour beyond path(), so sharing it does not couple the two layers.
# - Opt-in download (--local): 570 MB is too much to pull on every setup of
#   a CLOUD-mode checkout, and the native tests skip cleanly without them.
# - Hugging Face URLs carry the repository revision, not "main", so the
#   pinned hash and the URL describe the same bytes forever.
