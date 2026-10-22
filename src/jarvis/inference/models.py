"""The model files used in LOCAL mode and in tests: whisper.cpp, llama.cpp, Piper.

Every file is pinned by Hugging Face revision and SHA-256, like the audio
models, so an upstream re-upload fails verification instead of changing
what the native tests transcribe, generate or speak. These are the *small*
models that the test-suite and a first LOCAL-mode run use; a user who wants
better quality points ``JARVIS_LOCAL_STT_MODEL`` / ``JARVIS_LOCAL_LLM_MODEL``
/ ``JARVIS_LOCAL_TTS_MODEL`` at any other ggml / GGUF / Piper file.
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

_PIPER_VOICES_REVISION: Final = "5041a7a49ae5b3a0f30e6837769eb03fdbb231bd"
_PIPER_VOICE_DIR: Final = (
    f"https://huggingface.co/rhasspy/piper-voices/resolve/{_PIPER_VOICES_REVISION}"
    "/en/en_GB/northern_english_male/medium"
)

PIPER_VOICE_EN_GB: Final = ModelFile(
    name="en_GB-northern_english_male-medium.onnx",
    url=f"{_PIPER_VOICE_DIR}/en_GB-northern_english_male-medium.onnx",
    sha256="57a219ae8e638873db7d18893304be5069c42868f392bb95c3ff17f0690d0689",
    size_bytes=63_201_294,
)
"""Piper voice "northern_english_male", medium quality, 22.05 kHz (63 MB).

A British male voice to match the CLOUD default (en-GB-RyanNeural); the
training data is OpenSLR 83 under CC-BY-SA 4.0, a licence that permits
this use, unlike several other English Piper voices (research-only or NC)."""

PIPER_VOICE_EN_GB_CONFIG: Final = ModelFile(
    name="en_GB-northern_english_male-medium.onnx.json",
    url=f"{_PIPER_VOICE_DIR}/en_GB-northern_english_male-medium.onnx.json",
    sha256="69557ed3d974463453e9b0c09dd99a7ed0e52b8b87b64b357dbeeb2540a97d47",
    size_bytes=4_847,
)
"""The voice's config (sample rate, espeak voice, phoneme map); Piper looks
for it next to the ``.onnx`` file under the same name plus ``.json``."""

INFERENCE_MODELS: Final[tuple[ModelFile, ...]] = (
    WHISPER_TINY,
    QWEN2_5_0_5B_INSTRUCT,
    PIPER_VOICE_EN_GB,
    PIPER_VOICE_EN_GB_CONFIG,
)
"""Everything ``scripts/download_models.py --local`` fetches."""


def piper_config_path(voice_path: Path) -> Path:
    """Where Piper expects the config of the voice at ``voice_path``."""
    return voice_path.with_name(voice_path.name + ".json")


@dataclass(frozen=True, slots=True)
class InferenceModelPaths:
    """Resolved paths handed to the native wrappers and to Piper."""

    whisper: Path
    llama: Path
    piper_voice: Path

    @classmethod
    def in_directory(cls, models_dir: Path) -> InferenceModelPaths:
        """Standard layout under ``models_dir``."""
        return cls(
            whisper=WHISPER_TINY.path(models_dir),
            llama=QWEN2_5_0_5B_INSTRUCT.path(models_dir),
            piper_voice=PIPER_VOICE_EN_GB.path(models_dir),
        )

    @property
    def piper_config(self) -> Path:
        """The ``.onnx.json`` that belongs to :attr:`piper_voice`."""
        return piper_config_path(self.piper_voice)

    def missing(self) -> tuple[Path, ...]:
        """Paths that do not exist yet, for a precise error message."""
        candidates = (self.whisper, self.llama, self.piper_voice, self.piper_config)
        return tuple(path for path in candidates if not path.exists())


# Design notes:
# - ModelFile is reused from jarvis.audio.models: it is a value type with no
#   behaviour beyond path(), so sharing it does not couple the two layers.
# - Opt-in download (--local): 630 MB is too much to pull on every setup of
#   a CLOUD-mode checkout, and the native tests skip cleanly without them.
# - Hugging Face URLs carry the repository revision, not "main", so the
#   pinned hash and the URL describe the same bytes forever.
# - The Piper config is its own manifest entry because it is a separate
#   file upstream; hashing it too means a changed phoneme map cannot slip in
#   under an unchanged voice.
