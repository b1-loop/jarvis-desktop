"""The ONNX models the audio engine needs and where they live on disk.

Each model is pinned by URL and SHA-256 so that a file that silently changed
upstream fails verification instead of producing a detector that behaves
differently from the one the tests were written against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True, slots=True)
class ModelFile:
    """One downloadable model, pinned by content hash.

    Attributes:
        name: File name inside the models directory.
        url: Where to fetch it from.
        sha256: Expected hex digest of the file contents.
        size_bytes: Expected size, used for progress output only.
    """

    name: str
    url: str
    sha256: str
    size_bytes: int

    def path(self, models_dir: Path) -> Path:
        """Location of this model inside ``models_dir``."""
        return models_dir / self.name


SILERO_VAD: Final = ModelFile(
    name="silero_vad.onnx",
    url="https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx",
    sha256="2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f",
    size_bytes=2_327_524,
)
"""Silero VAD v5: speech probability per 32 ms frame."""

MELSPECTROGRAM: Final = ModelFile(
    name="melspectrogram.onnx",
    url="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    sha256="ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    size_bytes=1_087_958,
)
"""openWakeWord stage 1: raw audio to mel spectrogram."""

EMBEDDING: Final = ModelFile(
    name="embedding_model.onnx",
    url="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
    sha256="70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
    size_bytes=1_326_578,
)
"""openWakeWord stage 2: mel spectrogram to speech embedding."""

HEY_JARVIS: Final = ModelFile(
    name="hey_jarvis_v0.1.onnx",
    url="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx",
    sha256="94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
    size_bytes=1_271_370,
)
"""openWakeWord stage 3: embeddings to a "hey Jarvis" score."""

AUDIO_MODELS: Final[tuple[ModelFile, ...]] = (SILERO_VAD, MELSPECTROGRAM, EMBEDDING, HEY_JARVIS)
"""Everything ``scripts/download_models.py`` fetches for the audio engine."""


@dataclass(frozen=True, slots=True)
class AudioModelPaths:
    """Resolved paths handed to the native engine."""

    vad: Path
    melspectrogram: Path
    embedding: Path
    wake_word: Path

    @classmethod
    def in_directory(cls, models_dir: Path) -> AudioModelPaths:
        """Standard layout under ``models_dir``."""
        return cls(
            vad=SILERO_VAD.path(models_dir),
            melspectrogram=MELSPECTROGRAM.path(models_dir),
            embedding=EMBEDDING.path(models_dir),
            wake_word=HEY_JARVIS.path(models_dir),
        )

    def missing(self) -> tuple[Path, ...]:
        """Paths that do not exist yet, for a precise error message."""
        candidates = (self.vad, self.melspectrogram, self.embedding, self.wake_word)
        return tuple(path for path in candidates if not path.exists())


# Design notes:
# - The wake-word classifier is a separate ModelFile so a custom phrase is a
#   one-line change here plus a config value, exactly as openWakeWord intends.
# - Hashes were computed from the files at the pinned release tags on
#   2026-10-21; the native tests load the same files, so a hash change here
#   must be accompanied by re-running them.
