from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from jarvis import _native
from jarvis.config import Settings
from jarvis.inference import native as inference_native
from jarvis.stt import whisper_cpp
from jarvis.stt.base import STT, TranscriptionError
from jarvis.stt.whisper_cpp import WhisperCppSTT, to_transcript
from jarvis.utils.errors import ConfigError, NativeUnavailableError


@dataclass(frozen=True)
class FakeNativeSegment:
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class FakeNativeTranscript:
    text: str
    language: str
    segments: list[FakeNativeSegment]


@dataclass
class FakeTranscriber:
    """Records calls like ``jarvis_inference.WhisperTranscriber.transcribe``."""

    result: FakeNativeTranscript
    calls: list[dict[str, object]] = field(default_factory=list)
    failure: Exception | None = None

    def transcribe(
        self,
        samples: np.ndarray,
        language: str | None = None,
        translate: bool = False,
        threads: int = 0,
        beam_size: int = 1,
    ) -> FakeNativeTranscript:
        if self.failure is not None:
            raise self.failure
        self.calls.append(
            {
                "samples": int(samples.size),
                "dtype": samples.dtype,
                "language": language,
                "translate": translate,
                "threads": threads,
                "beam_size": beam_size,
            }
        )
        return self.result


HELLO = FakeNativeTranscript(
    text=" Hello there, Jarvis.",
    language="en",
    segments=[FakeNativeSegment(0, 600, " Hello there,"), FakeNativeSegment(600, 1000, " Jarvis.")],
)


def one_second(rate: int = 16_000) -> np.ndarray:
    return np.random.default_rng(seed=3).uniform(-0.1, 0.1, rate).astype(np.float32)


@pytest.fixture
def transcriber() -> FakeTranscriber:
    return FakeTranscriber(HELLO)


async def test_produces_the_same_transcript_type_as_faster_whisper(
    transcriber: FakeTranscriber,
) -> None:
    stt = WhisperCppSTT(lambda: transcriber, language="en", threads=4)

    transcript = await stt.transcribe(one_second())

    assert isinstance(stt, STT)
    assert transcript.text == "Hello there, Jarvis."
    assert transcript.language == "en"
    assert transcript.duration == 1.0
    assert transcript.confidence is None
    assert [(s.start, s.end, s.text) for s in transcript.segments] == [
        (0.0, 0.6, "Hello there,"),
        (0.6, 1.0, "Jarvis."),
    ]
    assert transcriber.calls == [
        {
            "samples": 16_000,
            "dtype": np.float32,
            "language": "en",
            "translate": False,
            "threads": 4,
            "beam_size": 1,
        }
    ]


async def test_other_rates_are_resampled_to_16_khz(transcriber: FakeTranscriber) -> None:
    stt = WhisperCppSTT(lambda: transcriber)

    transcript = await stt.transcribe(one_second(48_000), sample_rate=48_000)

    assert transcriber.calls[0]["samples"] == 16_000
    assert transcriber.calls[0]["language"] is None
    assert transcript.duration == pytest.approx(1.0)


async def test_empty_audio_skips_the_model() -> None:
    loads: list[int] = []

    def factory() -> FakeTranscriber:
        loads.append(1)
        return FakeTranscriber(HELLO)

    transcript = await WhisperCppSTT(factory).transcribe(np.zeros(0, dtype=np.float32))

    assert transcript.is_empty
    assert transcript.duration == 0.0
    assert not loads


async def test_stereo_input_is_rejected(transcriber: FakeTranscriber) -> None:
    with pytest.raises(TranscriptionError, match="mono"):
        await WhisperCppSTT(lambda: transcriber).transcribe(np.zeros((2, 10), dtype=np.float32))


def test_non_speech_annotations_are_dropped_so_silence_is_empty() -> None:
    native = FakeNativeTranscript(
        text=" [BLANK_AUDIO]",
        language="en",
        segments=[
            FakeNativeSegment(0, 500, " [BLANK_AUDIO]"),
            FakeNativeSegment(500, 900, " (wind blowing)"),
            FakeNativeSegment(900, 1000, "   "),
        ],
    )

    transcript = to_transcript(native, duration=1.0)

    assert transcript.is_empty
    assert transcript.language is None
    assert transcript.segments == ()


def test_brackets_inside_speech_are_kept() -> None:
    native = FakeNativeTranscript(
        text="", language="", segments=[FakeNativeSegment(0, 900, " Open (the) browser")]
    )

    transcript = to_transcript(native, duration=0.9)

    assert transcript.text == "Open (the) browser"
    assert transcript.language is None  # whisper.cpp reported none


async def test_model_loads_once_even_with_concurrent_first_calls(
    transcriber: FakeTranscriber,
) -> None:
    loads: list[int] = []

    def factory() -> FakeTranscriber:
        loads.append(1)
        return transcriber

    stt = WhisperCppSTT(factory)
    assert not stt.loaded
    await stt.warm_up()
    await stt.transcribe(one_second())

    assert stt.loaded
    assert loads == [1]


async def test_native_failure_becomes_transcription_error(transcriber: FakeTranscriber) -> None:
    transcriber.failure = RuntimeError("whisper_full failed")

    with pytest.raises(TranscriptionError, match="whisper_full failed"):
        await WhisperCppSTT(lambda: transcriber).transcribe(one_second())


async def test_load_failure_becomes_transcription_error() -> None:
    def broken() -> FakeTranscriber:
        raise RuntimeError("invalid model file")

    with pytest.raises(TranscriptionError, match="invalid model file"):
        await WhisperCppSTT(broken).warm_up()


async def test_config_errors_from_the_factory_pass_through() -> None:
    def unconfigured() -> FakeTranscriber:
        raise ConfigError("remedy")

    with pytest.raises(ConfigError, match="remedy"):
        await WhisperCppSTT(unconfigured).warm_up()


# --- construction from settings ---------------------------------------------------


def test_from_settings_requires_the_model_file(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match="JARVIS_LOCAL_STT_MODEL"):
        WhisperCppSTT.from_settings(make_settings(local_stt_model=tmp_path / "nope.bin"))


def test_from_settings_requires_the_native_build(
    make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "ggml.bin"
    model.write_bytes(b"ggml")

    def unavailable(name: str) -> ModuleType:
        raise NativeUnavailableError(name, "DLL load failed")

    monkeypatch.setattr(_native, "load", unavailable)

    with pytest.raises(ConfigError, match=r"DLL load failed.*build_native\.py"):
        WhisperCppSTT.from_settings(make_settings(local_stt_model=model))


def test_from_settings_rejects_the_stub_build(
    make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "ggml.bin"
    model.write_bytes(b"ggml")
    stub = SimpleNamespace(IMPLEMENTATION_STATUS="disabled: built with JARVIS_BUILD_INFERENCE=OFF")
    monkeypatch.setattr(_native, "load", lambda _name: stub)

    with pytest.raises(ConfigError, match="JARVIS_BUILD_INFERENCE=ON"):
        WhisperCppSTT.from_settings(make_settings(local_stt_model=model))


async def test_from_settings_wires_path_language_and_threads(
    make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "ggml.bin"
    model.write_bytes(b"ggml")
    transcriber = FakeTranscriber(HELLO)
    opened: list[Path] = []

    def whisper_transcriber(path: Path) -> FakeTranscriber:
        opened.append(path)
        return transcriber

    module = SimpleNamespace(
        IMPLEMENTATION_STATUS="implemented in step 12", WhisperTranscriber=whisper_transcriber
    )
    monkeypatch.setattr(inference_native._native, "load", lambda _name: module)
    settings = make_settings(local_stt_model=model, stt_language="de", local_stt_threads=2)

    stt = WhisperCppSTT.from_settings(settings)
    await stt.transcribe(one_second())

    assert opened == [model]
    assert transcriber.calls[0]["language"] == "de"
    assert transcriber.calls[0]["threads"] == 2
    assert transcriber.calls[0]["beam_size"] == whisper_cpp.GREEDY_BEAM_SIZE


def test_empty_language_setting_means_auto_detect(make_settings: Callable[..., Settings]) -> None:
    assert make_settings(stt_language="").stt_language is None
