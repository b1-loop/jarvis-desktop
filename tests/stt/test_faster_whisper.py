import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from structlog.testing import CapturingLogger

from jarvis.config import Settings
from jarvis.stt import faster_whisper as faster_whisper_module
from jarvis.stt.base import ENGINE_SAMPLE_RATE, STT, Transcript, TranscriptionError
from jarvis.stt.faster_whisper import (
    DecodeOptions,
    FasterWhisperSTT,
    is_gpu_library_error,
    missing_gpu_library,
    resample_linear,
)
from jarvis.utils.errors import JarvisError
from tests.stt.fakes import FakeInfo, FakeSegment, FakeWhisperModel


def one_second_of_noise(rate: int = ENGINE_SAMPLE_RATE) -> np.ndarray:
    generator = np.random.default_rng(seed=7)
    return generator.uniform(-0.1, 0.1, rate).astype(np.float32)


@pytest.fixture
def model() -> FakeWhisperModel:
    return FakeWhisperModel(
        segments=[
            FakeSegment(0.0, 0.6, " Hello there,", avg_logprob=math.log(0.9)),
            FakeSegment(0.6, 1.0, " Jarvis.", avg_logprob=math.log(0.5)),
        ],
        info=FakeInfo(language="en", duration=1.0),
    )


@pytest.fixture
def stt(model: FakeWhisperModel) -> FasterWhisperSTT:
    return FasterWhisperSTT(lambda: model)


def test_it_is_an_stt(stt: FasterWhisperSTT) -> None:
    assert isinstance(stt, STT)
    assert issubclass(TranscriptionError, JarvisError)


async def test_transcript_joins_segments_and_carries_timings(
    stt: FasterWhisperSTT, model: FakeWhisperModel
) -> None:
    transcript = await stt.transcribe(one_second_of_noise())

    assert transcript.text == "Hello there, Jarvis."
    assert transcript.language == "en"
    assert transcript.duration == 1.0
    assert [(s.start, s.end, s.text) for s in transcript.segments] == [
        (0.0, 0.6, "Hello there,"),
        (0.6, 1.0, "Jarvis."),
    ]
    assert not transcript.is_empty
    assert model.calls == [
        {
            "samples": 16_000,
            "language": None,
            "beam_size": 1,
            "vad_filter": False,
            "hotwords": None,
            "without_timestamps": True,
            "condition_on_previous_text": False,
        }
    ]


async def test_confidence_is_exp_logprob_weighted_by_duration(stt: FasterWhisperSTT) -> None:
    transcript = await stt.transcribe(one_second_of_noise())

    assert transcript.segments[0].confidence == pytest.approx(0.9)
    assert transcript.segments[1].confidence == pytest.approx(0.5)
    assert transcript.confidence == pytest.approx(0.6 * 0.9 + 0.4 * 0.5)


async def test_the_lazy_segment_iterator_is_consumed_off_the_loop(
    stt: FasterWhisperSTT, model: FakeWhisperModel
) -> None:
    await stt.transcribe(one_second_of_noise())

    # Both segments were decoded inside transcribe(), not left for later.
    assert model.log == ["decoded", "decoded"]


async def test_model_is_loaded_lazily_and_only_once() -> None:
    loads = 0

    def factory() -> FakeWhisperModel:
        nonlocal loads
        loads += 1
        return FakeWhisperModel(info=FakeInfo(duration=0.5))

    stt = FasterWhisperSTT(factory)
    assert not stt.loaded

    await stt.transcribe(one_second_of_noise())
    await stt.transcribe(one_second_of_noise())

    assert stt.loaded
    assert loads == 1


async def test_warm_up_loads_the_model_and_decodes_a_second_of_silence(
    model: FakeWhisperModel,
) -> None:
    stt = FasterWhisperSTT(lambda: model)

    await stt.warm_up()

    assert stt.loaded
    assert [call["samples"] for call in model.calls] == [16_000]


async def test_a_failing_warm_up_decode_is_logged_not_raised(
    model: FakeWhisperModel, logs: CapturingLogger
) -> None:
    model.failure = RuntimeError("CUDA out of memory")
    stt = FasterWhisperSTT(lambda: model)

    await stt.warm_up()

    assert stt.loaded
    assert [call.args[0] for call in logs.calls if call.method_name == "warning"] == [
        "stt.warm_up_failed"
    ]


async def test_empty_audio_is_an_empty_transcript_without_loading(
    model: FakeWhisperModel,
) -> None:
    stt = FasterWhisperSTT(lambda: model)

    transcript = await stt.transcribe(np.zeros(0, dtype=np.float32))

    assert transcript == Transcript(
        text="", language=None, segments=(), duration=0.0, confidence=None
    )
    assert transcript.is_empty
    assert not stt.loaded


async def test_silence_with_no_segments_is_empty_but_keeps_the_duration() -> None:
    stt = FasterWhisperSTT(lambda: FakeWhisperModel(info=FakeInfo(duration=2.0)))

    transcript = await stt.transcribe(one_second_of_noise())

    assert transcript.is_empty
    assert transcript.language is None
    assert transcript.duration == 2.0
    assert transcript.confidence is None


async def test_language_and_decode_options_are_forwarded(model: FakeWhisperModel) -> None:
    decode = DecodeOptions(
        beam_size=2, vad_filter=True, without_timestamps=False, condition_on_previous_text=True
    )
    stt = FasterWhisperSTT(lambda: model, language="bg", decode=decode)

    await stt.transcribe(one_second_of_noise())

    assert model.calls[0]["language"] == "bg"
    assert model.calls[0]["beam_size"] == 2
    assert model.calls[0]["vad_filter"] is True
    assert model.calls[0]["without_timestamps"] is False
    assert model.calls[0]["condition_on_previous_text"] is True


async def test_the_beam_size_comes_from_settings(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[..., Settings], registered: list[int]
) -> None:
    model = FakeWhisperModel(info=FakeInfo(duration=0.5))
    install(monkeypatch, SimpleNamespace(WhisperModel=lambda *_a, **_k: model))
    stt = FasterWhisperSTT.from_settings(make_settings(stt_device="cpu", stt_beam_size=4))

    await stt.transcribe(one_second_of_noise())

    assert model.calls[0]["beam_size"] == 4


async def test_other_sample_rates_are_resampled_to_16k(model: FakeWhisperModel) -> None:
    stt = FasterWhisperSTT(lambda: model)

    await stt.transcribe(one_second_of_noise(48_000), sample_rate=48_000)

    assert model.calls[0]["samples"] == 16_000


async def test_non_mono_audio_is_rejected(stt: FasterWhisperSTT) -> None:
    with pytest.raises(TranscriptionError, match="mono"):
        await stt.transcribe(np.zeros((2, 100), dtype=np.float32))


async def test_model_failures_become_transcription_errors() -> None:
    model = FakeWhisperModel(failure=RuntimeError("CUDA out of memory"))
    stt = FasterWhisperSTT(lambda: model)

    with pytest.raises(TranscriptionError, match="CUDA out of memory") as excinfo:
        await stt.transcribe(one_second_of_noise())

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_resample_linear_preserves_a_constant_and_scales_length() -> None:
    audio = np.full(480, 0.25, dtype=np.float32)

    resampled = resample_linear(audio, 48_000, 16_000)

    assert resampled.shape == (160,)
    assert resampled.dtype == np.float32
    np.testing.assert_allclose(resampled, 0.25)


def test_resample_linear_is_identity_for_equal_rates_and_empty_input() -> None:
    audio = np.arange(5, dtype=np.float32)

    assert resample_linear(audio, 16_000, 16_000) is audio
    assert resample_linear(np.zeros(0, dtype=np.float32), 8_000, 16_000).size == 0


def test_resample_linear_rejects_bad_rates() -> None:
    with pytest.raises(TranscriptionError):
        resample_linear(np.zeros(4, dtype=np.float32), 0, 16_000)


def test_from_settings_builds_a_lazy_provider(make_settings: Callable[..., Settings]) -> None:
    stt = FasterWhisperSTT.from_settings(make_settings(stt_model="tiny"))

    assert isinstance(stt, FasterWhisperSTT)
    assert not stt.loaded  # nothing downloaded or loaded until first use


# -- CPU fallback when CUDA libraries are missing (device=auto) ---------------

CUBLAS_MISSING = "Library cublas64_12.dll is not found or cannot be loaded"
"""CTranslate2's own message on a machine with a GPU but no CUDA runtime."""


class Factory:
    """Counts loads and hands out the given model, or raises the given error."""

    def __init__(self, model: FakeWhisperModel | None = None, error: str | None = None) -> None:
        self.model = model or FakeWhisperModel(info=FakeInfo(duration=0.5))
        self.error = error
        self.loads = 0

    def __call__(self) -> FakeWhisperModel:
        self.loads += 1
        if self.error is not None:
            raise RuntimeError(self.error)
        return self.model


@pytest.fixture
def logs(monkeypatch: pytest.MonkeyPatch) -> CapturingLogger:
    logger = CapturingLogger()
    monkeypatch.setattr(faster_whisper_module, "log", logger)
    return logger


def warnings(logger: CapturingLogger) -> list[dict[str, Any]]:
    return [call.kwargs for call in logger.calls if call.method_name == "warning"]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (CUBLAS_MISSING, True),
        ("Could not load library cudnn_ops64_9.dll. Error code 126", True),
        ("CUDA driver version is insufficient for CUDA runtime library", True),
        ("CUDA out of memory", False),
        ("model file is corrupt", False),
    ],
)
def test_gpu_library_errors_are_told_apart(message: str, expected: bool) -> None:
    assert is_gpu_library_error(RuntimeError(message)) is expected


def test_the_missing_library_is_named() -> None:
    assert missing_gpu_library(RuntimeError(CUBLAS_MISSING)) == "cublas64_12.dll"
    assert missing_gpu_library(RuntimeError("CUDA driver library")) == "a CUDA runtime library"


async def test_a_load_failure_falls_back_to_cpu_once(logs: CapturingLogger) -> None:
    gpu, cpu = Factory(error=CUBLAS_MISSING), Factory()
    stt = FasterWhisperSTT(gpu, cpu_fallback=cpu)

    await stt.transcribe(one_second_of_noise())
    await stt.transcribe(one_second_of_noise())

    assert (gpu.loads, cpu.loads) == (1, 1)
    assert len(cpu.model.calls) == 2
    (warning,) = warnings(logs)
    assert warning["missing"] == "cublas64_12.dll"


async def test_a_first_transcription_failure_retries_on_cpu(logs: CapturingLogger) -> None:
    gpu = Factory(FakeWhisperModel(failure=RuntimeError(CUBLAS_MISSING)))
    cpu = Factory(FakeWhisperModel(segments=[FakeSegment(0.0, 1.0, " hi")]))
    stt = FasterWhisperSTT(gpu, cpu_fallback=cpu)

    first = await stt.transcribe(one_second_of_noise())
    await stt.transcribe(one_second_of_noise())

    assert first.text == "hi"
    assert len(cpu.model.calls) == 2
    assert len(warnings(logs)) == 1


async def test_without_a_fallback_the_gpu_error_is_loud() -> None:
    stt = FasterWhisperSTT(Factory(FakeWhisperModel(failure=RuntimeError(CUBLAS_MISSING))))

    with pytest.raises(TranscriptionError, match="cublas64_12"):
        await stt.transcribe(one_second_of_noise())


async def test_other_errors_never_trigger_the_fallback() -> None:
    cpu = Factory()
    stt = FasterWhisperSTT(Factory(error="model file is corrupt"), cpu_fallback=cpu)

    with pytest.raises(RuntimeError, match="corrupt"):
        await stt.transcribe(one_second_of_noise())
    assert cpu.loads == 0


async def test_the_fallback_is_tried_only_once(logs: CapturingLogger) -> None:
    gpu = Factory(FakeWhisperModel(failure=RuntimeError(CUBLAS_MISSING)))
    cpu = Factory(FakeWhisperModel(failure=RuntimeError(CUBLAS_MISSING)))
    stt = FasterWhisperSTT(gpu, cpu_fallback=cpu)

    for _ in range(2):
        with pytest.raises(TranscriptionError):
            await stt.transcribe(one_second_of_noise())

    assert cpu.loads == 1
    assert len(warnings(logs)) == 1


class FakeFasterWhisper:
    """Stands in for ``faster_whisper`` (and ``ctranslate2``); CUDA libraries are missing."""

    def __init__(self, *, cuda_devices: int = 1, cuda_works: bool = False) -> None:
        self.devices: list[tuple[str, str]] = []
        self.models: list[str] = []
        self.cuda_devices = cuda_devices
        self.cuda_works = cuda_works

    def WhisperModel(self, size: str, *, device: str, compute_type: str, **_: object) -> object:  # noqa: N802
        self.devices.append((device, compute_type))
        self.models.append(size)
        if device != "cpu" and not self.cuda_works:
            raise RuntimeError(CUBLAS_MISSING)
        return FakeWhisperModel(info=FakeInfo(duration=0.5))

    def get_cuda_device_count(self) -> int:
        return self.cuda_devices


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replaces the DLL-path setup and counts how often it ran."""
    calls: list[int] = []
    monkeypatch.setattr(
        faster_whisper_module, "register_cuda_libraries", lambda: calls.append(1) or ()
    )
    return calls


def install(monkeypatch: pytest.MonkeyPatch, package: FakeFasterWhisper) -> None:
    monkeypatch.setattr(faster_whisper_module.importlib, "import_module", lambda _name: package)


@pytest.mark.parametrize(("device", "falls_back"), [("auto", True), ("cuda", False)])
async def test_from_settings_falls_back_only_for_the_auto_device(  # noqa: PLR0917
    monkeypatch: pytest.MonkeyPatch,
    make_settings: Callable[..., Settings],
    logs: CapturingLogger,
    registered: list[int],
    device: str,
    falls_back: bool,
) -> None:
    package = FakeFasterWhisper()
    monkeypatch.setattr(
        faster_whisper_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(WhisperModel=package.WhisperModel),
    )
    stt = FasterWhisperSTT.from_settings(make_settings(stt_device=device))

    if falls_back:
        await stt.transcribe(one_second_of_noise())
        assert package.devices == [("cuda", "int8_float16"), ("cpu", "int8")]
        assert package.models == ["large-v3-turbo", "small"]  # the CPU gets the CPU model
    else:
        with pytest.raises(RuntimeError, match="cublas64_12"):
            await stt.transcribe(one_second_of_noise())
        assert package.devices == [("cuda", "int8_float16")]
    assert registered == [1]  # the DLL directories are set up before the CUDA load


async def test_a_working_gpu_loads_the_gpu_model_in_int8_float16(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[..., Settings], registered: list[int]
) -> None:
    package = FakeFasterWhisper(cuda_works=True)
    install(monkeypatch, package)

    await FasterWhisperSTT.from_settings(make_settings()).warm_up()

    assert package.devices == [("cuda", "int8_float16")]
    assert package.models == ["large-v3-turbo"]
    assert registered == [1]


async def test_no_cuda_device_falls_back_to_cpu_without_trying_the_gpu(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: Callable[..., Settings],
    registered: list[int],
    logs: CapturingLogger,
) -> None:
    package = FakeFasterWhisper(cuda_devices=0, cuda_works=True)
    install(monkeypatch, package)

    await FasterWhisperSTT.from_settings(make_settings()).warm_up()

    assert package.devices == [("cpu", "int8")]
    assert package.models == ["small"]
    (warning,) = warnings(logs)
    assert warning["missing"] == "a CUDA device"


async def test_the_cpu_device_loads_the_cpu_model_without_cuda_setup(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[..., Settings], registered: list[int]
) -> None:
    package = FakeFasterWhisper()
    install(monkeypatch, package)

    await FasterWhisperSTT.from_settings(make_settings(stt_device="cpu")).warm_up()

    assert package.devices == [("cpu", "auto")]
    assert package.models == ["small"]
    assert registered == []


async def test_an_explicit_stt_model_is_used_on_every_device(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[..., Settings], registered: list[int]
) -> None:
    package = FakeFasterWhisper()
    install(monkeypatch, package)

    await FasterWhisperSTT.from_settings(make_settings(stt_model="medium")).warm_up()

    assert package.models == ["medium", "medium"]


async def test_hotwords_reach_faster_whisper(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[..., Settings], registered: list[int]
) -> None:
    model = FakeWhisperModel(info=FakeInfo(duration=0.5))
    install(monkeypatch, FakeFasterWhisper(cuda_works=True))
    monkeypatch.setattr(FakeFasterWhisper, "WhisperModel", lambda _self, _size, **_options: model)

    stt = FasterWhisperSTT.from_settings(make_settings())
    await stt.transcribe(one_second_of_noise())

    assert model.calls[0]["hotwords"] == "Jarvis, Discord, Gmail, Spotify"


async def test_empty_hotwords_disable_the_bias(model: FakeWhisperModel) -> None:
    stt = FasterWhisperSTT(lambda: model, hotwords="  ")

    await stt.transcribe(one_second_of_noise())

    assert model.calls[0]["hotwords"] is None
