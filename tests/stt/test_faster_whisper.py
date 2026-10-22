import math
from collections.abc import Callable

import numpy as np
import pytest

from jarvis.config import Settings
from jarvis.stt.base import ENGINE_SAMPLE_RATE, STT, Transcript, TranscriptionError
from jarvis.stt.faster_whisper import FasterWhisperSTT, resample_linear
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
        {"samples": 16_000, "language": None, "beam_size": 5, "vad_filter": False}
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


async def test_warm_up_loads_the_model_ahead_of_time(model: FakeWhisperModel) -> None:
    stt = FasterWhisperSTT(lambda: model)

    await stt.warm_up()

    assert stt.loaded
    assert model.calls == []


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


async def test_language_and_beam_size_are_forwarded(model: FakeWhisperModel) -> None:
    stt = FasterWhisperSTT(lambda: model, language="bg", beam_size=2, vad_filter=True)

    await stt.transcribe(one_second_of_noise())

    assert model.calls[0]["language"] == "bg"
    assert model.calls[0]["beam_size"] == 2
    assert model.calls[0]["vad_filter"] is True


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
