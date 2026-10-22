import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from jarvis.config import Settings
from jarvis.tts import piper as piper_module
from jarvis.tts.base import TTS, AudioChunk, SynthesisError
from jarvis.tts.piper import PiperTTS, read_voice_sample_rate
from jarvis.utils.errors import ConfigError

RATE = 22_050


@dataclass(frozen=True)
class FakePiperChunk:
    audio_float_array: np.ndarray
    sample_rate: int = RATE
    sample_channels: int = 1


def sentence(value: float, samples: int = 100) -> FakePiperChunk:
    return FakePiperChunk(np.full(samples, value, dtype=np.float32))


@dataclass
class FakeSynthesizer:
    """A lazy per-sentence generator, like ``PiperVoice.synthesize``."""

    chunks: list[FakePiperChunk]
    texts: list[str] = field(default_factory=list)
    produced: int = 0
    failure: Exception | None = None

    def synthesize(self, text: str) -> Iterator[FakePiperChunk]:
        self.texts.append(text)
        for chunk in self.chunks:
            self.produced += 1
            yield chunk
        if self.failure is not None:
            raise self.failure


def tts_for(synthesizer: FakeSynthesizer) -> PiperTTS:
    return PiperTTS(lambda: synthesizer, voice="en_GB-test-medium", sample_rate=RATE)


async def collect(tts: TTS, text: str) -> list[AudioChunk]:
    return [chunk async for chunk in tts.synthesize(text)]


async def test_streams_one_chunk_per_sentence_at_the_voice_rate() -> None:
    synthesizer = FakeSynthesizer([sentence(0.1), sentence(0.2, 50)])
    tts = tts_for(synthesizer)

    chunks = await collect(tts, "One. Two.")

    assert isinstance(tts, TTS)
    assert tts.voice == "en_GB-test-medium"
    assert tts.sample_rate == RATE
    assert [chunk.samples.size for chunk in chunks] == [100, 50]
    assert all(chunk.samples.dtype == np.float32 for chunk in chunks)
    assert all(chunk.sample_rate == RATE for chunk in chunks)
    assert chunks[1].samples[0] == pytest.approx(0.2)
    assert synthesizer.texts == ["One. Two."]


async def test_first_chunk_is_yielded_before_the_next_is_synthesised() -> None:
    synthesizer = FakeSynthesizer([sentence(0.1), sentence(0.2)])

    stream = tts_for(synthesizer).synthesize("One. Two.")
    await anext(stream)

    assert synthesizer.produced == 1
    await stream.aclose()  # type: ignore[attr-defined]


async def test_blank_text_yields_nothing_and_loads_nothing() -> None:
    loads: list[int] = []

    def factory() -> FakeSynthesizer:
        loads.append(1)
        return FakeSynthesizer([])

    tts = PiperTTS(factory, voice="v", sample_rate=RATE)

    assert await collect(tts, "   ") == []
    assert not loads
    assert not tts.loaded


async def test_voice_loads_once() -> None:
    loads: list[int] = []
    synthesizer = FakeSynthesizer([sentence(0.1)])

    def factory() -> FakeSynthesizer:
        loads.append(1)
        return synthesizer

    tts = PiperTTS(factory, voice="v", sample_rate=RATE)
    await tts.warm_up()
    await collect(tts, "Hi.")
    await collect(tts, "Again.")

    assert tts.loaded
    assert loads == [1]


async def test_no_audio_is_an_error() -> None:
    with pytest.raises(SynthesisError, match="no audio"):
        await collect(tts_for(FakeSynthesizer([])), "Hello.")


@pytest.mark.parametrize(
    "chunk",
    [
        FakePiperChunk(np.zeros(10, dtype=np.float32), sample_rate=16_000),
        FakePiperChunk(np.zeros(10, dtype=np.float32), sample_channels=2),
    ],
)
async def test_unexpected_format_is_an_error(chunk: FakePiperChunk) -> None:
    with pytest.raises(SynthesisError, match="expected mono at 22050 Hz"):
        await collect(tts_for(FakeSynthesizer([chunk])), "Hello.")


async def test_piper_failure_mid_stream_becomes_synthesis_error() -> None:
    synthesizer = FakeSynthesizer([sentence(0.1)], failure=RuntimeError("onnx run failed"))

    with pytest.raises(SynthesisError, match="onnx run failed"):
        await collect(tts_for(synthesizer), "One. Two.")


async def test_voice_load_failure_becomes_synthesis_error() -> None:
    def broken() -> FakeSynthesizer:
        raise RuntimeError("bad onnx")

    tts = PiperTTS(broken, voice="v", sample_rate=RATE)

    with pytest.raises(SynthesisError, match="bad onnx"):
        await tts.warm_up()


async def test_config_errors_from_the_factory_pass_through() -> None:
    def unconfigured() -> FakeSynthesizer:
        raise ConfigError("install it")

    with pytest.raises(ConfigError, match="install it"):
        await PiperTTS(unconfigured, voice="v", sample_rate=RATE).warm_up()


# --- voice files and construction ---------------------------------------------------


class FakeLocale:
    """Stands in for ``locale``: one process-wide LC_CTYPE that espeak-ng overwrites."""

    LC_CTYPE = 0

    def __init__(self) -> None:
        self.current = "Swedish_Sweden.1252"

    def setlocale(self, _category: int, value: str | None = None) -> str:
        if value is not None:
            self.current = value
        return self.current


class LocaleChangingSynthesizer(FakeSynthesizer):
    """Changes the locale on every call, as espeak-ng does while phonemising."""

    def __init__(self, fake_locale: FakeLocale) -> None:
        super().__init__([sentence(0.1)])
        self._locale = fake_locale

    def synthesize(self, text: str) -> Iterator[FakePiperChunk]:
        self._locale.current = "en_US.UTF-8"
        yield from super().synthesize(text)


async def test_the_character_locale_survives_loading_and_speaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_locale = FakeLocale()
    monkeypatch.setattr(piper_module, "locale", fake_locale)

    def load() -> FakeSynthesizer:
        fake_locale.current = "en_US.UTF-8"
        return LocaleChangingSynthesizer(fake_locale)

    tts = PiperTTS(load, voice="sv_SE-test-medium", sample_rate=RATE)
    await tts.warm_up()
    assert fake_locale.current == "Swedish_Sweden.1252"

    await collect(tts, "Hej.")

    assert fake_locale.current == "Swedish_Sweden.1252"


def write_voice(directory: Path, config: Any) -> Path:
    voice = directory / "en_GB-test-medium.onnx"
    voice.write_bytes(b"onnx")
    text = config if isinstance(config, str) else json.dumps(config)
    (directory / "en_GB-test-medium.onnx.json").write_text(text, encoding="utf-8")
    return voice


def test_sample_rate_comes_from_the_voice_config(tmp_path: Path) -> None:
    voice = write_voice(tmp_path, {"audio": {"sample_rate": 16_000}})

    assert read_voice_sample_rate(voice) == 16_000


def test_missing_voice_or_config_names_the_remedy(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"download_models\.py --local"):
        read_voice_sample_rate(tmp_path / "absent.onnx")
    voice = tmp_path / "lonely.onnx"
    voice.write_bytes(b"onnx")
    with pytest.raises(ConfigError, match=r"lonely\.onnx\.json"):
        read_voice_sample_rate(voice)


@pytest.mark.parametrize(
    "config",
    ["not json", {"audio": {}}, {"audio": {"sample_rate": "fast"}}, {"audio": {"sample_rate": 0}}],
)
def test_malformed_config_is_a_config_error(tmp_path: Path, config: Any) -> None:
    voice = write_voice(tmp_path, config)

    with pytest.raises(ConfigError, match=r"Piper voice config|sample_rate"):
        read_voice_sample_rate(voice)


def test_missing_package_names_the_extra(
    make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    voice = write_voice(tmp_path, {"audio": {"sample_rate": RATE}})
    real_import = piper_module.importlib.import_module

    def without_piper(name: str, package: str | None = None) -> ModuleType:
        if name == "piper":
            raise ModuleNotFoundError("No module named 'piper'")
        return real_import(name, package)

    monkeypatch.setattr(piper_module.importlib, "import_module", without_piper)

    with pytest.raises(ConfigError, match="uv sync --extra local"):
        PiperTTS.from_settings(make_settings(local_tts_model=voice))


async def test_from_settings_loads_the_configured_voice_lazily(
    make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    voice = write_voice(tmp_path, {"audio": {"sample_rate": RATE}})
    loaded: list[Path] = []
    synthesizer = FakeSynthesizer([sentence(0.3)])

    def load(path: Path) -> FakeSynthesizer:
        loaded.append(path)
        return synthesizer

    fake_piper = SimpleNamespace(PiperVoice=SimpleNamespace(load=load))
    real_import = piper_module.importlib.import_module

    def with_fake_piper(name: str, package: str | None = None) -> Any:
        return fake_piper if name == "piper" else real_import(name, package)

    monkeypatch.setattr(piper_module.importlib, "import_module", with_fake_piper)

    tts = PiperTTS.from_settings(make_settings(local_tts_model=voice))
    assert not loaded
    chunks = await collect(tts, "Hello.")

    assert loaded == [voice]
    assert tts.voice == "en_GB-test-medium"
    assert tts.sample_rate == RATE
    assert len(chunks) == 1
