import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from jarvis import _native
from jarvis.config import RuntimeMode, Settings
from jarvis.core import providers
from jarvis.core.providers import (
    DependencyProbe,
    build_llm,
    build_stt,
    build_tts,
    effective_tts_provider,
)
from jarvis.llm.anthropic_client import AnthropicClient
from jarvis.llm.local_llama_client import LocalLlamaClient
from jarvis.llm.ollama_client import OllamaClient
from jarvis.stt.faster_whisper import FasterWhisperSTT
from jarvis.stt.whisper_cpp import WhisperCppSTT
from jarvis.tts import piper as piper_module
from jarvis.tts.edge_tts import EdgeTTS
from jarvis.tts.piper import PiperTTS
from jarvis.utils.errors import ConfigError, NativeUnavailableError

EVERYTHING_INSTALLED = DependencyProbe(
    module_installed=lambda _: True, native_problem=lambda _: None
)
INFERENCE_MODULE = SimpleNamespace(IMPLEMENTATION_STATUS="implemented in step 12")


def probe(*, missing: tuple[str, ...] = (), native_problem: str | None = None) -> DependencyProbe:
    return DependencyProbe(
        module_installed=lambda name: name not in missing,
        native_problem=lambda _: native_problem,
    )


@pytest.fixture
def native_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend jarvis_inference is built; nothing loads weights in these tests."""
    monkeypatch.setattr(_native, "load", lambda _name: INFERENCE_MODULE)


@pytest.fixture
def native_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(name: str) -> ModuleType:
        raise NativeUnavailableError(name, "No module named 'jarvis_inference'")

    monkeypatch.setattr(_native, "load", unavailable)


@pytest.fixture
def piper_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_piper = SimpleNamespace(PiperVoice=SimpleNamespace(load=lambda path: path))
    real_import = piper_module.importlib.import_module

    def importing(name: str, package: str | None = None) -> Any:
        return fake_piper if name == "piper" else real_import(name, package)

    monkeypatch.setattr(piper_module.importlib, "import_module", importing)


@pytest.fixture
def local_settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    """LOCAL mode with every model file present (as placeholders)."""
    for name in ("ggml.bin", "model.gguf", "voice.onnx"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "voice.onnx.json").write_text(json.dumps({"audio": {"sample_rate": 22_050}}))
    return make_settings(
        mode=RuntimeMode.LOCAL,
        local_stt_model=tmp_path / "ggml.bin",
        local_llm_model=tmp_path / "model.gguf",
        local_tts_model=tmp_path / "voice.onnx",
    )


# --- CLOUD mode -------------------------------------------------------------


def test_cloud_mode_uses_faster_whisper_with_the_configured_language(
    make_settings: Callable[..., Settings],
) -> None:
    stt = build_stt(make_settings(stt_language="en"), probe=EVERYTHING_INSTALLED)

    assert isinstance(stt, FasterWhisperSTT)
    assert not stt.loaded


def test_cloud_stt_without_faster_whisper_names_the_extra(
    make_settings: Callable[..., Settings],
) -> None:
    with pytest.raises(ConfigError, match=r"faster-whisper.*uv sync --extra cloud"):
        build_stt(make_settings(), probe=probe(missing=("faster_whisper",)))


def test_cloud_mode_uses_edge_tts(make_settings: Callable[..., Settings]) -> None:
    tts = build_tts(make_settings(tts_voice="en-GB-SoniaNeural"), probe=EVERYTHING_INSTALLED)

    assert isinstance(tts, EdgeTTS)
    assert tts.voice == "en-GB-SoniaNeural"


def test_cloud_tts_without_edge_tts_names_the_extra(
    make_settings: Callable[..., Settings],
) -> None:
    with pytest.raises(ConfigError, match=r"edge-tts.*uv sync --extra cloud"):
        build_tts(make_settings(), probe=probe(missing=("edge_tts",)))


def test_cloud_tts_without_the_native_decoder_names_the_build(
    make_settings: Callable[..., Settings],
) -> None:
    with pytest.raises(ConfigError, match=r"DLL missing.*build_native\.py"):
        build_tts(make_settings(), probe=probe(native_problem="DLL missing"))


def test_cloud_mode_honours_the_llm_provider(make_settings: Callable[..., Settings]) -> None:
    anthropic = make_settings(llm_provider="anthropic", anthropic_api_key="sk-test")

    assert isinstance(build_llm(anthropic), AnthropicClient)
    assert isinstance(build_llm(make_settings(llm_provider="ollama")), OllamaClient)


def test_cloud_mode_can_pair_cloud_speech_with_llama_cpp(
    make_settings: Callable[..., Settings], local_settings: Settings, native_built: None
) -> None:
    settings = make_settings(llm_provider="local", local_llm_model=local_settings.local_llm_model)

    assert isinstance(build_llm(settings), LocalLlamaClient)


def test_cloud_anthropic_without_a_key_explains_the_remedy(
    make_settings: Callable[..., Settings],
) -> None:
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        build_llm(make_settings(llm_provider="anthropic"))


def test_cloud_mode_speaks_with_piper_when_asked(
    local_settings: Settings, piper_installed: None
) -> None:
    hybrid = local_settings.model_copy(update={"mode": RuntimeMode.CLOUD, "tts_provider": "piper"})

    tts = build_tts(hybrid, probe=probe(missing=("edge_tts",)))

    assert isinstance(tts, PiperTTS)
    assert effective_tts_provider(hybrid) == "piper"


@pytest.mark.parametrize(
    ("mode", "configured", "expected"),
    [
        (RuntimeMode.CLOUD, "auto", "edge"),
        (RuntimeMode.LOCAL, "auto", "piper"),
        (RuntimeMode.LOCAL, "edge", "edge"),
        (RuntimeMode.CLOUD, "piper", "piper"),
    ],
)
def test_effective_tts_provider_follows_the_mode_only_for_auto(
    make_settings: Callable[..., Settings], mode: RuntimeMode, configured: str, expected: str
) -> None:
    settings = make_settings(mode=mode, tts_provider=configured)

    assert effective_tts_provider(settings) == expected


# --- LOCAL mode -------------------------------------------------------------


def test_local_mode_builds_the_offline_stack(
    local_settings: Settings, native_built: None, piper_installed: None
) -> None:
    stt = build_stt(local_settings, probe=probe(missing=("faster_whisper", "edge_tts")))
    tts = build_tts(local_settings, probe=probe(missing=("faster_whisper", "edge_tts")))
    llm = build_llm(local_settings)

    assert isinstance(stt, WhisperCppSTT)
    assert isinstance(tts, PiperTTS)
    assert tts.sample_rate == 22_050
    assert isinstance(llm, LocalLlamaClient)
    assert llm.model == "model.gguf"


def test_local_mode_replaces_a_cloud_llm_provider_with_llama_cpp(
    local_settings: Settings, native_built: None
) -> None:
    settings = local_settings.model_copy(
        update={"llm_provider": "anthropic", "anthropic_api_key": None}
    )

    assert isinstance(build_llm(settings), LocalLlamaClient)
    assert settings.llm_provider == "anthropic"  # the caller's settings are untouched


def test_local_mode_keeps_an_explicit_ollama_provider(local_settings: Settings) -> None:
    settings = local_settings.model_copy(update={"llm_provider": "ollama"})

    assert isinstance(build_llm(settings), OllamaClient)


def test_local_mode_uses_llama_cpp_when_ollama_is_only_the_default(
    local_settings: Settings, native_built: None
) -> None:
    assert local_settings.llm_provider == "ollama"
    assert "llm_provider" not in local_settings.model_fields_set

    assert isinstance(build_llm(local_settings), LocalLlamaClient)


@pytest.mark.parametrize(
    ("field", "builder", "setting"),
    [
        ("local_stt_model", build_stt, "JARVIS_LOCAL_STT_MODEL"),
        ("local_llm_model", build_llm, "JARVIS_LOCAL_LLM_MODEL"),
        ("local_tts_model", build_tts, "JARVIS_LOCAL_TTS_MODEL"),
    ],
)
@pytest.mark.usefixtures("native_built", "piper_installed")
def test_local_mode_missing_model_names_the_download_and_the_setting(
    local_settings: Settings,
    tmp_path: Path,
    field: str,
    builder: Callable[[Settings], object],
    setting: str,
) -> None:
    settings = local_settings.model_copy(update={field: tmp_path / "absent" / "model"})

    with pytest.raises(ConfigError, match=rf"download_models\.py --local.*{setting}"):
        builder(settings)


@pytest.mark.parametrize("builder", [build_stt, build_llm])
def test_local_mode_without_the_native_build_names_the_build_command(
    local_settings: Settings, native_missing: None, builder: Callable[[Settings], object]
) -> None:
    with pytest.raises(ConfigError, match=r"jarvis_inference.*build_native\.py"):
        builder(local_settings)


def test_local_mode_without_piper_names_the_extra(
    local_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_piper(name: str, package: str | None = None) -> ModuleType:
        raise ModuleNotFoundError(f"No module named '{name}'")

    monkeypatch.setattr(piper_module.importlib, "import_module", no_piper)

    with pytest.raises(ConfigError, match="uv sync --extra local"):
        build_tts(local_settings)


# --- default probe ------------------------------------------------------------


def test_default_probe_checks_packages_without_importing_them() -> None:
    assert providers.DEFAULT_PROBE.module_installed("json")
    assert not providers.DEFAULT_PROBE.module_installed("jarvis_no_such_package")


def test_default_probe_reports_why_a_native_module_is_unavailable(
    native_missing: None,
) -> None:
    assert providers.DEFAULT_PROBE.native_problem("jarvis_audio") == (
        "No module named 'jarvis_inference'"
    )


def test_default_probe_reports_nothing_for_a_loadable_module(native_built: None) -> None:
    assert providers.DEFAULT_PROBE.native_problem("jarvis_audio") is None
