"""Tests against the compiled ``jarvis_inference`` module.

All of them need the native build; the model-backed ones additionally need
``scripts/download_models.py --local`` to have run and skip otherwise.
"""

from types import ModuleType

import numpy as np
import pytest

from jarvis import _native
from jarvis.config import Settings
from jarvis.inference.models import InferenceModelPaths
from jarvis.utils.errors import NativeUnavailableError

pytestmark = pytest.mark.native

SAY_HELLO = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "Say hello"},
]


@pytest.fixture(scope="module")
def jarvis_inference() -> ModuleType:
    try:
        module = _native.load("jarvis_inference")
    except NativeUnavailableError:
        pytest.skip("jarvis_inference is not built")
    if "disabled" in module.IMPLEMENTATION_STATUS:
        pytest.skip("native build was configured with JARVIS_BUILD_INFERENCE=OFF")
    return module


@pytest.fixture(scope="module")
def model_paths() -> InferenceModelPaths:
    paths = InferenceModelPaths.in_directory(Settings(_env_file=None).models_dir)
    if paths.missing():
        pytest.skip("inference models not downloaded; run scripts/download_models.py --local")
    return paths


@pytest.fixture(scope="module")
def whisper(jarvis_inference: ModuleType, model_paths: InferenceModelPaths) -> object:
    """One loaded whisper model per test module; loading dominates the test time."""
    return jarvis_inference.WhisperTranscriber(model_paths.whisper)


@pytest.fixture(scope="module")
def llama(jarvis_inference: ModuleType, model_paths: InferenceModelPaths) -> object:
    return jarvis_inference.LlamaGenerator(model_paths.llama, n_ctx=512, n_threads=2)


def test_module_exposes_versions_and_status(jarvis_inference: ModuleType) -> None:
    assert jarvis_inference.WHISPER_SAMPLE_RATE == 16_000
    assert jarvis_inference.WHISPER_VERSION
    assert jarvis_inference.LLAMA_VERSION
    assert "step 12" in jarvis_inference.IMPLEMENTATION_STATUS


def test_exceptions_form_a_hierarchy(jarvis_inference: ModuleType) -> None:
    assert issubclass(jarvis_inference.ModelError, jarvis_inference.NativeError)
    assert issubclass(jarvis_inference.NativeError, RuntimeError)


def test_missing_whisper_model_raises_model_error(
    jarvis_inference: ModuleType, tmp_path: object
) -> None:
    with pytest.raises(jarvis_inference.ModelError, match="not found"):
        jarvis_inference.WhisperTranscriber(f"{tmp_path}/nope.bin")


def test_missing_gguf_raises_model_error(jarvis_inference: ModuleType, tmp_path: object) -> None:
    with pytest.raises(jarvis_inference.ModelError, match="not found"):
        jarvis_inference.LlamaGenerator(f"{tmp_path}/nope.gguf")


def test_transcribe_rejects_non_mono_input(jarvis_inference: ModuleType, whisper: object) -> None:
    stereo = np.zeros((2, 16_000), dtype=np.float32)
    with pytest.raises(ValueError, match="one-dimensional"):
        whisper.transcribe(stereo)  # type: ignore[attr-defined]


def test_silence_transcribes_to_near_empty_text(whisper: object) -> None:
    silence = np.zeros(16_000, dtype=np.float32)

    transcript = whisper.transcribe(silence, language="en")  # type: ignore[attr-defined]

    assert len(transcript.text) <= 32
    assert transcript.language == "en"
    assert all(segment.end_ms >= segment.start_ms for segment in transcript.segments)


def test_generation_streams_tokens_through_the_iterator_protocol(llama: object) -> None:
    stream = llama.generate(SAY_HELLO, max_tokens=8, temperature=0.0)  # type: ignore[attr-defined]

    pieces = list(stream)

    assert pieces
    assert "".join(pieces).strip()
    assert stream.finished()
    assert stream.finish_reason() in {"max_tokens", "end_of_text"}
    assert 1 <= stream.generated_tokens() <= 8
    assert stream.next_token() is None


def test_cancel_stops_generation_early(llama: object) -> None:
    stream = llama.generate(SAY_HELLO, max_tokens=200)  # type: ignore[attr-defined]

    first = stream.next_token()
    stream.cancel()
    rest = list(stream)

    assert first is not None
    assert len(rest) < 200
    assert stream.finish_reason() == "cancelled"


def test_messages_need_role_and_content(jarvis_inference: ModuleType, llama: object) -> None:
    with pytest.raises(ValueError, match="role"):
        llama.format_prompt([{"content": "no role"}])  # type: ignore[attr-defined]


def test_prompt_uses_the_models_chat_template(llama: object) -> None:
    prompt = llama.format_prompt(SAY_HELLO)  # type: ignore[attr-defined]

    assert "<|im_start|>user\nSay hello<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert llama.count_tokens(prompt) > 5  # type: ignore[attr-defined]
    assert llama.context_size() == 512  # type: ignore[attr-defined]
