from types import ModuleType, SimpleNamespace

import pytest

from jarvis import _native
from jarvis.os_integration import audio_endpoint
from jarvis.os_integration.audio_endpoint import default_output_endpoint
from jarvis.os_integration.base import OutputEndpoint
from jarvis.utils.errors import NativeUnavailableError

KNOWN_FORM_FACTORS = {
    "remote_network_device",
    "speakers",
    "line_level",
    "headphones",
    "microphone",
    "headset",
    "handset",
    "unknown_digital_passthrough",
    "spdif",
    "digital_audio_display_device",
    "unknown",
}


def fake_module(monkeypatch: pytest.MonkeyPatch, module: object) -> None:
    def load(_name: str) -> object:
        if isinstance(module, Exception):
            raise module
        return module

    monkeypatch.setattr(audio_endpoint._native, "load", load)


def test_the_native_endpoint_is_converted(monkeypatch: pytest.MonkeyPatch) -> None:
    native = SimpleNamespace(name="Speakers (Realtek(R) Audio)", form_factor="speakers")
    fake_module(monkeypatch, SimpleNamespace(default_output_endpoint=lambda: native))

    assert default_output_endpoint() == OutputEndpoint("Speakers (Realtek(R) Audio)", "speakers")


def test_no_device_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module(monkeypatch, SimpleNamespace(default_output_endpoint=lambda: None))

    assert default_output_endpoint() is None


def test_a_missing_build_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module(monkeypatch, NativeUnavailableError("jarvis_os", "not built"))

    assert default_output_endpoint() is None


def test_an_older_build_without_the_query_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module(monkeypatch, SimpleNamespace())

    assert default_output_endpoint() is None


@pytest.fixture
def jarvis_os() -> ModuleType:
    try:
        return _native.load("jarvis_os")
    except NativeUnavailableError:
        pytest.skip("jarvis_os is not built")


@pytest.mark.native
def test_the_real_default_output_is_read_without_side_effects(jarvis_os: ModuleType) -> None:
    endpoint = jarvis_os.default_output_endpoint()
    if endpoint is None:
        pytest.skip("no audio output device")

    assert endpoint.form_factor in KNOWN_FORM_FACTORS
    assert isinstance(endpoint.name, str)
    assert default_output_endpoint() == OutputEndpoint(endpoint.name, endpoint.form_factor)
