import pytest

from jarvis.utils.errors import ConfigError, JarvisError, NativeUnavailableError


@pytest.mark.parametrize("error_type", [ConfigError, NativeUnavailableError])
def test_every_error_derives_from_jarvis_error(error_type: type[Exception]) -> None:
    assert issubclass(error_type, JarvisError)


def test_jarvis_error_does_not_swallow_builtin_errors() -> None:
    assert not issubclass(KeyError, JarvisError)


def test_native_unavailable_error_carries_context() -> None:
    error = NativeUnavailableError("jarvis_audio", "not built")

    assert error.module == "jarvis_audio"
    assert error.reason == "not built"
    assert "jarvis_audio" in str(error)
