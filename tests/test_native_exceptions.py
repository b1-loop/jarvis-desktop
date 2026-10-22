"""Each native module must translate C++ errors into its own Python classes.

pybind11 exception translators registered with ``register_exception`` are
process-global, so the module imported last used to decide which Python
class every ``jarvis::NativeError`` became. The modules now register
module-local translators; these tests import them in different orders and
check that an error raised inside ``jarvis_audio`` is still ``jarvis_audio``'s
``DecodeError``.
"""

from __future__ import annotations

import subprocess
import sys
from itertools import permutations

import pytest

from jarvis import _native

pytestmark = pytest.mark.native

PROBE = """
import importlib, sys
order = sys.argv[1].split(",")
modules = {name: importlib.import_module(f"jarvis._native.{name}") for name in order}
audio = modules["jarvis_audio"]
try:
    audio.decode_mp3(b"not an mp3 at all")
except audio.DecodeError:
    print("ok")
except Exception as error:
    print(f"wrong:{type(error).__module__}.{type(error).__name__}")
"""


def _available() -> tuple[str, ...]:
    names = _native.available_modules()
    if "jarvis_audio" not in names or len(names) < 2:
        pytest.skip("needs jarvis_audio and at least one other native module")
    return names


@pytest.mark.parametrize("rotation", [0, 1, 2])
def test_decode_error_survives_any_import_order(rotation: int) -> None:
    names = _available()
    orders = list(permutations(names))
    order = orders[(rotation * len(orders)) // 3]
    # A fresh interpreter per order: translators are registered at import time.
    result = subprocess.run(  # noqa: S603 - fixed interpreter and inline probe
        [sys.executable, "-c", PROBE, ",".join(order)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.stdout.strip() == "ok", (order, result.stdout, result.stderr)
