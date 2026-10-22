from pathlib import Path

from PIL import Image

from jarvis.app.icon import ICON_PATH, ICON_SIZES, write_app_icon
from jarvis.core.state_machine import AssistantState
from jarvis.ui.tray import render_state_icon


def _frame(path: Path, size: int) -> Image.Image:
    with Image.open(path) as icon:
        icon.size = (size, size)
        icon.load()
        return icon.convert("RGBA")


def test_icon_has_every_size_from_16_to_256(tmp_path: Path) -> None:
    path = write_app_icon(tmp_path / "assets" / "jarvis.ico")

    with Image.open(path) as icon:
        assert sorted(icon.info["sizes"]) == [(size, size) for size in ICON_SIZES]
    assert min(ICON_SIZES) == 16
    assert max(ICON_SIZES) == 256


def test_each_size_is_the_tray_artwork_drawn_natively(tmp_path: Path) -> None:
    path = write_app_icon(tmp_path / "jarvis.ico")

    for size in (16, 32, 256):
        expected = render_state_icon(AssistantState.IDLE, size)
        assert _frame(path, size).tobytes() == expected.tobytes()


def test_committed_icon_is_up_to_date() -> None:
    for size in ICON_SIZES:
        expected = render_state_icon(AssistantState.IDLE, size)
        assert _frame(ICON_PATH, size).tobytes() == expected.tobytes(), size
