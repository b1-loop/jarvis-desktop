from collections.abc import Callable

from jarvis.config import Settings
from jarvis.os_integration import StubPlatform
from jarvis.tools.type_text import TypeTextInput, TypeTextTool
from tests.tools.local_fakes import CODE, local_context


def test_confirmation_prompt_previews_the_text() -> None:
    tool = TypeTextTool()

    short = tool.confirmation_prompt(TypeTextInput(text="see you\nat five", press_enter=True))
    long = tool.confirmation_prompt(TypeTextInput(text="word " * 40))

    assert tool.requires_confirmation
    assert (
        short == "Do you want me to type 'see you at five' and press Enter into the focused window?"
    )
    assert "…'" in long
    assert len(long) < 140


async def test_types_into_the_focused_window(make_settings: Callable[..., Settings]) -> None:
    platform = StubPlatform(windows=[CODE], active=CODE)

    result = await TypeTextTool().call(
        {"text": "hello", "press_enter": True}, local_context(make_settings(), platform=platform)
    )

    assert result.ok
    assert result.spoken == f"Typed it into {CODE.title}."
    assert platform.typed == ["hello\n"]
    assert result.data == {"characters": 5, "press_enter": True, "window": CODE.title}
    assert "hello" not in result.detail


async def test_types_even_without_a_known_focused_window(
    make_settings: Callable[..., Settings],
) -> None:
    platform = StubPlatform()

    result = await TypeTextTool().call(
        {"text": "x"}, local_context(make_settings(), platform=platform)
    )

    assert result.spoken == "Typed it."
    assert platform.typed == ["x"]
