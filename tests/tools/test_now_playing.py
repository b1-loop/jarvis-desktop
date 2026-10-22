from collections.abc import Callable

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.base import NowPlaying
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubMusicProvider
from jarvis.tools.now_playing import NowPlayingTool
from tests.tools.integration_helpers import integration_context


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered_and_read_only() -> None:
    tool = ToolRegistry.discover().get("now_playing")

    assert isinstance(tool, NowPlayingTool)
    assert not tool.requires_confirmation


async def test_playing_with_progress(settings: Settings) -> None:
    music = StubMusicProvider()
    music.current = NowPlaying(music.catalogue[0], is_playing=True, progress_ms=83_000)
    context = integration_context(settings, IntegrationServices(music=(music,)))

    result = await NowPlayingTool().call({}, context)

    assert result.spoken == "Now playing Clair de Lune by Claude Debussy."
    assert "(Suite bergamasque) at 1:23 on stub-music" in result.detail
    assert result.data is not None
    assert result.data["playing"] is True
    assert result.data["progress_ms"] == 83_000


async def test_paused(settings: Settings) -> None:
    music = StubMusicProvider()
    music.current = NowPlaying(music.catalogue[1], is_playing=False)
    context = integration_context(settings, IntegrationServices(music=(music,)))

    result = await NowPlayingTool().call({}, context)

    assert result.spoken == "Paused on Blue in Green by Miles Davis."


async def test_nothing_playing(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(music=(StubMusicProvider(),)))

    result = await NowPlayingTool().call({}, context)

    assert result.spoken == "Nothing is playing right now."
    assert result.data == {"playing": False, "track": None}


async def test_takes_no_arguments(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(music=(StubMusicProvider(),)))

    result = await NowPlayingTool().call({"verbose": True}, context)

    assert not result.ok
