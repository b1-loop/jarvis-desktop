from collections.abc import Callable

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.base import Track
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubMusicProvider
from jarvis.tools.play_music import PlayMusicTool
from tests.tools.integration_helpers import integration_context


class ContextOnlyMusic(StubMusicProvider):
    """Starts playlists without naming a track, as Spotify may."""

    async def play(self, query_or_uri: str) -> Track | None:  # type: ignore[override]
        self._record("play", query_or_uri=query_or_uri)
        return None


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered_and_never_confirms() -> None:
    tool = ToolRegistry.discover().get("play_music")

    assert isinstance(tool, PlayMusicTool)
    assert not tool.requires_confirmation


async def test_plays_the_best_match(settings: Settings) -> None:
    music = StubMusicProvider()
    context = integration_context(settings, IntegrationServices(music=(music,)))

    result = await PlayMusicTool().call({"query": "debussy"}, context)

    assert result.spoken == "Playing Clair de Lune by Claude Debussy."
    assert result.data is not None
    assert result.data["track"]["uri"] == "stub:track:1"
    assert music.current is not None
    assert music.current.is_playing


async def test_play_without_a_named_track(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(music=(ContextOnlyMusic(),)))

    result = await PlayMusicTool().call({"query": "spotify:playlist:focus"}, context)

    assert result.spoken == "Playing spotify:playlist:focus."
    assert result.data == {"query": "spotify:playlist:focus", "track": None}


async def test_no_match_is_a_failed_result(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(music=(StubMusicProvider(),)))

    result = await PlayMusicTool().call({"query": "nonexistent band"}, context)

    assert not result.ok
    assert "nothing matches" in result.detail
