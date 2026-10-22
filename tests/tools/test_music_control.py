from collections.abc import Callable

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool import ToolContext
from jarvis.config import Settings
from jarvis.integrations.base import NowPlaying
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubMusicProvider
from jarvis.tools.music_control import MusicControlTool
from tests.tools.integration_helpers import integration_context


@pytest.fixture
def music() -> StubMusicProvider:
    provider = StubMusicProvider()
    provider.current = NowPlaying(provider.catalogue[1], is_playing=True)
    return provider


@pytest.fixture
def context(make_settings: Callable[..., Settings], music: StubMusicProvider) -> ToolContext:
    return integration_context(make_settings(), IntegrationServices(music=(music,)))


def test_is_discovered_and_never_confirms() -> None:
    tool = ToolRegistry.discover().get("music_control")

    assert isinstance(tool, MusicControlTool)
    assert not tool.requires_confirmation


async def test_pause_and_resume(context: ToolContext, music: StubMusicProvider) -> None:
    paused = await MusicControlTool().call({"action": "pause"}, context)
    assert music.current is not None
    assert not music.current.is_playing

    resumed = await MusicControlTool().call({"action": "resume"}, context)

    assert paused.spoken == "Paused."
    assert resumed.spoken == "Resuming the music."
    assert music.current.is_playing


async def test_play_without_a_query_resumes(context: ToolContext, music: StubMusicProvider) -> None:
    result = await MusicControlTool().call({"action": "play"}, context)

    assert result.data == {"action": "play", "provider": "stub-music"}
    assert music.calls == [("resume", {})]


async def test_next_reports_the_new_track_or_the_end(
    context: ToolContext, music: StubMusicProvider
) -> None:
    music.queued.append(music.catalogue[2])

    skipped = await MusicControlTool().call({"action": "next"}, context)
    skipped_again = await MusicControlTool().call({"action": "next"}, context)

    assert skipped.spoken == "Now playing Weightless by Marconi Union."
    assert skipped_again.spoken == "Skipped; nothing else is lined up."


async def test_unknown_action_is_rejected(context: ToolContext) -> None:
    result = await MusicControlTool().call({"action": "shuffle"}, context)

    assert not result.ok
    assert "action" in result.detail
