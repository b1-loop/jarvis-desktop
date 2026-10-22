from collections.abc import Callable

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.stubs import StubMusicProvider
from jarvis.tools.queue_music import QueueMusicTool
from tests.tools.integration_helpers import integration_context


def test_is_discovered_and_never_confirms() -> None:
    tool = ToolRegistry.discover().get("queue_music")

    assert isinstance(tool, QueueMusicTool)
    assert not tool.requires_confirmation


async def test_queues_the_best_match(make_settings: Callable[..., Settings]) -> None:
    music = StubMusicProvider()
    context = integration_context(make_settings(), IntegrationServices(music=(music,)))

    result = await QueueMusicTool().call({"query": "weightless"}, context)

    assert result.ok
    assert result.spoken == "Added Weightless by Marconi Union to the queue."
    assert [track.title for track in music.queued] == ["Weightless"]
    assert result.data is not None
    assert result.data["track"]["artist"] == "Marconi Union"


async def test_empty_query_is_rejected(make_settings: Callable[..., Settings]) -> None:
    context = integration_context(
        make_settings(), IntegrationServices(music=(StubMusicProvider(),))
    )

    result = await QueueMusicTool().call({"query": ""}, context)

    assert not result.ok
