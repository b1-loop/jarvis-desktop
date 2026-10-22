"""Tool: add the best match for a query to the playback queue.

"Queue Blue in Green after this." The queue is session playback state on
the user's own device, not their library, so it needs no confirmation
(see ``music_control``'s design notes).
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._integration_support import Service, music_providers, not_configured, track_item

MAX_QUERY_CHARS: Final = 200


class QueueMusicInput(ToolInput):
    """Arguments for :class:`QueueMusicTool`."""

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="Song (optionally with artist) to add after the current track.",
    )


class QueueMusicTool(Tool[QueueMusicInput]):
    """Appends a track to the playback queue."""

    name = "queue_music"
    description = (
        "Add a song to the playback queue on the user's active Spotify device so it plays "
        "after the current one. Use for 'queue <song>' or 'play <song> next'. Returns the "
        "track that was queued."
    )
    Input = QueueMusicInput
    category = ToolCategory.MEDIA
    keywords = ("kö", "kön", "köa")

    async def run(self, params: QueueMusicInput, context: ToolContext) -> ToolResult:
        """Queue the best hit and say what it was."""
        providers = music_providers(context)
        if not providers:
            return not_configured(Service.MUSIC)
        provider = providers[0]
        track = await provider.queue(params.query)
        return ToolResult.success(
            f"Added {track.title} by {track.artist} to the queue.",
            detail=f"Queued {track.uri} on {provider.provider_name}.",
            data={"query": params.query, "track": track_item(provider, track)},
        )
