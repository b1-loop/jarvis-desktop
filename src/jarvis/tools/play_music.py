"""Tool: search for music and start playing the best match.

"Play some Miles Davis" or "play Clair de Lune". The provider resolves a
free-text query (or a provider URI) to the best hit and starts it on the
user's active device. No confirmation, for the reasons in
``music_control``'s design notes.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.tools._integration_support import Service, music_providers, not_configured, track_item

MAX_QUERY_CHARS: Final = 200


class PlayMusicInput(ToolInput):
    """Arguments for :class:`PlayMusicTool`."""

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="What to play: a song, artist, album or genre in plain words, or a "
        "provider URI such as 'spotify:track:...'.",
    )


class PlayMusicTool(Tool[PlayMusicInput]):
    """Starts playback of the best match for a query."""

    name = "play_music"
    description = (
        "Search for music and start playing the best match on the user's active Spotify "
        "device, replacing what is playing. Use for 'play <song/artist/album>'. Returns the "
        "track that started. To add to the queue instead use queue_music."
    )
    Input = PlayMusicInput
    category = ToolCategory.MEDIA

    async def run(self, params: PlayMusicInput, context: ToolContext) -> ToolResult:
        """Ask the provider to play the query and report the track."""
        providers = music_providers(context)
        if not providers:
            return not_configured(Service.MUSIC)
        provider = providers[0]
        track = await provider.play(params.query)
        if track is None:
            return ToolResult.success(
                f"Playing {params.query}.",
                detail=f"{provider.provider_name} started a context for {params.query!r} "
                "without naming the first track.",
                data={"query": params.query, "track": None},
            )
        return ToolResult.success(
            f"Playing {track.title} by {track.artist}.",
            detail=f"Started {track.uri} on {provider.provider_name}: {track.title} - "
            f"{track.artist} ({track.album})",
            data={"query": params.query, "track": track_item(provider, track)},
        )


# Design notes:
# - A playlist or album URI may start without the provider saying which
#   track plays first (MusicProvider.play returns None); the spoken answer
#   then echoes the request instead of guessing.
