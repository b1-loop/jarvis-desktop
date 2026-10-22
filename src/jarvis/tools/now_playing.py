"""Tool: what is playing right now.

"What's this song?" Reads the playback state of the user's active player
and says the track, the artist and whether it is paused.
"""

from __future__ import annotations

from typing import Final

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.base import MusicProvider, NowPlaying
from jarvis.tools._integration_support import Service, music_providers, not_configured, track_item

_MS_PER_SECOND: Final = 1000
_SECONDS_PER_MINUTE: Final = 60


class NowPlayingInput(ToolInput):
    """No arguments."""


class NowPlayingTool(Tool[NowPlayingInput]):
    """Reports the current track and playback state."""

    name = "now_playing"
    description = (
        "Tell what music is playing right now (track, artist, album, position, paused or "
        "playing). Use for 'what song is this?' or 'is anything playing?'."
    )
    Input = NowPlayingInput
    category = ToolCategory.MEDIA

    async def run(self, params: NowPlayingInput, context: ToolContext) -> ToolResult:
        """Read playback state from the first music provider."""
        del params  # no arguments
        providers = music_providers(context)
        if not providers:
            return not_configured(Service.MUSIC)
        provider = providers[0]
        current = await provider.now_playing()
        if current is None:
            return ToolResult.success(
                "Nothing is playing right now.", data={"playing": False, "track": None}
            )
        return _describe(provider, current)


def _describe(provider: MusicProvider, current: NowPlaying) -> ToolResult:
    track = current.track
    state = "Now playing" if current.is_playing else "Paused on"
    position = _clock(current.progress_ms)
    return ToolResult.success(
        f"{state} {track.title} by {track.artist}.",
        detail=f"{state} {track.title} - {track.artist} ({track.album}) at {position} "
        f"on {provider.provider_name}.",
        data={
            "playing": current.is_playing,
            "progress_ms": current.progress_ms,
            "track": track_item(provider, track),
        },
    )


def _clock(milliseconds: int) -> str:
    minutes, seconds = divmod(milliseconds // _MS_PER_SECOND, _SECONDS_PER_MINUTE)
    return f"{minutes}:{seconds:02d}"
