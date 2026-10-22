"""Tool: pause, resume or skip music on the user's active player.

The everyday "pause the music" / "next song". Playback control acts on the
user's own device, is instantly reversible and costs nothing, so it never
asks for confirmation (see the design notes).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.base import MusicProvider
from jarvis.tools._integration_support import Service, music_providers, not_configured


class MusicAction(StrEnum):
    """Playback commands; ``play`` without a query resumes."""

    PLAY = "play"
    PAUSE = "pause"
    RESUME = "resume"
    NEXT = "next"


class MusicControlInput(ToolInput):
    """Arguments for :class:`MusicControlTool`."""

    action: MusicAction = Field(
        description="'pause', 'resume' (or 'play') to continue, 'next' to skip the track."
    )


class MusicControlTool(Tool[MusicControlInput]):
    """Pauses, resumes or skips playback."""

    name = "music_control"
    description = (
        "Control music playback on the user's active Spotify device: pause, resume (or play "
        "to continue what was paused) or skip to the next track. To start something specific "
        "use play_music instead."
    )
    Input = MusicControlInput
    category = ToolCategory.MEDIA
    keywords = (
        "pause",
        "resume",
        "skip",
        "next",
        "song",
        "track",
        "pausa",
        "hoppa",
        "nästa",
        "fortsätt",
        "låten",
    )

    async def run(self, params: MusicControlInput, context: ToolContext) -> ToolResult:
        """Send the command to the first music provider."""
        providers = music_providers(context)
        if not providers:
            return not_configured(Service.MUSIC)
        provider = providers[0]
        if params.action is MusicAction.PAUSE:
            await provider.pause()
            return _done("Paused.", provider, params.action)
        if params.action is MusicAction.NEXT:
            return await _skip(provider)
        await provider.resume()
        return _done("Resuming the music.", provider, params.action)


async def _skip(provider: MusicProvider) -> ToolResult:
    await provider.next_track()
    current = await provider.now_playing()
    if current is None:
        return _done("Skipped; nothing else is lined up.", provider, MusicAction.NEXT)
    track = current.track
    return _done(f"Now playing {track.title} by {track.artist}.", provider, MusicAction.NEXT)


def _done(spoken: str, provider: MusicProvider, action: MusicAction) -> ToolResult:
    return ToolResult.success(
        spoken,
        detail=f"{action.value} sent to {provider.provider_name}. {spoken}",
        data={"action": action.value, "provider": provider.provider_name},
    )


# Design notes:
# - No confirmation for any action here. CLAUDE.md asks for confirmation on
#   anything that modifies the system, sends, spends or touches external
#   accounts; pausing a song touches the Spotify account in the most literal
#   sense, but it is the user's own player, costs nothing, is undone by the
#   opposite command and is exactly what was asked. A "are you sure you want
#   to pause?" would teach the user to say yes without listening, which
#   weakens the confirmations that matter. Anything that would change the
#   user's library or playlists would confirm; no tool does that (the
#   Spotify scopes do not even allow it, ADR-0009).
# - Only the first music provider is used: playback has one active device,
#   and merging "pause" across services has no meaning.
