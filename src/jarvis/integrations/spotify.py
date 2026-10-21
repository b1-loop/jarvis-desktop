"""Spotify provider: spotipy for OAuth, httpx for the Web API.

spotipy's ``SpotifyOAuth`` is kept because it implements the authorization-
code flow end to end (browser, local redirect listener, refresh, cache
file) and that is the part nobody should rewrite. Every API call goes
through httpx instead of ``spotipy.Spotify`` so requests are typed, async
and testable with ``httpx.MockTransport``; the Web API is small enough that
the client adds little. Scopes are the two needed to read and control
playback, nothing about the library or the user's profile.
"""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import httpx
from spotipy.cache_handler import CacheFileHandler
from spotipy.oauth2 import SpotifyOAuth, SpotifyOauthError

from jarvis.integrations.base import (
    AuthError,
    MusicProvider,
    NotConfiguredError,
    NowPlaying,
    ProviderError,
    ProviderStatus,
    Track,
)
from jarvis.integrations.rest import RestSession
from jarvis.integrations.tokens import ThreadedTokenSource, TokenSource

if TYPE_CHECKING:
    from jarvis.config import Settings

SPOTIFY_SCOPES = ("user-read-playback-state", "user-modify-playback-state")
BASE_URL = "https://api.spotify.com/v1"
PROVIDER = "spotify"

_TRACK_PREFIX = "spotify:track:"
_NO_DEVICE_HINT = "no active device; open Spotify on a device and try again"


class SpotifyOAuthFlow(Protocol):
    """The slice of ``spotipy.SpotifyOAuth`` Jarvis relies on."""

    def get_access_token(self, as_dict: bool = ...) -> Any:
        """Cached, refreshed or newly authorised token (string when ``as_dict`` is False)."""


class SpotifyAuth(ThreadedTokenSource):
    """Token source over spotipy's authorization-code flow."""

    def __init__(self, flow: SpotifyOAuthFlow) -> None:
        super().__init__()
        self._flow = flow

    @classmethod
    def from_settings(cls, settings: Settings, cache_path: Path) -> SpotifyAuth:
        """Build the real flow; the token cache file lives at ``cache_path``.

        Raises:
            NotConfiguredError: If the client id or secret is unset.
        """
        missing = [
            name
            for name, value in (
                ("SPOTIFY_CLIENT_ID", settings.spotify_client_id),
                ("SPOTIFY_CLIENT_SECRET", settings.spotify_client_secret),
            )
            if value is None
        ]
        if missing or settings.spotify_client_secret is None:
            raise NotConfiguredError(PROVIDER, missing)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        flow = SpotifyOAuth(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret.get_secret_value(),
            redirect_uri=settings.spotify_redirect_uri,
            scope=" ".join(SPOTIFY_SCOPES),
            cache_handler=CacheFileHandler(cache_path=str(cache_path)),
            open_browser=True,
        )
        return cls(cast(SpotifyOAuthFlow, flow))

    def _acquire_blocking(self) -> str:
        try:
            token = self._flow.get_access_token(as_dict=False)
        except SpotifyOauthError as error:
            raise AuthError(PROVIDER, f"authorisation failed: {error}") from error
        if not isinstance(token, str) or not token:
            raise AuthError(PROVIDER, "authorisation returned no access token")
        return token


class SpotifyProvider(MusicProvider):
    """Playback control for the user's active Spotify device."""

    provider_name = PROVIDER
    scopes = SPOTIFY_SCOPES

    def __init__(self, http: httpx.AsyncClient, token_source: TokenSource) -> None:
        self._rest = RestSession(
            http, provider=self.provider_name, base_url=BASE_URL, token_source=token_source
        )

    async def status(self) -> ProviderStatus:
        """Session-level status; see :meth:`RestSession.status`."""
        return self._rest.status()

    async def now_playing(self) -> NowPlaying | None:
        """Current track, or ``None`` when idle or playing a non-track item."""
        data = await self._rest.request("GET", "/me/player/currently-playing")
        if not data or not data.get("item"):
            return None
        return NowPlaying(
            track=parse_track(data["item"]),
            is_playing=bool(data.get("is_playing")),
            progress_ms=int(data.get("progress_ms") or 0),
        )

    async def play(self, query_or_uri: str) -> Track | None:
        """Play a track/context URI or the best search hit for free text."""
        if not query_or_uri.startswith("spotify:"):
            track = await self._best_match(query_or_uri)
            await self._player("PUT", "/me/player/play", json={"uris": [track.uri]})
            return track
        if query_or_uri.startswith(_TRACK_PREFIX):
            await self._player("PUT", "/me/player/play", json={"uris": [query_or_uri]})
            track_id = query_or_uri.removeprefix(_TRACK_PREFIX)
            return parse_track(await self._rest.request("GET", f"/tracks/{track_id}"))
        await self._player("PUT", "/me/player/play", json={"context_uri": query_or_uri})
        return None

    async def pause(self) -> None:
        """Pause the active device."""
        await self._player("PUT", "/me/player/pause")

    async def resume(self) -> None:
        """Resume the active device."""
        await self._player("PUT", "/me/player/play")

    async def next_track(self) -> None:
        """Skip forward."""
        await self._player("POST", "/me/player/next")

    async def queue(self, query: str) -> Track:
        """Queue the best hit for ``query`` and return it."""
        track = await self._best_match(query)
        await self._player("POST", "/me/player/queue", params={"uri": track.uri})
        return track

    async def search(self, query: str, limit: int = 5) -> list[Track]:
        """Track search, Spotify's relevance order."""
        data = await self._rest.request(
            "GET", "/search", params={"q": query, "type": "track", "limit": limit}
        )
        return [parse_track(item) for item in data.get("tracks", {}).get("items", [])]

    async def _best_match(self, query: str) -> Track:
        hits = await self.search(query, limit=1)
        if not hits:
            raise ProviderError(self.provider_name, f"nothing found for {query!r}")
        return hits[0]

    async def _player(
        self, method: str, path: str, *, json: Any = None, params: dict[str, Any] | None = None
    ) -> None:
        """Player endpoints answer 404 when no device is active; say so plainly."""
        try:
            await self._rest.request(method, path, json=json, params=params)
        except ProviderError as error:
            if error.status_code == HTTPStatus.NOT_FOUND:
                raise ProviderError(
                    self.provider_name, _NO_DEVICE_HINT, status_code=error.status_code
                ) from error
            raise


def parse_track(item: dict[str, Any]) -> Track:
    """Map a Spotify ``TrackObject`` to the neutral model."""
    return Track(
        id=item["id"],
        title=item.get("name", ""),
        artist=", ".join(a.get("name", "") for a in item.get("artists", [])),
        album=item.get("album", {}).get("name", ""),
        uri=item["uri"],
        duration_ms=int(item.get("duration_ms") or 0),
    )


# Design notes:
# - spotipy owns its token cache file (data_dir/tokens/spotify.json) through
#   its own CacheFileHandler instead of going through TokenStore: SpotifyOAuth
#   insists on a CacheHandler subclass, and spotipy is untyped, so adapting
#   TokenStore would need a mypy ignore for a few lines of glue. A file next to
#   the other tokens is the honest, simpler option.
# - play() returns None for context URIs (album, playlist): the Web API does
#   not say which track started, and polling currently-playing immediately
#   afterwards races the device.
