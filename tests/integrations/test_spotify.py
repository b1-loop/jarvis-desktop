from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from spotipy.oauth2 import SpotifyOauthError

from jarvis.config import Settings
from jarvis.integrations.base import AuthError, NotConfiguredError, ProviderError
from jarvis.integrations.spotify import SPOTIFY_SCOPES, SpotifyAuth, SpotifyProvider
from jarvis.integrations.tokens import StaticTokenSource
from tests.integrations.conftest import Router

TRACK = {
    "id": "t1",
    "name": "Blue in Green",
    "uri": "spotify:track:t1",
    "duration_ms": 337000,
    "artists": [{"name": "Miles Davis"}, {"name": "Bill Evans"}],
    "album": {"name": "Kind of Blue"},
}


class FakeFlow:
    def __init__(self, result: Any = "fake-spotify-token", error: Exception | None = None) -> None:
        self.result, self.error, self.calls = result, error, 0

    def get_access_token(self, as_dict: bool = True) -> Any:
        self.calls += 1
        assert as_dict is False
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def provider(router: Router, token_source: StaticTokenSource) -> SpotifyProvider:
    return SpotifyProvider(router.client(), token_source)


def test_scopes_cover_playback_only() -> None:
    assert SpotifyProvider.scopes == SPOTIFY_SCOPES
    assert set(SPOTIFY_SCOPES) == {"user-read-playback-state", "user-modify-playback-state"}


async def test_auth_returns_the_flow_token_and_maps_failures() -> None:
    assert await SpotifyAuth(FakeFlow()).access_token() == "fake-spotify-token"

    with pytest.raises(AuthError, match="authorisation failed"):
        await SpotifyAuth(FakeFlow(error=SpotifyOauthError("denied"))).access_token()
    with pytest.raises(AuthError, match="no access token"):
        await SpotifyAuth(FakeFlow(result=None)).access_token()


def test_from_settings_names_missing_variables(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    with pytest.raises(NotConfiguredError) as raised:
        SpotifyAuth.from_settings(make_settings(), tmp_path / "spotify.json")
    assert raised.value.missing == ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET")

    with pytest.raises(NotConfiguredError, match="SPOTIFY_CLIENT_SECRET"):
        SpotifyAuth.from_settings(make_settings(spotify_client_id="id"), tmp_path / "s.json")

    auth = SpotifyAuth.from_settings(
        make_settings(spotify_client_id="id", spotify_client_secret="fake"),
        tmp_path / "tokens" / "spotify.json",
    )
    assert isinstance(auth, SpotifyAuth)
    assert (tmp_path / "tokens").is_dir()


async def test_now_playing_handles_idle_and_active_players(
    router: Router, provider: SpotifyProvider
) -> None:
    router.add("GET", "/v1/me/player/currently-playing", status=204)
    assert await provider.now_playing() is None

    router.add(
        "GET",
        "/v1/me/player/currently-playing",
        {"item": TRACK, "is_playing": True, "progress_ms": 1200},
    )
    playing = await provider.now_playing()

    assert playing is not None
    assert playing.track.title == "Blue in Green"
    assert playing.track.artist == "Miles Davis, Bill Evans"
    assert playing.track.album == "Kind of Blue"
    assert playing.track.duration_ms == 337000
    assert playing.is_playing and playing.progress_ms == 1200

    router.add("GET", "/v1/me/player/currently-playing", {"item": None, "is_playing": True})
    assert await provider.now_playing() is None


async def test_play_free_text_searches_then_starts_the_hit(
    router: Router, provider: SpotifyProvider
) -> None:
    router.add("GET", "/v1/search", {"tracks": {"items": [TRACK]}})
    router.add("PUT", "/v1/me/player/play", status=204)

    track = await provider.play("blue in green")

    search = router.requests[0]
    assert search.url.params["q"] == "blue in green"
    assert search.url.params["type"] == "track"
    assert search.url.params["limit"] == "1"
    assert router.sent_json() == {"uris": ["spotify:track:t1"]}
    assert track is not None and track.id == "t1"


async def test_play_by_uri_distinguishes_tracks_from_contexts(
    router: Router, provider: SpotifyProvider
) -> None:
    router.add("PUT", "/v1/me/player/play", status=204)
    router.add("GET", "/v1/tracks/t1", TRACK)

    track = await provider.play("spotify:track:t1")
    assert track is not None and track.uri == "spotify:track:t1"
    assert router.sent_json(0) == {"uris": ["spotify:track:t1"]}

    assert await provider.play("spotify:playlist:p9") is None
    assert router.sent_json() == {"context_uri": "spotify:playlist:p9"}


async def test_pause_resume_next_and_queue(router: Router, provider: SpotifyProvider) -> None:
    router.add("PUT", "/v1/me/player/pause", status=204)
    router.add("PUT", "/v1/me/player/play", status=204)
    router.add("POST", "/v1/me/player/next", status=204)
    router.add("POST", "/v1/me/player/queue", status=204)
    router.add("GET", "/v1/search", {"tracks": {"items": [TRACK]}})

    await provider.pause()
    await provider.resume()
    await provider.next_track()
    queued = await provider.queue("blue in green")

    assert [(r.method, r.url.path) for r in router.requests] == [
        ("PUT", "/v1/me/player/pause"),
        ("PUT", "/v1/me/player/play"),
        ("POST", "/v1/me/player/next"),
        ("GET", "/v1/search"),
        ("POST", "/v1/me/player/queue"),
    ]
    assert router.last().url.params["uri"] == "spotify:track:t1"
    assert queued.id == "t1"


async def test_search_limit_and_empty_results(router: Router, provider: SpotifyProvider) -> None:
    router.add("GET", "/v1/search", {"tracks": {"items": []}})

    assert await provider.search("nothing", limit=3) == []
    assert router.last().url.params["limit"] == "3"
    with pytest.raises(ProviderError, match="nothing found"):
        await provider.play("nothing")


async def test_no_active_device_gets_a_helpful_message(
    router: Router, provider: SpotifyProvider
) -> None:
    router.add(
        "PUT", "/v1/me/player/pause", status=404, json={"error": {"reason": "NO_ACTIVE_DEVICE"}}
    )
    router.add("POST", "/v1/me/player/next", status=500)

    with pytest.raises(ProviderError, match="no active device"):
        await provider.pause()
    with pytest.raises(ProviderError, match="HTTP 500"):
        await provider.next_track()
