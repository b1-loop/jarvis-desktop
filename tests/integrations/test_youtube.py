from collections.abc import Sequence
from typing import ClassVar

import pytest
import youtube_transcript_api

from jarvis.integrations.youtube import (
    FetchedCaptions,
    InvalidVideoError,
    Transcript,
    TranscriptSegment,
    TranscriptUnavailableError,
    YouTubeTranscriptApiFetcher,
    fetch_transcript,
    parse_video_id,
)

VIDEO_ID = "dQw4w9WgXcQ"


class FakeFetcher:
    """Records calls and returns canned captions, or raises what it is told to."""

    def __init__(self, captions: FetchedCaptions | None = None, error: Exception | None = None):
        self.captions = captions or FetchedCaptions("en", ())
        self.error = error
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def fetch(self, video_id: str, languages: Sequence[str]) -> FetchedCaptions:
        self.calls.append((video_id, tuple(languages)))
        if self.error is not None:
            raise self.error
        return self.captions


@pytest.mark.parametrize(
    "url",
    [
        VIDEO_ID,
        f"  {VIDEO_ID}  ",
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://www.youtube.com/watch?list=PL123&v={VIDEO_ID}&t=42s",
        f"http://youtube.com/watch?v={VIDEO_ID}",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",
        f"https://music.youtube.com/watch?v={VIDEO_ID}",
        f"youtube.com/watch?v={VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}?t=10",
        f"youtu.be/{VIDEO_ID}",
        f"https://www.youtube.com/embed/{VIDEO_ID}",
        f"https://www.youtube.com/embed/{VIDEO_ID}?autoplay=1",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}",
        f"https://www.youtube.com/v/{VIDEO_ID}",
        f"https://www.youtube-nocookie.com/embed/{VIDEO_ID}",
    ],
)
def test_parse_video_id_handles_every_common_form(url: str) -> None:
    assert parse_video_id(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        "https://vimeo.com/123456",
        "https://www.youtube.com/",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/watch?v=tooshort",
        "https://www.youtube.com/channel/UCabc",
        "https://youtu.be/",
        "https://evil.test/watch?v=" + VIDEO_ID,
        "https://notyoutube.com/watch?v=" + VIDEO_ID,
    ],
)
def test_parse_video_id_rejects_everything_else(url: str) -> None:
    with pytest.raises(InvalidVideoError, match="not a YouTube video"):
        parse_video_id(url)


async def test_fetch_transcript_normalises_segments_and_joins_text() -> None:
    fetcher = FakeFetcher(
        FetchedCaptions(
            "en-GB",
            (
                TranscriptSegment("hello\nworld", 0, 1.5),
                TranscriptSegment("  second   cue ", 1.5, 2),
                TranscriptSegment("", 3.5, 1),
            ),
        )
    )

    transcript = await fetch_transcript(fetcher, f"https://youtu.be/{VIDEO_ID}", ("de", "en"))

    assert fetcher.calls == [(VIDEO_ID, ("de", "en"))]
    assert transcript == Transcript(
        video_id=VIDEO_ID,
        language="en-GB",
        segments=(
            TranscriptSegment("hello world", 0.0, 1.5),
            TranscriptSegment("second cue", 1.5, 2.0),
            TranscriptSegment("", 3.5, 1.0),
        ),
        text="hello world second cue",
    )


async def test_fetch_transcript_defaults_to_english() -> None:
    fetcher = FakeFetcher()

    await fetch_transcript(fetcher, VIDEO_ID)

    assert fetcher.calls == [(VIDEO_ID, ("en",))]


async def test_fetch_transcript_rejects_bad_input_before_fetching() -> None:
    fetcher = FakeFetcher()

    with pytest.raises(InvalidVideoError):
        await fetch_transcript(fetcher, "https://vimeo.com/1")
    assert fetcher.calls == []


async def test_fetch_transcript_propagates_unavailable_errors() -> None:
    error = TranscriptUnavailableError(VIDEO_ID, "disabled")
    fetcher = FakeFetcher(error=error)

    with pytest.raises(TranscriptUnavailableError, match="disabled") as info:
        await fetch_transcript(fetcher, VIDEO_ID)

    assert info.value.video_id == VIDEO_ID
    assert info.value.reason == "disabled"


def test_unavailable_error_message_includes_detail_when_given() -> None:
    assert str(TranscriptUnavailableError("abc", "retrieval_failed", "IpBlocked")).endswith(
        "(retrieval_failed): IpBlocked"
    )
    assert str(TranscriptUnavailableError("abc", "disabled")).endswith("(disabled)")


# --- Real fetcher, with the library's client replaced ------------------------


class _Snippet:
    def __init__(self, text: str, start: float, duration: float) -> None:
        self.text, self.start, self.duration = text, start, duration


class _Fetched:
    language_code = "en"
    snippets: ClassVar[list[_Snippet]] = [_Snippet("one", 0.0, 1.0), _Snippet("two", 1.0, 1.0)]


def _patch_api(
    monkeypatch: pytest.MonkeyPatch, *, result: object = None, error: Exception | None = None
) -> None:
    class FakeApi:
        def fetch(self, video_id: str, languages: list[str]) -> object:
            assert video_id == VIDEO_ID
            assert languages == ["en"]
            if error is not None:
                raise error
            return result

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", FakeApi)


def test_real_fetcher_maps_library_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_api(monkeypatch, result=_Fetched())

    captions = YouTubeTranscriptApiFetcher().fetch(VIDEO_ID, ["en"])

    assert captions.language == "en"
    assert captions.segments == (
        TranscriptSegment("one", 0.0, 1.0),
        TranscriptSegment("two", 1.0, 1.0),
    )


@pytest.mark.parametrize(
    ("error_name", "reason"),
    [
        ("TranscriptsDisabled", "disabled"),
        ("NoTranscriptFound", "no_language_match"),
        ("VideoUnavailable", "video_unavailable"),
        ("IpBlocked", "retrieval_failed"),
    ],
)
def test_real_fetcher_maps_library_errors(
    monkeypatch: pytest.MonkeyPatch, error_name: str, reason: str
) -> None:
    error_type = getattr(youtube_transcript_api, error_name)
    error = error_type.__new__(error_type)
    Exception.__init__(error, "library message")
    _patch_api(monkeypatch, error=error)

    with pytest.raises(TranscriptUnavailableError) as info:
        YouTubeTranscriptApiFetcher().fetch(VIDEO_ID, ["en"])

    assert info.value.reason == reason
    assert info.value.__cause__ is error
    if reason == "retrieval_failed":
        assert error_name in str(info.value)
