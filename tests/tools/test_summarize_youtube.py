from collections.abc import Callable, Sequence

import pytest

from jarvis.agent.registry import ToolRegistry
from jarvis.config import Settings
from jarvis.integrations.services import IntegrationServices
from jarvis.integrations.youtube import (
    FetchedCaptions,
    TranscriptSegment,
    TranscriptUnavailableError,
)
from jarvis.tools.summarize_youtube import SummarizeYoutubeTool, fit_transcript
from tests.tools.integration_helpers import integration_context

VIDEO_ID = "dQw4w9WgXcQ"


class FakeFetcher:
    """Canned captions, or a raised error; records every request."""

    def __init__(self, captions: FetchedCaptions | None = None, error: Exception | None = None):
        self.captions = captions or FetchedCaptions("en", ())
        self.error = error
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def fetch(self, video_id: str, languages: Sequence[str]) -> FetchedCaptions:
        self.calls.append((video_id, tuple(languages)))
        if self.error is not None:
            raise self.error
        return self.captions


def captions(count: int, words_each: int = 10) -> FetchedCaptions:
    segments = tuple(
        TranscriptSegment(" ".join([f"w{index}"] * words_each), start=index * 6.0, duration=6.0)
        for index in range(count)
    )
    return FetchedCaptions("en", segments)


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


def test_is_discovered_and_read_only() -> None:
    tool = ToolRegistry.discover().get("summarize_youtube")

    assert isinstance(tool, SummarizeYoutubeTool)
    assert not tool.requires_confirmation


async def test_short_transcript_is_returned_whole(settings: Settings) -> None:
    fetcher = FakeFetcher(captions(20))
    context = integration_context(settings, IntegrationServices(transcript_fetcher=fetcher))

    result = await SummarizeYoutubeTool().call(
        {"video": f"https://youtu.be/{VIDEO_ID}", "languages": ["en", "de"]}, context
    )

    assert result.ok
    assert result.spoken == "I've got the transcript of that video, about 2 minutes long."
    assert result.data is not None
    assert result.data["video_id"] == VIDEO_ID
    assert result.data["duration_seconds"] == 120
    assert result.data["sampled"] is False
    assert fetcher.calls == [(VIDEO_ID, ("en", "de"))]


async def test_long_transcript_is_sampled_across_the_video(settings: Settings) -> None:
    fetcher = FakeFetcher(captions(600))  # an hour, ~16k characters
    context = integration_context(settings, IntegrationServices(transcript_fetcher=fetcher))

    result = await SummarizeYoutubeTool().call({"video": VIDEO_ID, "max_chars": 2000}, context)

    assert result.data is not None
    transcript = result.data["transcript"]
    assert isinstance(transcript, str)
    assert len(transcript) <= 2000
    assert result.data["sampled"] is True
    assert transcript.startswith("w0 ")
    assert any(f"w{index} " in transcript for index in range(525, 600))  # the last eighth
    assert "Sampled in 8 excerpts" in result.detail


def test_fit_transcript_leaves_short_text_alone() -> None:
    assert fit_transcript("short text", 100) == ("short text", False)


@pytest.mark.parametrize(
    ("reason", "spoken"),
    [
        ("disabled", "Captions are turned off for that video, so I can't summarise it."),
        ("no_language_match", "That video has no captions in a language I asked for."),
        ("video_unavailable", "That video isn't available."),
        ("retrieval_failed", "I couldn't get captions for that video."),
    ],
)
async def test_unavailable_transcripts_are_explained(
    settings: Settings, reason: str, spoken: str
) -> None:
    fetcher = FakeFetcher(error=TranscriptUnavailableError(VIDEO_ID, reason))
    context = integration_context(settings, IntegrationServices(transcript_fetcher=fetcher))

    result = await SummarizeYoutubeTool().call({"video": VIDEO_ID}, context)

    assert not result.ok
    assert result.spoken == spoken
    assert reason in result.detail


async def test_not_a_youtube_link(settings: Settings) -> None:
    fetcher = FakeFetcher()
    context = integration_context(settings, IntegrationServices(transcript_fetcher=fetcher))

    result = await SummarizeYoutubeTool().call({"video": "https://vimeo.com/123456789"}, context)

    assert not result.ok
    assert result.spoken == "That doesn't look like a YouTube video."
    assert fetcher.calls == []


async def test_language_codes_are_validated(settings: Settings) -> None:
    context = integration_context(settings, IntegrationServices(transcript_fetcher=FakeFetcher()))

    result = await SummarizeYoutubeTool().call(
        {"video": VIDEO_ID, "languages": ["english!"]}, context
    )

    assert not result.ok
    assert "languages.0" in result.detail
