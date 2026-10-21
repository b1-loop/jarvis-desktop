"""YouTube transcripts, for "summarise that video".

YouTube serves the captions a video already has (uploaded or auto-generated)
through an undocumented endpoint that ``youtube-transcript-api`` wraps. That
library is synchronous and talks to the network, so it sits behind the
:class:`TranscriptFetcher` Protocol: production injects
:class:`YouTubeTranscriptApiFetcher`, tests inject a fake, and
:func:`fetch_transcript` runs whichever it gets in a worker thread.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol
from urllib.parse import parse_qs, urlsplit

from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_LANGUAGES: Final[tuple[str, ...]] = ("en",)

_VIDEO_ID: Final = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS: Final = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
    }
)
_SHORT_HOSTS: Final = frozenset({"youtu.be", "www.youtu.be"})
_PATH_PREFIXES: Final = ("/embed/", "/shorts/", "/live/", "/v/")


class TranscriptError(JarvisError):
    """Base class for transcript failures."""


class InvalidVideoError(TranscriptError):
    """The input is neither a YouTube URL nor an 11-character video id."""


class TranscriptUnavailableError(TranscriptError):
    """The video exists but no usable transcript could be retrieved.

    Attributes:
        video_id: The video in question.
        reason: Short machine-friendly cause, e.g. ``"disabled"``.
    """

    def __init__(self, video_id: str, reason: str, detail: str = "") -> None:
        self.video_id = video_id
        self.reason = reason
        message = f"no transcript for video {video_id} ({reason})"
        super().__init__(f"{message}: {detail}" if detail else message)


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One caption cue.

    Attributes:
        text: Caption text with line breaks collapsed.
        start: Offset from the start of the video, in seconds.
        duration: How long the cue stays on screen, in seconds.
    """

    text: str
    start: float
    duration: float


@dataclass(frozen=True, slots=True)
class Transcript:
    """A video's captions in one language.

    Attributes:
        video_id: The 11-character YouTube id.
        language: BCP-47 code of the transcript that was found.
        segments: Timed cues in order.
        text: All cues joined with spaces, for summarising.
    """

    video_id: str
    language: str
    segments: tuple[TranscriptSegment, ...]
    text: str


@dataclass(frozen=True, slots=True)
class FetchedCaptions:
    """What a :class:`TranscriptFetcher` returns before it is normalised.

    Attributes:
        language: BCP-47 code of the captions.
        segments: Timed cues.
    """

    language: str
    segments: tuple[TranscriptSegment, ...]


class TranscriptFetcher(Protocol):
    """Synchronous transcript source; called from a worker thread."""

    def fetch(self, video_id: str, languages: Sequence[str]) -> FetchedCaptions:
        """Return captions in the first available of ``languages``.

        Raises:
            TranscriptUnavailableError: If the video has no usable captions.
        """


class YouTubeTranscriptApiFetcher:
    """Real fetcher over ``youtube-transcript-api`` (the ``documents`` extra).

    Imported lazily so the package stays importable, and the stub path
    usable, without the extra installed.
    """

    def fetch(self, video_id: str, languages: Sequence[str]) -> FetchedCaptions:
        """Fetch through the library and translate its exceptions."""
        # Lazy: the library is in the optional ``documents`` extra.
        from youtube_transcript_api import (
            CouldNotRetrieveTranscript,
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
            YouTubeTranscriptApi,
        )

        try:
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
        except TranscriptsDisabled as error:
            raise TranscriptUnavailableError(video_id, "disabled") from error
        except NoTranscriptFound as error:
            raise TranscriptUnavailableError(video_id, "no_language_match") from error
        except VideoUnavailable as error:
            raise TranscriptUnavailableError(video_id, "video_unavailable") from error
        except CouldNotRetrieveTranscript as error:
            raise TranscriptUnavailableError(
                video_id, "retrieval_failed", type(error).__name__
            ) from error
        segments = tuple(
            TranscriptSegment(text=item.text, start=item.start, duration=item.duration)
            for item in fetched.snippets
        )
        return FetchedCaptions(language=fetched.language_code, segments=segments)


def parse_video_id(url_or_id: str) -> str:
    """Extract the video id from any common YouTube URL form, or validate a bare id.

    Handles ``watch?v=``, ``youtu.be/``, ``/embed/``, ``/shorts/``, ``/live/``
    and ``/v/`` links, with or without a scheme, and ignores playlist and
    timestamp parameters.

    Raises:
        InvalidVideoError: If no 11-character id can be found.
    """
    candidate = url_or_id.strip()
    if _VIDEO_ID.match(candidate):
        return candidate
    parts = urlsplit(candidate if "://" in candidate else f"https://{candidate}")
    host = parts.hostname or ""
    found: str | None = None
    if host in _SHORT_HOSTS:
        found = parts.path.strip("/").split("/")[0]
    elif host in _YOUTUBE_HOSTS:
        found = _id_from_youtube_path(parts.path, parts.query)
    if found and _VIDEO_ID.match(found):
        return found
    raise InvalidVideoError(f"not a YouTube video URL or id: {url_or_id!r}")


def _id_from_youtube_path(path: str, query: str) -> str | None:
    if path == "/watch":
        return parse_qs(query).get("v", [None])[0]
    for prefix in _PATH_PREFIXES:
        if path.startswith(prefix):
            return path.removeprefix(prefix).split("/")[0]
    return None


async def fetch_transcript(
    fetcher: TranscriptFetcher,
    url_or_id: str,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
) -> Transcript:
    """Fetch and normalise the transcript of a video.

    Args:
        fetcher: Transcript source; the real one or a test fake.
        url_or_id: Any YouTube URL form or a bare video id.
        languages: Preferred language codes, best first.

    Raises:
        InvalidVideoError: If ``url_or_id`` is not recognisable.
        TranscriptUnavailableError: If the video has no usable captions.
    """
    video_id = parse_video_id(url_or_id)
    captions = await asyncio.to_thread(fetcher.fetch, video_id, tuple(languages))
    segments = tuple(_normalise(segment) for segment in captions.segments)
    text = " ".join(segment.text for segment in segments if segment.text)
    log.info(
        "youtube.transcript.fetched",
        video_id=video_id,
        language=captions.language,
        segments=len(segments),
        chars=len(text),
    )
    return Transcript(video_id=video_id, language=captions.language, segments=segments, text=text)


def _normalise(segment: TranscriptSegment) -> TranscriptSegment:
    """Collapse the line breaks YouTube inserts for on-screen layout."""
    return TranscriptSegment(
        text=" ".join(segment.text.split()),
        start=float(segment.start),
        duration=float(segment.duration),
    )


# Design notes:
# - The fetcher Protocol returns a tiny neutral dataclass instead of the
#   library's FetchedTranscript, so tests need no third-party objects and a
#   different transcript source (yt-dlp, Whisper on the audio) can be dropped
#   in without touching callers.
# - Library exceptions are mapped to one error with a ``reason`` code; the
#   agent turns the code into a spoken sentence ("captions are disabled on
#   that video") without parsing library messages.
# - No API key: the captions endpoint is the same one the YouTube player
#   uses. It is unofficial, so a breakage shows up as retrieval_failed with
#   the library's exception name attached for diagnosis.
