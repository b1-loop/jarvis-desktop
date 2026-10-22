"""Tool: a YouTube video's transcript, fitted to a budget for the LLM to summarise.

"Summarise that video" without watching it: the captions YouTube already
has are fetched (``fetch_transcript``, ADR-0012) and handed to the model.
A long talk's transcript can exceed any sensible budget, so instead of
keeping only the beginning, the tool samples evenly spaced excerpts across
the whole video; a summary of the first ten minutes of an hour-long talk
would be confidently wrong.
"""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import Field

from jarvis.agent.tool import Tool, ToolCategory, ToolContext, ToolInput, ToolResult
from jarvis.integrations.youtube import (
    InvalidVideoError,
    Transcript,
    TranscriptUnavailableError,
    fetch_transcript,
)
from jarvis.tools._integration_support import Service, not_configured, transcript_fetcher, trim

MIN_CHARS: Final = 1000
MAX_CHARS: Final = 30_000
SAMPLE_PARTS: Final = 8
"""Excerpts taken across an over-long transcript, so every part of the video is represented."""
_SECONDS_PER_MINUTE: Final = 60
_UNAVAILABLE_SPOKEN: Final = {
    "disabled": "Captions are turned off for that video, so I can't summarise it.",
    "no_language_match": "That video has no captions in a language I asked for.",
    "video_unavailable": "That video isn't available.",
}

LanguageCode = Annotated[str, Field(pattern=r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?$")]


class SummarizeYoutubeInput(ToolInput):
    """Arguments for :class:`SummarizeYoutubeTool`."""

    video: str = Field(
        min_length=11,
        max_length=300,
        description="YouTube URL (watch, youtu.be, shorts, embed, live) or the 11-character id.",
    )
    max_chars: int = Field(
        default=12_000,
        ge=MIN_CHARS,
        le=MAX_CHARS,
        description="Transcript budget in characters; long videos are sampled evenly to fit.",
    )
    languages: list[LanguageCode] = Field(
        default_factory=lambda: ["en"],
        min_length=1,
        max_length=5,
        description="Preferred caption languages, best first, e.g. ['en', 'de'].",
    )


class SummarizeYoutubeTool(Tool[SummarizeYoutubeInput]):
    """Fetches a video's captions for an LLM-written summary."""

    name = "summarize_youtube"
    description = (
        "Get the transcript of a YouTube video so you can summarise it or answer questions "
        "about it. Use when the user shares or names a YouTube link and asks what it says. "
        "Returns the caption text (sampled across the whole video if long); you write the "
        "summary."
    )
    Input = SummarizeYoutubeInput
    category = ToolCategory.WEB

    async def run(self, params: SummarizeYoutubeInput, context: ToolContext) -> ToolResult:
        """Fetch captions, fit them to the budget and explain failures in words."""
        fetcher = transcript_fetcher(context)
        if fetcher is None:
            return not_configured(Service.YOUTUBE)
        try:
            transcript = await fetch_transcript(fetcher, params.video, params.languages)
        except InvalidVideoError as error:
            return ToolResult.failure("That doesn't look like a YouTube video.", detail=str(error))
        except TranscriptUnavailableError as error:
            spoken = _UNAVAILABLE_SPOKEN.get(
                error.reason, "I couldn't get captions for that video."
            )
            return ToolResult.failure(spoken, detail=str(error))
        return _result(transcript, params.max_chars)


def fit_transcript(text: str, budget: int, parts: int = SAMPLE_PARTS) -> tuple[str, bool]:
    """``text`` unchanged if it fits, else ``parts`` evenly spaced excerpts within ``budget``.

    Returns:
        The fitted text and whether anything was left out.
    """
    if len(text) <= budget:
        return text, False
    share = budget // parts - 1  # room for the joining space
    stride = len(text) / parts
    excerpts = [trim(text[round(index * stride) :], share) for index in range(parts)]
    return " ".join(excerpts), True


def _result(transcript: Transcript, budget: int) -> ToolResult:
    text, sampled = fit_transcript(transcript.text, budget)
    last = transcript.segments[-1] if transcript.segments else None
    seconds = round(last.start + last.duration) if last else 0
    minutes = max(1, round(seconds / _SECONDS_PER_MINUTE))
    note = f" Sampled in {SAMPLE_PARTS} excerpts across the video." if sampled else ""
    return ToolResult.success(
        f"I've got the transcript of that video, about {minutes} minutes long.",
        detail=(
            f"Transcript of YouTube video {transcript.video_id} ({transcript.language}, "
            f"{seconds} s).{note} Summarise it or answer from it as the user asked.\n\n{text}"
        ),
        data={
            "video_id": transcript.video_id,
            "language": transcript.language,
            "duration_seconds": seconds,
            "sampled": sampled,
            "transcript": text,
        },
    )


# Design notes:
# - Even sampling over head-truncation: excerpts lose some continuity, but
#   the model sees the introduction, the middle and the conclusion, which is
#   what a summary needs. Each excerpt is cut on a word boundary and marked
#   with an ellipsis so the gaps are visible to the model.
# - Failure reasons from the transcript layer map to spoken sentences here,
#   where the wording belongs, instead of parsing exception messages.
