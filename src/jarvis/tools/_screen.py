"""Shared plumbing for the screen-understanding tools.

``describe_screen``, ``explain_screen_error`` and ``ask_about_screen``
differ only in the prompt they build; asking the vision service, turning
its errors into clear spoken results and telling the user when the
screenshot went to a cloud provider happens here, once. The leading
underscore keeps the registry from treating this module as a plugin.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import Field, JsonValue

from jarvis.agent.tool import ToolContext, ToolInput, ToolResult
from jarvis.vision.errors import VisionError, VisionUnavailableError
from jarvis.vision.screen import ScreenAnswer, ScreenTarget

MAX_SPOKEN_CHARS: Final = 240
"""Longest piece of the answer read aloud; the agent gets the full text."""

_SENTENCE_END: Final = re.compile(r"(?<=[.!?])\s")

TARGET_FIELD_DESCRIPTION: Final = (
    "What to capture: 'primary' (the main monitor, default), 'all_monitors' "
    "(every monitor as one image) or 'active_window' (only the focused window)."
)


class ScreenInput(ToolInput):
    """The capture target, common to every screen tool."""

    target: ScreenTarget = Field(default=ScreenTarget.PRIMARY, description=TARGET_FIELD_DESCRIPTION)


async def answer_from_screen(context: ToolContext, prompt: str, target: ScreenTarget) -> ToolResult:
    """Ask the vision service ``prompt`` about ``target`` and phrase the result.

    Unavailability and failures come back as failed results with a reason
    the user can act on, rather than the generic "couldn't complete" line.
    """
    if context.vision is None:
        return ToolResult.failure(
            "I can't look at your screen in this setup.",
            detail="no vision service is configured in ToolContext.vision",
        )
    try:
        answer = await context.vision.ask(prompt, target)
    except VisionUnavailableError as error:
        return ToolResult.failure(
            f"I can't look at your screen: {error.reason}.", detail=str(error)
        )
    except VisionError as error:
        return ToolResult.failure(
            "I couldn't make sense of your screen just now.", detail=str(error)
        )
    return ToolResult.success(spoken_summary(answer), detail=answer.text, data=answer_data(answer))


def spoken_summary(answer: ScreenAnswer) -> str:
    """The first sentence of the answer, plus a cloud-upload notice when it applies."""
    spoken = first_sentence(answer.text)
    if answer.sent_to_cloud:
        spoken += f" I sent a screenshot to {answer.provider} to answer that."
    return spoken


def first_sentence(text: str, limit: int = MAX_SPOKEN_CHARS) -> str:
    """Up to the first sentence break, cut on a word boundary at ``limit``."""
    sentence = _SENTENCE_END.split(text.strip(), maxsplit=1)[0]
    if len(sentence) <= limit:
        return sentence
    return sentence[:limit].rsplit(" ", 1)[0] + "…"


def answer_data(answer: ScreenAnswer) -> dict[str, JsonValue]:
    """JSON payload for chaining and the dashboard; never the image itself."""
    return {
        "answer": answer.text,
        "target": answer.target.value,
        "width": answer.width,
        "height": answer.height,
        "image_bytes": answer.image_bytes,
        "media_type": answer.media_type,
        "latency_ms": round(answer.latency_ms, 1),
        "model": answer.model,
        "provider": answer.provider,
        "sent_to_cloud": answer.sent_to_cloud,
        "screenshot_path": str(answer.original_path) if answer.original_path else None,
    }


# Design notes:
# - The cloud notice is part of the spoken text, not only the data, because
#   the user should hear that their screen left the computer every time it
#   does; for a local Ollama model there is nothing to disclose.
# - None of these tools needs confirmation: they only read the screen, and
#   the user's question is the request to look. The notice covers the
#   transparency that a confirmation would otherwise provide.
