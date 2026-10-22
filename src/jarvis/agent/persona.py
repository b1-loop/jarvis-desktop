"""Who Jarvis is: name, tone, speaking style and language, and the system prompt.

The persona is configuration, not code: name and tone come from
``Settings`` so a user can rename the assistant or change its manner in
``.env`` without touching the prompt. The prompt itself is one short,
explicit template in this module because every rule in it changes how the
model behaves on every turn, and a reviewer should be able to read all of
them in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jarvis.config import Settings

DEFAULT_VOICE_STYLE: Final = "a measured pace with short, clear sentences"
"""How the answers should sound when spoken; a hint for wording, not for TTS."""

DEFAULT_LANGUAGE: Final = "English"
"""Used when the configured voice id does not reveal a language."""

_LANGUAGE_NAMES: Final[dict[str, str]] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "bg": "Bulgarian",
    "sv": "Swedish",
    "nl": "Dutch",
    "pt": "Portuguese",
}
"""Languages edge-tts voices are commonly configured in, by ISO 639-1 prefix."""

SYSTEM_PROMPT_TEMPLATE: Final = """\
You are {name}, a voice assistant running on the user's own computer.
Your tone is {tone}. Answer in {language}, with {voice_style}.

Everything you write is read aloud by text-to-speech:
- Use short, plain sentences. No markdown, lists, headings, code blocks, links or emoji.
- Be brief. Give detail only when the user asks for it.

Tools:
- Use the tools whenever they can answer the question or do the task.
  Call several in one reply when they do not depend on each other.
- The system itself asks the user before any destructive, sending or spending action,
  so just call the tool; do not ask for permission yourself.
- Never invent tool results. If a tool fails or the user declines, say so plainly.
- After using tools, always finish with one or two sentences that summarise what you did.
"""

MEMORY_SECTION_TEMPLATE: Final = """
What you remember about the user and earlier conversations:
{memory}
"""


@dataclass(frozen=True, slots=True)
class Persona:
    """The assistant's identity as the model is told about it.

    Attributes:
        name: What the assistant calls itself (``Settings.assistant_name``).
        tone: Manner of speaking (``Settings.persona_tone``).
        voice_style: How the wording should sound when spoken aloud.
        language: Natural language to answer in.
    """

    name: str
    tone: str
    voice_style: str = DEFAULT_VOICE_STYLE
    language: str = DEFAULT_LANGUAGE

    @classmethod
    def from_settings(cls, settings: Settings) -> Persona:
        """Build the persona from settings; the language follows the TTS voice."""
        return cls(
            name=settings.assistant_name,
            tone=settings.persona_tone,
            language=language_of_voice(settings.tts_voice),
        )

    def system_prompt(self, memory: str = "") -> str:
        """The system message for one turn, with recalled memory appended if any."""
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            name=self.name,
            tone=self.tone,
            language=self.language,
            voice_style=self.voice_style,
        )
        if memory.strip():
            prompt += MEMORY_SECTION_TEMPLATE.format(memory=memory.strip())
        return prompt


def language_of_voice(voice_id: str) -> str:
    """``en-GB-RyanNeural`` -> ``English``; unknown prefixes fall back to English."""
    prefix = voice_id.split("-", 1)[0].lower()
    return _LANGUAGE_NAMES.get(prefix, DEFAULT_LANGUAGE)


# Design notes:
# - The language is derived from the TTS voice rather than a separate
#   setting: answering in German with an English voice sounds broken, so
#   the two must agree, and one knob cannot disagree with itself.
# - "Just call the tool" is deliberate. Confirmation is enforced centrally
#   (ADR-0006); a model that also asks "shall I?" makes the user confirm
#   twice, once to the model and once to the policy.
# - Memory goes into the system prompt, not into fake user/assistant
#   messages, so the model can tell remembered context from what the user
#   just said, and so history never violates a provider's role alternation.
