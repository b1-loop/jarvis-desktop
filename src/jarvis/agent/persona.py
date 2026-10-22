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

_LANGUAGE_NAMES: Final[dict[str, tuple[str, str]]] = {
    "en": ("English", "English"),
    "de": ("German", "Deutsch"),
    "fr": ("French", "français"),
    "es": ("Spanish", "español"),
    "it": ("Italian", "italiano"),
    "bg": ("Bulgarian", "български"),
    "sv": ("Swedish", "svenska"),
    "nl": ("Dutch", "Nederlands"),
    "pt": ("Portuguese", "português"),
    "zh": ("Chinese", "中文"),
    "ja": ("Japanese", "日本語"),
    "ko": ("Korean", "한국어"),
}
"""Languages edge-tts voices are commonly configured in, by ISO 639-1 prefix:
the English name (for the prompt) and the name in the language itself."""

CJK_LANGUAGES: Final = frozenset({"Chinese", "Japanese", "Korean"})
"""Languages whose answers legitimately contain CJK characters."""

SYSTEM_PROMPT_TEMPLATE: Final = """\
You are {name}, a voice assistant running on the user's own computer.
Your tone is {tone}. Speak with {voice_style}.
Always reply in {language} ({native_language}), even when the user's words are unclear, \
in another language or look misheard; never switch to Chinese or any other language.

Everything you write is read aloud by text-to-speech:
- Use short, plain sentences. No markdown, lists, headings, code blocks, links or emoji.
- Be brief. Give detail only when the user asks for it.

Tools:
- Use the tools whenever they can answer the question or do the task.
  Call several in one reply when they do not depend on each other.
- The system itself asks the user before any destructive, sending or spending action,
  so just call the tool; do not ask for permission yourself.
- Never invent tool results. If a tool fails or the user declines, say so plainly.
- A failed or declined tool did nothing: never say it is done or describe output it never returned.
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
        language: Natural language to answer in, in English ("Swedish").
        native_language: The same language named in itself ("svenska").
    """

    name: str
    tone: str
    voice_style: str = DEFAULT_VOICE_STYLE
    language: str = DEFAULT_LANGUAGE
    native_language: str = DEFAULT_LANGUAGE

    @property
    def allows_cjk(self) -> bool:
        """Whether answers may contain Chinese, Japanese or Korean script."""
        return self.language in CJK_LANGUAGES

    @classmethod
    def from_settings(cls, settings: Settings) -> Persona:
        """Build the persona from settings; the language follows the TTS voice."""
        language, native = languages_of_voice(settings.tts_voice)
        return cls(
            name=settings.assistant_name,
            tone=settings.persona_tone,
            language=language,
            native_language=native,
        )

    def system_prompt(self, memory: str = "") -> str:
        """The system message for one turn, with recalled memory appended if any."""
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            name=self.name,
            tone=self.tone,
            language=self.language,
            native_language=self.native_language,
            voice_style=self.voice_style,
        )
        if memory.strip():
            prompt += MEMORY_SECTION_TEMPLATE.format(memory=memory.strip())
        return prompt


def language_of_voice(voice_id: str) -> str:
    """``en-GB-RyanNeural`` -> ``English``; unknown prefixes fall back to English."""
    return languages_of_voice(voice_id)[0]


def languages_of_voice(voice_id: str) -> tuple[str, str]:
    """``sv-SE-MattiasNeural`` -> ``("Swedish", "svenska")``; unknown -> English."""
    prefix = voice_id.split("-", 1)[0].lower()
    return _LANGUAGE_NAMES.get(prefix, (DEFAULT_LANGUAGE, DEFAULT_LANGUAGE))


# Design notes:
# - The language is derived from the TTS voice rather than a separate
#   setting: answering in German with an English voice sounds broken, so
#   the two must agree, and one knob cannot disagree with itself.
# - The language sentence names the language twice, in English and in
#   itself ("Swedish (svenska)"), and names the observed failure ("never
#   switch to Chinese"): qwen2.5:7b answered a misheard English line with
#   Chinese when the prompt only said "Answer in Swedish". The agent loop
#   also drops CJK sentences for non-CJK languages (agent/language_guard.py),
#   because a prompt lowers the odds but cannot guarantee anything.
# - "Just call the tool" is deliberate. Confirmation is enforced centrally
#   (ADR-0006); a model that also asks "shall I?" makes the user confirm
#   twice, once to the model and once to the policy.
# - The failed/declined sentence exists because a 7B model answered
#   "Done. It printed: ansel" after the user refused its shell command. The
#   loop enforces the same rule deterministically (agent/loop.py); the
#   prompt line makes the model less likely to need the enforcement.
# - Memory goes into the system prompt, not into fake user/assistant
#   messages, so the model can tell remembered context from what the user
#   just said, and so history never violates a provider's role alternation.
