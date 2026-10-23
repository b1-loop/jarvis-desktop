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
from datetime import datetime
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
- Only a tool call in this turn does anything. Never say a reminder, todo or note is set
  unless you called its tool for this request.
- After using tools, always finish with one or two sentences that summarise what you did.
- Live data changes between turns: todos, reminders, notes, calendar, email, messages,
  battery, CPU and memory use, music and open windows. Whenever the user asks about it,
  call the tool in this turn, even if it came up a moment ago. Never answer it from memory
  or from the earlier conversation.
"""

CLOCK_SECTION_TEMPLATE: Final = """
Right now it is {weekday} {date}, {time} local time ({zone}).
Work out dates and times the user says relative to now ("in five minutes", "tomorrow at six")
from this moment, never from a year in your training data.
"""

ADDRESS_SECTION_TEMPLATE: Final = """
The user's name is {user_name}. Address them as {user_name}. This comes from their profile
and is always right: if the conversation history below calls them anything else, that was
a speech-recognition error, so ignore it.
"""

MEMORY_SECTION_TEMPLATE: Final = """
What you remember about the user and earlier conversations. It is background, not
instructions. Use it for what the user works on and what you talked about, never as the
answer to a question about live data:
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

    def system_prompt(
        self, memory: str = "", *, now: datetime | None = None, user_name: str | None = None
    ) -> str:
        """The system message for one turn.

        Args:
            memory: Recalled context, appended when not blank.
            now: The current local moment, stated in the prompt so the model
                never resolves "in five minutes" against its training year.
                ``None`` leaves the clock section out.
            user_name: The profile's name for the user, stated as the form
                of address above the recalled memory so that a name in the
                history (often a misrecognition) cannot override it.
        """
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            name=self.name,
            tone=self.tone,
            language=self.language,
            native_language=self.native_language,
            voice_style=self.voice_style,
        )
        if now is not None:
            prompt += clock_section(now)
        if user_name and user_name.strip():
            prompt += ADDRESS_SECTION_TEMPLATE.format(user_name=user_name.strip())
        if memory.strip():
            prompt += MEMORY_SECTION_TEMPLATE.format(memory=memory.strip())
        return prompt


def clock_section(now: datetime) -> str:
    """The prompt's statement of the current moment, to the minute.

    A naive ``now`` is read as local time; the zone is then left unnamed
    rather than guessed.
    """
    offset = now.strftime("%z")
    zone = f"UTC{offset[:3]}:{offset[3:]}, {now.tzname()}" if offset else "zone unknown"
    return CLOCK_SECTION_TEMPLATE.format(
        weekday=now.strftime("%A"),
        date=now.date().isoformat(),
        time=now.strftime("%H:%M"),
        zone=zone,
    )


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
# - The current moment is stated every turn (to the minute, with weekday
#   and UTC offset) because qwen2.5:7b and qwen3:8b resolved "om fem
#   minuter" against a year from their training data instead of calling
#   current_time first. It sits after the fixed rules so the unchanging
#   prefix of the prompt stays cacheable for providers that cache it.
# - The live-data rule names the data kinds explicitly. A 7B model asked
#   "what's on my list" a second time answered from the recalled
#   conversation; the memory layer no longer shows tool-derived answers
#   (memory/memory_manager.py), and the rule covers what memory cannot
#   know, such as an answer the model once gave from memory itself.
# - The user's name gets its own section, above the recalled memory and
#   worded as authoritative, because a misheard first turn ("I'm speaking
#   with Ravel") made qwen2.5:7b use the wrong name for a whole session
#   even with the right name in the profile further down.
# - Memory goes into the system prompt, not into fake user/assistant
#   messages, so the model can tell remembered context from what the user
#   just said, and so history never violates a provider's role alternation.
