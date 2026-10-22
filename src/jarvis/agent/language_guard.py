"""Keep spoken answers in the configured language: never speak drifted CJK text.

Small local models drift: qwen2.5:7b, configured for Swedish, answered a
misheard English line with "Var god och的具体任务..." and the Swedish
voice tried to read Chinese aloud. The persona prompt asks for the language
explicitly; this module is the deterministic backstop the agent loop uses.
When the configured language is not Chinese, Japanese or Korean, any
sentence containing CJK script is dropped before it reaches the speaker,
and a final reply that drifted is regenerated once with a stricter
instruction (``agent/loop.py``).
"""

from __future__ import annotations

import re
from typing import Final

from jarvis.agent.persona import Persona

_CJK: Final = re.compile(
    "["
    "\u3000-\u303f"  # CJK symbols and punctuation
    "\u3040-\u30ff"  # hiragana and katakana
    "\u3400-\u4dbf"  # CJK unified ideographs extension A
    "\u4e00-\u9fff"  # CJK unified ideographs
    "\uac00-\ud7af"  # hangul syllables
    "\uf900-\ufaff"  # CJK compatibility ideographs
    "\uff00-\uffef"  # half- and full-width forms (the full-width colon U+FF1A)
    "]"
)

STRICT_LANGUAGE_REMINDER: Final = (
    "Your previous reply was not in {language}. Reply again, only in {language} "
    "({native_language}), in plain sentences without any Chinese, Japanese or Korean characters."
)


def contains_cjk(text: str) -> bool:
    """Whether ``text`` contains Chinese, Japanese or Korean script or punctuation."""
    return _CJK.search(text) is not None


class LanguageGuard:
    """Decides which sentences may be spoken for one persona.

    Args:
        persona: Its language decides whether CJK script is legitimate.
    """

    def __init__(self, persona: Persona) -> None:
        self._persona = persona

    def rejects(self, sentence: str) -> bool:
        """True when ``sentence`` drifted into CJK script the persona does not speak."""
        return not self._persona.allows_cjk and contains_cjk(sentence)

    def stricter_prompt(self, system_prompt: str) -> str:
        """``system_prompt`` with the one-time reminder appended for the regeneration."""
        reminder = STRICT_LANGUAGE_REMINDER.format(
            language=self._persona.language, native_language=self._persona.native_language
        )
        return f"{system_prompt}\n{reminder}\n"


# Design notes:
# - A character-class check, not language identification: it costs a regex
#   per sentence, needs no model, and catches exactly the observed failure
#   (Chinese from a Swedish persona). Drift into another Latin-script
#   language is not detected; the persona prompt is the only defence there.
# - The whole sentence is dropped, not only the CJK characters: the Latin
#   prefix of a drifted sentence ("Var god och") is a fragment, and reading
#   it alone would sound broken.
# - One regeneration at most: a model that drifts twice in a row will not
#   be talked out of it, and every retry adds a full LLM round of latency.
