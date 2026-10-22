"""The HUD's settings page: what it offers, how edits are validated, saved and applied.

The native side only renders the page and reports edits (ADR-0024). This
module owns everything else:

* **What exists.** :data:`SETTING_DEFS` lists every row: its key (a
  ``Settings`` field, or an action), label, section, widget and whether a
  change can be **applied live** or only **after a restart**.
* **Edits.** :meth:`SettingsController.change` turns the text value into
  the field's type, validates it against the field (the same check the
  user-settings source uses at startup), persists it to
  ``data_dir/user_settings.json``, applies it live when an applier exists
  and otherwise flags "applies after restart", then pushes a fresh page.
* **Lists.** Voices are a curated list; Ollama models are fetched each time
  the page opens (:meth:`SettingsController.refresh`).
* **Colour.** Picking a custom accent also switches the theme to
  ``custom`` (a row's ``implies``), and :meth:`SettingsController.set_colour`
  is the same path for the voice tool ``set_hud_colour`` (ADR-0028).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Protocol

from pydantic import ValidationError

from jarvis.config import Settings
from jarvis.ui.hud_protocol import SettingItem, SettingKind, SettingsPage
from jarvis.ui.model_catalog import EDGE_VOICES, STT_LANGUAGES, ModelCatalogError, with_current
from jarvis.ui.overlay import OverlayIntensity, OverlaySize, OverlayTheme
from jarvis.user_settings import UserSettingsError, UserSettingsStore, validate_field_value
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

AUTO_LANGUAGE: Final = "auto"
TRUE: Final = "true"
FALSE: Final = "false"
START_WITH_WINDOWS: Final = "start_with_windows"
OPEN_LOG: Final = "open_log"
SAVED_NOTICE: Final = "Saved."
LIVE_NOTICE: Final = "Saved and applied."
RESTART_NOTICE: Final = "Saved; applies after a restart."
SLIDER_FORMAT: Final = "{:.2f}"
AUTO_MINIMIZE_CHOICES: Final = (
    ("0.00", "Off"),
    ("30.00", "After 30 s idle"),
    ("60.00", "After 1 min idle"),
    ("120.00", "After 2 min idle"),
    ("300.00", "After 5 min idle"),
)
"""Delays offered for ``hud_auto_minimize_seconds``, spelled as ``to_text`` spells floats;
another value from ``.env`` is still shown (``with_current``)."""
THEME_CHOICES: Final = (
    ("cyan", "Cyan"),
    ("amber", "Amber"),
    ("red", "Red (Mark)"),
    ("green", "Green"),
    ("violet", "Violet"),
    ("ice", "Ice white"),
    ("pink", "Pink"),
    ("custom", "Custom colour"),
)
"""Every ``OverlayTheme``, in the order the page lists them."""
CUSTOM_THEME: Final = "custom"


class SettingRejectedError(JarvisError):
    """A value for a setting did not validate.

    Attributes:
        key: The setting.
        value: The rejected text.
    """

    def __init__(self, key: str, value: str) -> None:
        super().__init__(f"{value!r} is not a valid value for {key}")
        self.key = key
        self.value = value


class Apply(StrEnum):
    """When a changed setting takes effect."""

    LIVE = "live"
    RESTART = "restart"


@dataclass(frozen=True, slots=True)
class SettingDef:
    """One row of the page before values and choices are filled in."""

    key: str
    label: str
    section: str
    kind: SettingKind
    apply: Apply
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.0
    static_choices: tuple[tuple[str, str], ...] = ()
    implies: tuple[tuple[str, str], ...] = ()
    """``(key, text)`` edits that follow a change of this row (an accent implies ``custom``)."""


SETTING_DEFS: Final[tuple[SettingDef, ...]] = (
    SettingDef(
        "tts_voice", "Voice", "VOICE", SettingKind.CHOICE, Apply.LIVE, static_choices=EDGE_VOICES
    ),
    SettingDef("speak_replies", "Speak replies", "VOICE", SettingKind.TOGGLE, Apply.LIVE),
    SettingDef(
        "stt_language",
        "Speech language",
        "VOICE",
        SettingKind.CHOICE,
        Apply.RESTART,
        static_choices=STT_LANGUAGES,
    ),
    SettingDef("wake_word_enabled", "Wake word", "VOICE", SettingKind.TOGGLE, Apply.RESTART),
    SettingDef(
        "wake_word_threshold",
        "Wake word threshold",
        "VOICE",
        SettingKind.SLIDER,
        Apply.LIVE,
        minimum=0.5,
        maximum=0.95,
        step=0.05,
    ),
    SettingDef(
        "llm_provider",
        "LLM provider",
        "MODEL",
        SettingKind.CHOICE,
        Apply.RESTART,
        static_choices=(
            ("ollama", "Ollama (local server)"),
            ("anthropic", "Claude (cloud)"),
            ("local", "llama.cpp (offline)"),
        ),
    ),
    SettingDef("ollama_model", "Ollama model", "MODEL", SettingKind.CHOICE, Apply.RESTART),
    SettingDef(
        "overlay_size",
        "HUD size",
        "OVERLAY",
        SettingKind.CHOICE,
        Apply.LIVE,
        static_choices=(("compact", "Compact"), ("normal", "Normal"), ("large", "Large")),
    ),
    SettingDef(
        "overlay_theme",
        "Theme",
        "OVERLAY",
        SettingKind.CHOICE,
        Apply.LIVE,
        static_choices=THEME_CHOICES,
    ),
    SettingDef(
        "overlay_accent",
        "Custom colour",
        "OVERLAY",
        SettingKind.COLOUR,
        Apply.LIVE,
        implies=(("overlay_theme", CUSTOM_THEME),),
    ),
    SettingDef(
        "overlay_intensity",
        "Intensity",
        "OVERLAY",
        SettingKind.CHOICE,
        Apply.LIVE,
        static_choices=(("calm", "Calm"), ("normal", "Normal"), ("max", "Max")),
    ),
    SettingDef(
        "hud_auto_minimize_seconds",
        "Auto-minimize to the orb",
        "OVERLAY",
        SettingKind.CHOICE,
        Apply.LIVE,
        static_choices=AUTO_MINIMIZE_CHOICES,
    ),
    SettingDef(
        "hud_hide_in_fullscreen", "Hide in full screen", "OVERLAY", SettingKind.TOGGLE, Apply.LIVE
    ),
    SettingDef(START_WITH_WINDOWS, "Start with Windows", "SYSTEM", SettingKind.TOGGLE, Apply.LIVE),
    SettingDef(OPEN_LOG, "Open log file", "SYSTEM", SettingKind.ACTION, Apply.LIVE),
)
"""Every row of the settings page, in display order."""

_DEFS_BY_KEY: Final = {definition.key: definition for definition in SETTING_DEFS}

Applier = Callable[[Any], Awaitable[None] | None]
"""Applies one validated value to the running session; may be a coroutine."""


class SettingsView(Protocol):
    """Where the page is shown (``ui.overlay.Overlay``)."""

    def set_settings(self, page: SettingsPage) -> None:
        """Replace the page."""


class ModelLister(Protocol):
    """Lists installed Ollama models (``ui.model_catalog.OllamaModels``)."""

    async def names(self) -> list[str]:
        """Model names; raises ``ModelCatalogError`` when unavailable."""


class Autostart(Protocol):
    """The "start with Windows" shortcut (``app.autostart``)."""

    def enabled(self) -> bool:
        """Whether the Startup-folder shortcut exists."""

    def set_enabled(self, enabled: bool) -> None:
        """Create or remove it."""


class AppearanceView(Protocol):
    """Where appearance changes go (``ui.overlay.Overlay``)."""

    def set_appearance(
        self,
        theme: OverlayTheme,
        intensity: OverlayIntensity,
        size: OverlaySize,
        accent: str,
    ) -> None:
        """Switch palette (and custom accent), richness and size."""


class AppearanceSwitch:
    """Live appliers for theme, accent, intensity and size: the native call takes all four.

    Args:
        view: The overlay.
        settings: The starting values.
    """

    def __init__(self, view: AppearanceView, settings: Settings) -> None:
        self._view = view
        self._theme = OverlayTheme(settings.overlay_theme)
        self._accent = settings.overlay_accent
        self._intensity = OverlayIntensity(settings.overlay_intensity)
        self._size = OverlaySize(settings.overlay_size)

    def set_theme(self, theme: object) -> None:
        """Apply a validated ``overlay_theme``."""
        self._theme = OverlayTheme(str(theme))
        self._push()

    def set_accent(self, accent: object) -> None:
        """Apply a validated ``overlay_accent`` (``#rrggbb``)."""
        self._accent = str(accent)
        self._push()

    def set_intensity(self, intensity: object) -> None:
        """Apply a validated ``overlay_intensity``."""
        self._intensity = OverlayIntensity(str(intensity))
        self._push()

    def set_size(self, size: object) -> None:
        """Apply a validated ``overlay_size``."""
        self._size = OverlaySize(str(size))
        self._push()

    def _push(self) -> None:
        self._view.set_appearance(self._theme, self._intensity, self._size, self._accent)


def apply_mode(key: str) -> Apply:
    """Whether a change of ``key`` can take effect live.

    Raises:
        KeyError: If ``key`` is not on the settings page.
    """
    return _DEFS_BY_KEY[key].apply


def to_text(value: object) -> str:
    """How a setting's value is shown and reported by the native page."""
    if value is None:
        return AUTO_LANGUAGE
    if isinstance(value, bool):
        return TRUE if value else FALSE
    if isinstance(value, float):
        return SLIDER_FORMAT.format(value)
    return str(value)


def from_text(definition: SettingDef, text: str) -> object:
    """The Python value of ``text`` for ``definition``'s widget, before validation."""
    if definition.kind is SettingKind.TOGGLE:
        return text.strip().lower() == TRUE
    if definition.kind is SettingKind.SLIDER:
        return float(text)
    if definition.key == "stt_language" and text == AUTO_LANGUAGE:
        return None
    return text


@dataclass(slots=True)
class _PageState:
    values: dict[str, object] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)
    model_note: str = ""
    restart_pending: bool = False
    notice: str = ""


class SettingsController:
    """Validates, persists and applies edits from the HUD's settings page.

    Args:
        settings: The session's settings (initial values).
        store: ``user_settings.json``.
        view: Where the page is shown.
        appliers: Live appliers by key; a LIVE key without one is treated
            as restart-only (e.g. the voice in LOCAL mode, where Piper runs).
        models: Ollama model lister, or None when Ollama is not the provider.
        autostart: The Startup shortcut, or None where it is unavailable.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        store: UserSettingsStore,
        view: SettingsView,
        appliers: Mapping[str, Applier],
        models: ModelLister | None = None,
        autostart: Autostart | None = None,
    ) -> None:
        self._store = store
        self._view = view
        self._appliers = dict(appliers)
        self._models = models
        self._autostart = autostart
        if autostart is not None:
            # PowerShell writes the shortcut: seconds, so never on the loop.
            self._appliers[START_WITH_WINDOWS] = lambda on: asyncio.to_thread(
                autostart.set_enabled, bool(on)
            )
        self._state = _PageState(values=self._initial_values(settings))

    @property
    def restart_pending(self) -> bool:
        """Whether a restart-only setting changed this session."""
        return self._state.restart_pending

    def value(self, key: str) -> object:
        """The current value of ``key`` as this session knows it."""
        return self._state.values.get(key)

    def is_live(self, key: str) -> bool:
        """Whether a change of ``key`` applies now (LIVE and an applier exists)."""
        return apply_mode(key) is Apply.LIVE and key in self._appliers

    async def refresh(self) -> None:
        """Re-list the Ollama models and push the page."""
        if self._models is not None:
            try:
                self._state.models = await self._models.names()
                self._state.model_note = ""
            except ModelCatalogError as error:
                log.warning("hud.settings.models_unavailable", error=str(error))
                self._state.model_note = str(error)
        self.push()

    def push(self) -> None:
        """Show the page as it stands."""
        self._view.set_settings(self.page())

    def show_notice(self, text: str) -> None:
        """Put ``text`` on the page's status line."""
        self._state.notice = text
        self.push()

    def page(self) -> SettingsPage:
        """The page with current values, choices and notes."""
        return SettingsPage(
            items=tuple(
                self._item(definition) for definition in SETTING_DEFS if self._shown(definition)
            ),
            restart_pending=self._state.restart_pending,
            notice=self._state.notice,
        )

    async def change(self, key: str, text: str) -> None:
        """Handle one edit from the page, then push the page again.

        Invalid values are rejected with a notice (the page shows the old
        value again); unknown keys are a protocol error and raise.

        Raises:
            KeyError: If ``key`` is not a setting on the page.
        """
        definition = _DEFS_BY_KEY[key]
        try:
            value = self._validated(definition, text)
        except (ValidationError, ValueError) as error:
            log.warning("hud.settings.rejected", key=key, value=text, error=str(error))
            self._state.notice = f"{definition.label}: not a valid value ({text})."
            self.push()
            return
        await self._commit(definition, value)
        await self._commit_implied(definition)
        self.push()

    async def set_colour(self, theme: str, accent: str | None = None) -> None:
        """Switch the HUD's palette as if picked on the page: validated, saved, applied.

        The voice tool's way in (``tools/set_hud_colour.py``). With an
        ``accent`` the custom accent is set first, so ``theme="custom"``
        shows it.

        Raises:
            SettingRejectedError: If ``theme`` or ``accent`` is not a valid value.
        """
        if accent is not None:
            await self._commit_text("overlay_accent", accent)
        await self._commit_text("overlay_theme", theme)
        self.push()

    async def _commit_text(self, key: str, text: str) -> None:
        definition = _DEFS_BY_KEY[key]
        try:
            value = self._validated(definition, text)
        except (ValidationError, ValueError) as error:
            raise SettingRejectedError(key, text) from error
        await self._commit(definition, value)

    async def _commit_implied(self, definition: SettingDef) -> None:
        for key, text in definition.implies:
            if to_text(self._state.values.get(key)) != text:
                implied = _DEFS_BY_KEY[key]
                await self._commit(implied, self._validated(implied, text))

    def _shown(self, definition: SettingDef) -> bool:
        return definition.key != START_WITH_WINDOWS or self._autostart is not None

    def _validated(self, definition: SettingDef, text: str) -> object:
        value = from_text(definition, text)
        if definition.key == START_WITH_WINDOWS:
            return value
        info = Settings.model_fields[definition.key]
        return validate_field_value(info, value)

    async def _commit(self, definition: SettingDef, value: object) -> None:
        key = definition.key
        self._state.values[key] = value
        if self.is_live(key):
            self._state.notice = LIVE_NOTICE
            await self._apply(key, value)
        else:
            self._state.restart_pending = True
            self._state.notice = RESTART_NOTICE
        if key != START_WITH_WINDOWS:
            await self._persist(key, value)
        log.info("hud.settings.changed", key=key, live=self.is_live(key))

    async def _apply(self, key: str, value: object) -> None:
        try:
            result = self._appliers[key](value)
            if inspect.isawaitable(result):
                await result
        except JarvisError as error:
            log.exception("hud.settings.apply_failed", key=key)
            self._state.notice = f"Could not apply: {error}"

    async def _persist(self, key: str, value: object) -> None:
        try:
            await asyncio.to_thread(self._store.update, key, _jsonable(value))
        except (UserSettingsError, OSError) as error:
            # The session still uses the value; say that it will not survive.
            log.exception("hud.settings.not_saved", key=key)
            self._state.notice = f"Applied but not saved: {error}"

    def _item(self, definition: SettingDef) -> SettingItem:
        value = self._state.values.get(definition.key)
        restart = definition.kind is not SettingKind.ACTION and not self.is_live(definition.key)
        return SettingItem(
            key=definition.key,
            label=definition.label,
            section=definition.section,
            kind=definition.kind,
            value="" if definition.kind is SettingKind.ACTION else to_text(value),
            choices=self._choices(definition, to_text(value)),
            minimum=definition.minimum,
            maximum=definition.maximum,
            step=definition.step,
            restart_required=restart,
            note=self._state.model_note if definition.key == "ollama_model" else "",
        )

    def _choices(self, definition: SettingDef, current: str) -> tuple[tuple[str, str], ...]:
        if definition.kind is not SettingKind.CHOICE:
            return ()
        if definition.key == "ollama_model":
            models = tuple((name, name) for name in self._state.models)
            return tuple(with_current(models, current))
        return tuple(with_current(definition.static_choices, current))

    def _initial_values(self, settings: Settings) -> dict[str, object]:
        values: dict[str, object] = {
            definition.key: getattr(settings, definition.key)
            for definition in SETTING_DEFS
            if definition.key in Settings.model_fields
        }
        if self._autostart is not None:
            values[START_WITH_WINDOWS] = self._autostart.enabled()
        return values


def _jsonable(value: object) -> object:
    """``value`` as JSON can hold it (StrEnum and Path become text)."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


# Design notes:
# - LIVE is a property of the setting *and* of the session: the voice is
#   live with edge-tts but not with Piper, the threshold only when the wake
#   word engine runs. The controller therefore asks for an applier per key
#   and falls back to "applies after restart" when there is none, so the
#   page never claims to have applied something it could not.
# - Values are validated with the same per-field check the startup source
#   uses (jarvis.user_settings), so what the HUD saves is exactly what the
#   next start accepts.
# - A row's ``implies`` keeps the page's two colour controls consistent:
#   picking a custom colour while a preset is shown would otherwise save an
#   accent nobody sees. The reverse is deliberately absent; choosing a
#   preset keeps the accent, so "Custom colour" brings it back.
# - "Start with Windows" is not persisted: the Startup-folder shortcut is
#   the state, read back at start (app.autostart), so Explorer and the HUD
#   can never disagree.
