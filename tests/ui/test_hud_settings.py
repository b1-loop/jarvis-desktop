"""The HUD settings page: classification, validation, persistence and live apply."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from jarvis.config import Settings
from jarvis.ui.hud_protocol import SettingKind, SettingsPage
from jarvis.ui.hud_settings import (
    SETTING_DEFS,
    START_WITH_WINDOWS,
    AppearanceSwitch,
    Apply,
    SettingsController,
    apply_mode,
    from_text,
    to_text,
)
from jarvis.ui.model_catalog import EDGE_VOICES, ModelCatalogError, OllamaModels, with_current
from jarvis.ui.overlay import OverlayIntensity, OverlaySize, OverlayTheme
from jarvis.user_settings import UserSettingsStore, user_settings_path

MakeSettings = Callable[..., Settings]


class PageView:
    def __init__(self) -> None:
        self.pages: list[SettingsPage] = []

    def set_settings(self, page: SettingsPage) -> None:
        self.pages.append(page)

    def item(self, key: str) -> Any:
        return next(item for item in self.pages[-1].items if item.key == key)


class FakeModels:
    def __init__(self, names: list[str] | None = None) -> None:
        self._names = names

    async def names(self) -> list[str]:
        if self._names is None:
            raise ModelCatalogError("Ollama model list unavailable: connection refused")
        return self._names


class FakeAutostart:
    def __init__(self) -> None:
        self.on = False

    def enabled(self) -> bool:
        return self.on

    def set_enabled(self, enabled: bool) -> None:
        self.on = enabled


def controller(
    settings: Settings,
    tmp_path: Path,
    appliers: dict[str, Callable[[Any], None]] | None = None,
    **kwargs: Any,
) -> tuple[SettingsController, PageView, UserSettingsStore]:
    view = PageView()
    store = UserSettingsStore(user_settings_path(tmp_path))
    built = SettingsController(
        settings=settings, store=store, view=view, appliers=appliers or {}, **kwargs
    )
    return built, view, store


def test_live_and_restart_settings_are_classified() -> None:
    live = {d.key for d in SETTING_DEFS if d.apply is Apply.LIVE}
    restart = {d.key for d in SETTING_DEFS if d.apply is Apply.RESTART}

    assert {"overlay_size", "overlay_theme", "overlay_intensity", "tts_voice"} <= live
    assert {"speak_replies", "wake_word_threshold"} <= live
    assert restart == {"llm_provider", "ollama_model", "stt_language", "wake_word_enabled"}
    assert apply_mode("llm_provider") is Apply.RESTART
    with pytest.raises(KeyError):
        apply_mode("no_such_setting")


def test_values_travel_as_text_both_ways() -> None:
    defs = {d.key: d for d in SETTING_DEFS}
    assert to_text(True) == "true"
    assert to_text(0.8) == "0.80"
    assert to_text(None) == "auto"
    assert from_text(defs["speak_replies"], "false") is False
    assert from_text(defs["wake_word_threshold"], "0.75") == 0.75
    assert from_text(defs["stt_language"], "auto") is None
    assert from_text(defs["stt_language"], "sv") == "sv"


async def test_a_live_change_is_applied_saved_and_shown(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    applied: list[object] = []
    ctrl, view, store = controller(
        make_settings(), tmp_path, appliers={"overlay_theme": applied.append}
    )

    await ctrl.change("overlay_theme", "amber")

    assert applied == ["amber"]
    assert store.load() == {"overlay_theme": "amber"}
    assert view.item("overlay_theme").value == "amber"
    assert view.pages[-1].notice == "Saved and applied."
    assert not view.pages[-1].restart_pending


async def test_a_restart_only_change_is_saved_and_flags_a_restart(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    ctrl, view, store = controller(make_settings(), tmp_path)

    await ctrl.change("stt_language", "sv")

    assert store.load() == {"stt_language": "sv"}
    assert view.pages[-1].restart_pending
    assert view.item("stt_language").restart_required
    assert ctrl.restart_pending


async def test_auto_language_is_saved_as_no_language(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    ctrl, _view, store = controller(make_settings(stt_language="sv"), tmp_path)

    await ctrl.change("stt_language", "auto")

    assert store.load() == {"stt_language": None}


async def test_a_live_key_without_an_applier_applies_after_restart(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    ctrl, view, _store = controller(make_settings(), tmp_path)  # e.g. Piper: no live voice

    await ctrl.change("tts_voice", "sv-SE-MattiasNeural")

    assert not ctrl.is_live("tts_voice")
    assert view.item("tts_voice").restart_required
    assert view.pages[-1].restart_pending


async def test_an_invalid_value_is_rejected_with_a_notice(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    applied: list[object] = []
    ctrl, view, store = controller(
        make_settings(), tmp_path, appliers={"wake_word_threshold": applied.append}
    )

    await ctrl.change("wake_word_threshold", "1.5")
    await ctrl.change("overlay_size", "gigantic")

    assert applied == []
    assert store.load() == {}
    assert "not a valid value" in view.pages[-1].notice
    assert view.item("wake_word_threshold").value == "0.80"


async def test_an_unknown_key_is_a_protocol_error(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    ctrl, _view, _store = controller(make_settings(), tmp_path)
    with pytest.raises(KeyError):
        await ctrl.change("anthropic_api_key", "sk-x")


async def test_refresh_lists_installed_models_and_keeps_the_current_one(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    ctrl, view, _store = controller(
        make_settings(ollama_model="qwen2.5:7b"),
        tmp_path,
        models=FakeModels(["llama3.2:3b", "mistral:7b"]),
    )

    await ctrl.refresh()

    values = [value for value, _ in view.item("ollama_model").choices]
    assert values == ["llama3.2:3b", "mistral:7b", "qwen2.5:7b"]


async def test_refresh_shows_why_models_are_missing(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    ctrl, view, _store = controller(make_settings(), tmp_path, models=FakeModels(None))

    await ctrl.refresh()

    assert "unavailable" in view.item("ollama_model").note


async def test_start_with_windows_drives_the_shortcut_and_is_not_saved(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    autostart = FakeAutostart()
    ctrl, view, store = controller(make_settings(), tmp_path, autostart=autostart)
    ctrl.push()
    assert view.item(START_WITH_WINDOWS).value == "false"

    await ctrl.change(START_WITH_WINDOWS, "true")

    assert autostart.on
    assert store.load() == {}
    assert view.item(START_WITH_WINDOWS).value == "true"


def test_start_with_windows_is_hidden_without_a_shortcut(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    ctrl, _view, _store = controller(make_settings(), tmp_path)
    assert START_WITH_WINDOWS not in {item.key for item in ctrl.page().items}


def test_the_page_offers_swedish_and_english_voices(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    ctrl, _view, _store = controller(make_settings(), tmp_path)
    voice = next(item for item in ctrl.page().items if item.key == "tts_voice")

    values = [value for value, _ in voice.choices]
    assert voice.kind is SettingKind.CHOICE
    assert any(v.startswith("sv-SE") for v in values)
    assert any(v.startswith("en-GB") for v in values) and any(v.startswith("en-US") for v in values)


def test_with_current_keeps_a_value_set_by_hand() -> None:
    choices = with_current(EDGE_VOICES, "de-DE-KatjaNeural")
    assert choices[-1] == ("de-DE-KatjaNeural", "de-DE-KatjaNeural (current)")
    assert with_current(EDGE_VOICES, "en-GB-RyanNeural") == list(EDGE_VOICES)


def test_appearance_switch_sends_all_three_values(make_settings: MakeSettings) -> None:
    calls: list[tuple[OverlayTheme, OverlayIntensity, OverlaySize]] = []

    class View:
        def set_appearance(
            self, theme: OverlayTheme, intensity: OverlayIntensity, size: OverlaySize
        ) -> None:
            calls.append((theme, intensity, size))

    switch = AppearanceSwitch(View(), make_settings())
    switch.set_theme("amber")
    switch.set_size("compact")

    assert calls[-1] == (OverlayTheme.AMBER, OverlayIntensity.NORMAL, OverlaySize.COMPACT)


async def test_ollama_models_are_listed_from_api_tags() -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200, json={"models": [{"name": "qwen2.5:7b"}, {"name": "llama3.2:3b"}, {"x": 1}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        names = await OllamaModels(client, "http://localhost:11434/").names()

    assert names == ["llama3.2:3b", "qwen2.5:7b"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"models": "nope"}),
    ],
)
async def test_ollama_failures_raise_a_catalog_error(response: httpx.Response) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: response)) as client:
        with pytest.raises(ModelCatalogError):
            await OllamaModels(client, "http://localhost:11434").names()


async def test_ollama_down_raises_a_catalog_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as client:
        with pytest.raises(ModelCatalogError):
            await OllamaModels(client, "http://localhost:11434").names()
