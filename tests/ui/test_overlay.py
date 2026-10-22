from types import ModuleType

import pytest

from jarvis import _native
from jarvis.ui.overlay import (
    NativeOverlay,
    Overlay,
    OverlayConfig,
    OverlayCorner,
    OverlayError,
    OverlayIntensity,
    OverlaySize,
    OverlayState,
    OverlayTheme,
    StubOverlay,
    create_native_overlay,
    create_overlay,
    default_microphone_name,
)
from jarvis.utils.errors import JarvisError, NativeUnavailableError
from tests.ui.fakes import FakeNativeOverlay


@pytest.fixture
def native() -> FakeNativeOverlay:
    return FakeNativeOverlay()


@pytest.fixture
def overlay(native: FakeNativeOverlay) -> Overlay:
    return Overlay(native)


def test_state_names_mirror_the_native_enum() -> None:
    assert [state.value for state in OverlayState] == ["IDLE", "LISTENING", "THINKING", "SPEAKING"]
    assert [theme.value for theme in OverlayTheme] == [
        "cyan",
        "amber",
        "red",
        "green",
        "violet",
        "ice",
        "pink",
        "custom",
    ]
    assert [corner.value for corner in OverlayCorner] == [
        "TOP_LEFT",
        "TOP_RIGHT",
        "BOTTOM_LEFT",
        "BOTTOM_RIGHT",
    ]


def test_overlay_starts_idle_and_running(overlay: Overlay) -> None:
    assert overlay.state is OverlayState.IDLE
    assert overlay.running
    assert overlay.visible


def test_set_state_forwards_the_member_name(overlay: Overlay, native: FakeNativeOverlay) -> None:
    overlay.set_state(OverlayState.LISTENING)

    assert overlay.state is OverlayState.LISTENING
    assert native.calls == [("set_state", "LISTENING")]


def test_text_and_levels_are_forwarded_in_order(
    overlay: Overlay, native: FakeNativeOverlay
) -> None:
    overlay.push_level(0.4)
    overlay.set_transcript("hello")
    overlay.set_response("hi there")
    overlay.flash_wake_word()

    assert native.calls == [
        ("push_level", 0.4),
        ("set_transcript", "hello"),
        ("set_response", "hi there"),
        ("wake_pulse", None),
    ]


def test_levels_are_coerced_to_float(overlay: Overlay, native: FakeNativeOverlay) -> None:
    overlay.push_level(1)

    assert native.calls == [("push_level", 1.0)]
    assert isinstance(native.calls[0][1], float)


def test_show_and_hide_toggle_visibility(overlay: Overlay, native: FakeNativeOverlay) -> None:
    overlay.hide()
    assert not overlay.visible

    overlay.show()
    assert overlay.visible
    assert native.names() == ["set_visible", "set_visible"]


def test_close_is_idempotent_and_stops_running(overlay: Overlay, native: FakeNativeOverlay) -> None:
    overlay.close()
    overlay.close()

    assert not overlay.running
    assert native.names() == ["close"]


def test_setters_after_close_are_ignored(overlay: Overlay, native: FakeNativeOverlay) -> None:
    overlay.close()

    overlay.set_state(OverlayState.SPEAKING)
    overlay.push_level(0.5)
    overlay.set_transcript("late")
    overlay.set_response("late")
    overlay.flash_wake_word()
    overlay.show()
    overlay.hide()

    assert native.names() == ["close"]
    assert overlay.state is OverlayState.SPEAKING  # the facade still remembers


def test_context_manager_closes(native: FakeNativeOverlay) -> None:
    with Overlay(native) as overlay:
        assert overlay.running
    assert not overlay.running
    assert native.names() == ["close"]


def test_running_reflects_a_dead_native_thread(overlay: Overlay, native: FakeNativeOverlay) -> None:
    native._running = False

    assert not overlay.running


def test_stub_overlay_satisfies_the_protocol_and_records_values() -> None:
    stub: NativeOverlay = StubOverlay()
    overlay = Overlay(stub)

    overlay.set_state(OverlayState.THINKING)
    overlay.push_level(0.2)
    overlay.set_transcript("what time is it")
    overlay.set_response("It is noon.")
    overlay.flash_wake_word()
    overlay.hide()

    assert isinstance(stub, StubOverlay)
    assert stub.state == "THINKING"
    assert stub.levels == [0.2]
    assert stub.transcript == "what time is it"
    assert stub.response == "It is noon."
    assert stub.wake_flashes == 1
    assert not stub.visible()
    assert stub.running()

    overlay.close()
    assert not stub.running()


def test_wake_pulse_and_its_bridge_alias_forward_the_same_call(
    overlay: Overlay, native: FakeNativeOverlay
) -> None:
    overlay.wake_pulse()
    overlay.flash_wake_word()

    assert native.names() == ["wake_pulse", "wake_pulse"]


def test_play_boot_is_idempotent_and_reports_who_started_it(
    overlay: Overlay, native: FakeNativeOverlay
) -> None:
    assert overlay.play_boot()
    assert not overlay.play_boot()

    assert native.calls == [("play_boot", True), ("play_boot", False)]


def test_play_boot_passes_the_readout_in_order(overlay: Overlay, native: FakeNativeOverlay) -> None:
    assert overlay.play_boot({"MODEL": "qwen2.5:7b", "MODE": "CLOUD", "MIC": "Mic"})

    assert native.calls == [
        ("play_boot", True),
        ("boot_info", [("MODEL", "qwen2.5:7b"), ("MODE", "CLOUD"), ("MIC", "Mic")]),
    ]


def test_stub_keeps_the_first_readout_only() -> None:
    stub = StubOverlay()

    assert stub.play_boot([("MODE", "LOCAL")])
    assert not stub.play_boot([("MODE", "CLOUD")])
    assert stub.boot_info == [("MODE", "LOCAL")]
    assert not stub.bloom()
    assert "stub" in stub.bloom_reason()


def test_bloom_reflects_the_native_window(overlay: Overlay, native: FakeNativeOverlay) -> None:
    assert overlay.bloom
    native._transparent = False
    assert not overlay.bloom


def test_microphone_name_is_none_without_the_native_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str) -> ModuleType:
        raise NativeUnavailableError(name, "not built")

    monkeypatch.setattr(_native, "load", unavailable)

    assert default_microphone_name() is None


def test_play_boot_is_a_no_op_when_the_animation_is_disabled(native: FakeNativeOverlay) -> None:
    overlay = Overlay(native, boot_animation=False)

    assert not overlay.play_boot()
    assert native.calls == []


def test_boot_calls_after_close_do_nothing(overlay: Overlay, native: FakeNativeOverlay) -> None:
    overlay.close()

    assert not overlay.play_boot()
    assert not overlay.skip_boot()
    assert native.names() == ["close"]


def test_replay_boot_always_restarts_with_the_new_readout(
    overlay: Overlay, native: FakeNativeOverlay
) -> None:
    assert overlay.play_boot({"MIC": "Old"})

    assert overlay.replay_boot({"MIC": "New", "MODE": "CLOUD"})
    assert overlay.replay_boot()

    assert native.calls[-2:] == [
        ("replay_boot", [("MIC", "New"), ("MODE", "CLOUD")]),
        ("replay_boot", []),
    ]


def test_replay_boot_respects_the_setting_and_close(native: FakeNativeOverlay) -> None:
    disabled = Overlay(native, boot_animation=False)
    assert not disabled.replay_boot({"MODE": "CLOUD"})
    closed = Overlay(native)
    closed.close()
    assert not closed.replay_boot()
    assert native.names() == ["close"]


def test_stub_records_replays_and_their_readout() -> None:
    stub = StubOverlay()
    assert stub.play_boot([("MODE", "LOCAL")])
    assert stub.skip_boot()

    stub.replay_boot([("MODE", "CLOUD")])

    assert stub.boot_replays == 1
    assert stub.boot_info == [("MODE", "CLOUD")]
    assert stub.skip_boot()  # the replay can be skipped again


def test_skip_boot_forwards(overlay: Overlay, native: FakeNativeOverlay) -> None:
    overlay.play_boot()

    assert overlay.skip_boot()
    assert native.names() == ["play_boot", "skip_boot"]


def test_transparent_reflects_the_native_surface(
    overlay: Overlay, native: FakeNativeOverlay
) -> None:
    assert overlay.transparent
    native._transparent = False
    assert not overlay.transparent


def test_stub_boot_mirrors_the_native_rules() -> None:
    stub = StubOverlay()

    assert not stub.skip_boot()  # nothing to skip yet
    assert stub.play_boot()
    assert not stub.play_boot()
    assert stub.skip_boot()
    assert not stub.skip_boot()
    assert stub.boot_started
    assert stub.boot_skipped
    assert not stub.transparent()
    assert "stub" in stub.fallback_reason()


def test_disabled_overlay_keeps_the_boot_setting() -> None:
    overlay = create_overlay(OverlayConfig(boot_animation=False), enabled=False)

    assert not overlay.play_boot()
    assert isinstance(overlay._native, StubOverlay)
    assert not overlay._native.boot_started


def test_overlay_error_is_a_jarvis_error() -> None:
    assert issubclass(OverlayError, JarvisError)


def test_config_defaults_are_the_documented_ones() -> None:
    config = OverlayConfig()

    assert config.corner is OverlayCorner.BOTTOM_RIGHT
    assert config.size is OverlaySize.LARGE
    assert (config.width, config.height, config.margin) == (None, None, 24)
    assert config.intensity is OverlayIntensity.NORMAL
    assert config.bloom
    assert config.click_through
    assert config.opacity == 235
    assert (config.active_fps, config.idle_fps) == (60, 15)
    assert (config.level_history, config.max_text_chars) == (64, 240)
    assert config.transparent
    assert config.theme is OverlayTheme.CYAN
    assert config.boot_animation


def test_create_overlay_disabled_returns_a_stub() -> None:
    overlay = create_overlay(enabled=False)

    assert isinstance(overlay._native, StubOverlay)


def test_create_overlay_falls_back_to_a_stub_when_native_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str) -> ModuleType:
        raise NativeUnavailableError(name, "not built")

    monkeypatch.setattr(_native, "load", unavailable)

    overlay = create_overlay()

    assert isinstance(overlay._native, StubOverlay)
    assert overlay.running


def test_create_overlay_falls_back_to_a_stub_when_the_window_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_device(config: OverlayConfig) -> Overlay:
        raise OverlayError("no device")

    monkeypatch.setattr("jarvis.ui.overlay.create_native_overlay", no_device)

    overlay = create_overlay()

    assert isinstance(overlay._native, StubOverlay)


# --- against the compiled module -------------------------------------------

pytestmark_native = pytest.mark.native


@pytest.fixture
def jarvis_overlay() -> ModuleType:
    try:
        return _native.load("jarvis_overlay")
    except NativeUnavailableError:
        pytest.skip("jarvis_overlay is not built")


@pytestmark_native
def test_native_module_mirrors_the_python_enums(jarvis_overlay: ModuleType) -> None:
    assert set(jarvis_overlay.AssistantState.__members__) == {s.value for s in OverlayState}
    assert set(jarvis_overlay.Corner.__members__) == {c.value for c in OverlayCorner}
    assert set(jarvis_overlay.Theme.__members__) == {t.name for t in OverlayTheme}
    assert set(jarvis_overlay.Intensity.__members__) == {i.name for i in OverlayIntensity}
    assert set(jarvis_overlay.PanelSize.__members__) == {s.name for s in OverlaySize}
    assert 3.5 < jarvis_overlay.BOOT_DURATION_SECONDS < 4.5
    assert issubclass(jarvis_overlay.OverlayError, jarvis_overlay.NativeError)
    assert "step 13" in jarvis_overlay.IMPLEMENTATION_STATUS


@pytestmark_native
def test_native_window_round_trips_state_through_the_facade(jarvis_overlay: ModuleType) -> None:
    config = OverlayConfig(corner=OverlayCorner.TOP_LEFT, opacity=200)
    try:
        overlay = create_native_overlay(config)
    except OverlayError as exc:
        pytest.skip(f"overlay window unavailable here: {exc}")

    with overlay:
        assert overlay.running
        overlay.set_state(OverlayState.LISTENING)
        overlay.push_level(0.5)
        overlay.set_transcript("smoke test")
        overlay.set_response("ok")
        overlay.flash_wake_word()
        overlay.hide()

        snapshot = overlay._native.snapshot()  # type: ignore[attr-defined]
        assert snapshot.state == jarvis_overlay.AssistantState.LISTENING
        assert snapshot.levels[-1] == pytest.approx(0.5)
        assert snapshot.transcript == "smoke test"
        assert snapshot.response == "ok"
        assert snapshot.wake_flash > 0
        assert not snapshot.visible
        assert not overlay.visible
        assert overlay.transparent == (overlay._native.fallback_reason() == "")  # type: ignore[attr-defined]
    assert not overlay.running


@pytestmark_native
def test_native_boot_is_non_blocking_and_idempotent(jarvis_overlay: ModuleType) -> None:
    try:
        overlay = create_native_overlay(OverlayConfig(theme=OverlayTheme.AMBER))
    except OverlayError as exc:
        pytest.skip(f"overlay window unavailable here: {exc}")

    with overlay:
        assert overlay.play_boot({"MODEL": "test-model", "MODE": "CLOUD"})
        assert not overlay.play_boot()
        snapshot = overlay._native.snapshot()  # type: ignore[attr-defined]
        assert snapshot.boot_started
        assert snapshot.boot_readout == [("MODEL", "test-model"), ("MODE", "CLOUD")]
        assert not snapshot.boot_skipped
        assert overlay.skip_boot()
        assert overlay._native.snapshot().boot_skipped  # type: ignore[attr-defined]
        stats = overlay._native.frame_stats()  # type: ignore[attr-defined]
        assert stats.frames >= 0
        assert overlay.replay_boot({"MODEL": "again"})
        replay = overlay._native.snapshot()  # type: ignore[attr-defined]
        assert replay.boot_started
        assert not replay.boot_skipped
        assert replay.boot_readout == [("MODEL", "again")]


@pytestmark_native
def test_native_set_state_rejects_unknown_names(jarvis_overlay: ModuleType) -> None:
    try:
        window = jarvis_overlay.OverlayWindow()
    except jarvis_overlay.OverlayError as exc:
        pytest.skip(f"overlay window unavailable here: {exc}")
    with window:
        window.set_state(jarvis_overlay.AssistantState.SPEAKING)
        assert window.state() == jarvis_overlay.AssistantState.SPEAKING
        with pytest.raises(ValueError, match="unknown assistant state"):
            window.set_state("DANCING")
