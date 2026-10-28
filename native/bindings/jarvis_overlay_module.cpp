/// @file jarvis_overlay_module.cpp
/// @brief Python module @c jarvis_overlay.
///
/// A thin layer over jarvis::overlay: the window runs its own UI thread and
/// never calls back into Python, so every setter is a plain, cheap forward.
/// Only construction and close() release the GIL, because they wait for
/// that thread.

#include "jarvis/common/errors.hpp"
#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/effects.hpp"
#include "jarvis/overlay/errors.hpp"
#include "jarvis/overlay/hud_data.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/palette.hpp"
#include "jarvis/overlay/placement.hpp"
#include "jarvis/overlay/system_info.hpp"
#include "jarvis/overlay/ui_events.hpp"
#include "jarvis/overlay/window.hpp"

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "module_info.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;
using namespace jarvis::overlay;  // NOLINT(google-build-using-namespace) - binding file only

namespace {

// Translators are consulted most-recently-registered first, so the base
// class goes first and the derived class after it.
void defineExceptions(const py::module_& module) {
    const auto& nativeError = py::register_local_exception<jarvis::NativeError>(
        module, "NativeError", PyExc_RuntimeError);
    py::register_local_exception<OverlayError>(module, "OverlayError", nativeError.ptr());
}

/// The accent for @p hex ("#rrggbb"); empty is the default accent.
Rgba accentFrom(std::string_view hex) {
    if (hex.empty()) {
        return kDefaultAccent;
    }
    const std::optional<Rgba> parsed = parseHexColour(hex);
    if (!parsed.has_value()) {
        throw py::value_error("accent '" + std::string{hex} + "' is not a '#rrggbb' colour");
    }
    return *parsed;
}

py::tuple colourTuple(Rgba colour) {
    return py::make_tuple(colour.r, colour.g, colour.b, colour.a);
}

/// What themeFor() resolves @p theme (and, for custom, @p accent) to.
py::dict paletteOf(std::string_view theme, std::string_view accent) {
    const std::optional<ThemeName> name = parseThemeName(theme);
    if (!name.has_value()) {
        throw py::value_error("unknown theme '" + std::string{theme} + "'");
    }
    const Theme palette = themeFor(*name, accentFrom(accent));
    using py::literals::operator""_a;
    return py::dict(
        "accent"_a = colourTuple(palette.accent), "idle"_a = colourTuple(palette.idle),
        "listening"_a = colourTuple(palette.listening),
        "thinking"_a = colourTuple(palette.thinking), "speaking"_a = colourTuple(palette.speaking),
        "wake_flash"_a = colourTuple(palette.wakeFlash), "text"_a = colourTuple(palette.text),
        "muted_text"_a = colourTuple(palette.mutedText),
        "background"_a = colourTuple(palette.background), "plate"_a = colourTuple(palette.plate),
        "warning"_a = colourTuple(palette.warning), "bloom"_a = palette.bloom);
}

void defineEnums(const py::module_& module) {
    py::enum_<AssistantState>(module, "AssistantState",
                              "What Jarvis is doing; mirrors the Python state machine.")
        .value("IDLE", AssistantState::Idle)
        .value("LISTENING", AssistantState::Listening)
        .value("THINKING", AssistantState::Thinking)
        .value("SPEAKING", AssistantState::Speaking);

    py::enum_<Corner>(module, "Corner", "Screen corner the panel is anchored to.")
        .value("TOP_LEFT", Corner::TopLeft)
        .value("TOP_RIGHT", Corner::TopRight)
        .value("BOTTOM_LEFT", Corner::BottomLeft)
        .value("BOTTOM_RIGHT", Corner::BottomRight);

    py::enum_<ThemeName>(module, "Theme", "Palette of the HUD; CUSTOM derives one from an accent.")
        .value("CYAN", ThemeName::Cyan)
        .value("AMBER", ThemeName::Amber)
        .value("RED", ThemeName::Red)
        .value("GREEN", ThemeName::Green)
        .value("VIOLET", ThemeName::Violet)
        .value("ICE", ThemeName::Ice)
        .value("PINK", ThemeName::Pink)
        .value("CUSTOM", ThemeName::Custom);

    py::enum_<PanelSize>(module, "PanelSize", "How big the corner HUD is.")
        .value("COMPACT", PanelSize::Compact)
        .value("NORMAL", PanelSize::Normal)
        .value("LARGE", PanelSize::Large);

    py::enum_<anim::Intensity>(module, "Intensity",
                               "How rich the HUD is: particles, bloom and motion.")
        .value("CALM", anim::Intensity::Calm)
        .value("NORMAL", anim::Intensity::Normal)
        .value("MAX", anim::Intensity::Max);
}

/// The minimize-to-orb options of OverlayConfig (ADR-0024, update).
void defineMinimizeConfig(py::class_<OverlayConfig>& config) {
    config
        .def_readwrite("start_minimized", &OverlayConfig::startMinimized,
                       "Start as the orb (after the boot, if one plays).")
        .def_readwrite("orb_position", &OverlayConfig::orbPosition,
                       "Remembered orb position (SavedPlacement) or None to dock by the HUD.")
        .def_readwrite("auto_minimize_seconds", &OverlayConfig::autoMinimizeSeconds,
                       "Shrink to the orb after this many quiet seconds; 0 is off.")
        .def_readwrite("hide_in_fullscreen", &OverlayConfig::hideInFullscreen,
                       "Hide while a full-screen application is in front on the HUD's monitor.");
}

/// The palette options of OverlayConfig (ADR-0028).
void defineAppearanceConfig(py::class_<OverlayConfig>& config) {
    config.def_readwrite("theme", &OverlayConfig::theme)
        .def_property(
            "accent", [](const OverlayConfig& self) { return toHex(self.accent); },
            [](OverlayConfig& self, std::string_view hex) { self.accent = accentFrom(hex); },
            "Accent of the CUSTOM theme as '#rrggbb'.");
}

void defineConfig(const py::module_& module) {
    py::class_<ModelLimits>(module, "ModelLimits",
                            "Bounds that keep the overlay's memory independent of its input.")
        .def(py::init<>())
        .def_readwrite("level_history", &ModelLimits::levelHistory)
        .def_readwrite("max_text_chars", &ModelLimits::maxTextChars)
        .def_readwrite("wake_flash", &ModelLimits::wakeFlashDuration,
                       "datetime.timedelta the wake-word flash stays visible.");

    py::class_<OverlayConfig> config(module, "OverlayConfig",
                                     "Geometry and behaviour of the window (sizes at 96 DPI).");
    config.def(py::init<>())
        .def_readwrite("corner", &OverlayConfig::corner)
        .def_readwrite("size", &OverlayConfig::size)
        .def_readwrite("width", &OverlayConfig::width, "Explicit width; 0 takes the size preset's.")
        .def_readwrite("height", &OverlayConfig::height,
                       "Explicit height; 0 takes the size preset's.")
        .def_readwrite("margin", &OverlayConfig::margin)
        .def_readwrite("click_through", &OverlayConfig::clickThrough)
        .def_readwrite("opacity", &OverlayConfig::opacity)
        .def_readwrite("active_fps", &OverlayConfig::activeFps)
        .def_readwrite("idle_fps", &OverlayConfig::idleFps)
        .def_readwrite("transparent", &OverlayConfig::transparent,
                       "Prefer per-pixel transparency (DirectComposition).")
        .def_readwrite("intensity", &OverlayConfig::intensity)
        .def_readwrite("bloom", &OverlayConfig::bloom,
                       "Run the GPU bloom pipeline (falls back to plain additive light).")
        .def_readwrite("title", &OverlayConfig::title)
        .def_readwrite("limits", &OverlayConfig::limits)
        .def_readwrite("interactive", &OverlayConfig::interactive,
                       "Clickable HUD shapes, hover controls, typing, dragging (ADR-0024).")
        .def_readwrite("position", &OverlayConfig::position,
                       "Remembered corner-HUD position (SavedPlacement) or None.")
        .def_readwrite("expanded_bounds", &OverlayConfig::expandedBounds,
                       "Remembered expanded-panel position and size (SavedPlacement) or None.");
    defineMinimizeConfig(config);
    defineAppearanceConfig(config);
}

void definePlacement(const py::module_& module) {
    py::class_<SavedPlacement>(module, "SavedPlacement",
                               "A position in logical pixels from a monitor's work-area corner.")
        .def(py::init<>())
        .def(py::init([](std::string monitor, float x, float y, float width, float height) {
                 return SavedPlacement{.monitor = std::move(monitor),
                                       .x = x,
                                       .y = y,
                                       .width = width,
                                       .height = height};
             }),
             py::arg("monitor"), py::arg("x"), py::arg("y"), py::arg("width") = 0.0F,
             py::arg("height") = 0.0F)
        .def_readwrite("monitor", &SavedPlacement::monitor)
        .def_readwrite("x", &SavedPlacement::x)
        .def_readwrite("y", &SavedPlacement::y)
        .def_readwrite("width", &SavedPlacement::width)
        .def_readwrite("height", &SavedPlacement::height);
}

void defineSnapshot(const py::module_& module) {
    py::class_<OverlaySnapshot>(module, "OverlaySnapshot",
                                "Copy of what the UI thread draws; for tests and the dashboard.")
        .def_readonly("state", &OverlaySnapshot::state)
        .def_readonly("levels", &OverlaySnapshot::levels)
        .def_readonly("transcript", &OverlaySnapshot::transcript)
        .def_readonly("response", &OverlaySnapshot::response)
        .def_readonly("visible", &OverlaySnapshot::visible)
        .def_readonly("wake_flash", &OverlaySnapshot::wakeFlash)
        .def_property_readonly(
            "boot_started",
            [](const OverlaySnapshot& snapshot) { return snapshot.bootStartedAt.has_value(); })
        .def_property_readonly(
            "boot_skipped",
            [](const OverlaySnapshot& snapshot) { return snapshot.bootSkippedAt.has_value(); })
        .def_property_readonly("boot_readout",
                               [](const OverlaySnapshot& snapshot) {
                                   std::vector<std::pair<std::string, std::string>> lines;
                                   if (snapshot.bootReadout != nullptr) {
                                       for (const ReadoutLine& line : *snapshot.bootReadout) {
                                           lines.emplace_back(line.label, line.value);
                                       }
                                   }
                                   return lines;
                               })
        .def_readonly("revision", &OverlaySnapshot::revision);
}

void defineFrameStats(const py::module_& module) {
    py::class_<anim::FrameStats>(module, "FrameStats",
                                 "Smoothed frame timings measured on the UI thread.")
        .def_readonly("frames", &anim::FrameStats::frames)
        .def_readonly("render_ms", &anim::FrameStats::renderMs,
                      "CPU time to build and submit one frame, milliseconds.")
        .def_readonly("interval_ms", &anim::FrameStats::intervalMs,
                      "Time between consecutive frames, milliseconds.")
        .def_readonly("gpu_ms", &anim::FrameStats::gpuMs,
                      "GPU time of one frame from timestamp queries, milliseconds.")
        .def_readonly("gpu_frames", &anim::FrameStats::gpuFrames,
                      "Frames for which a GPU time was measured.")
        .def_readonly("interval_p50_ms", &anim::FrameStats::intervalP50Ms,
                      "Median frame interval over the recent window, milliseconds.")
        .def_readonly("interval_p95_ms", &anim::FrameStats::intervalP95Ms,
                      "95th percentile frame interval, milliseconds.")
        .def_readonly("interval_p99_ms", &anim::FrameStats::intervalP99Ms,
                      "99th percentile frame interval, milliseconds.")
        .def_readonly("interval_max_ms", &anim::FrameStats::intervalMaxMs,
                      "Longest frame interval in the recent window, milliseconds.")
        .def_readonly("render_p99_ms", &anim::FrameStats::renderP99Ms,
                      "99th percentile CPU time per frame, milliseconds.")
        .def_readonly("hitches", &anim::FrameStats::hitches,
                      "Frames that started more than 1.5x their scheduled period late.")
        .def_readonly("presented_frames", &anim::FrameStats::presentedFrames,
                      "Frames presented while the compositor's vblank grid was known.")
        .def_readonly("off_cadence_refreshes", &anim::FrameStats::offCadenceRefreshes,
                      "Refreshes by which presented frames missed their even cadence.")
        .def_readonly("refresh_hz", &anim::FrameStats::refreshHz,
                      "The compositor's refresh rate, Hz (0 until known).");
}

void setStateByName(OverlayWindow& window, std::string_view name) {
    const auto state = parseAssistantState(name);
    if (!state.has_value()) {
        throw py::value_error("unknown assistant state '" + std::string{name} +
                              "'; expected IDLE, LISTENING, THINKING or SPEAKING");
    }
    window.setState(*state);
}

using ReadoutPairs = std::vector<std::pair<std::string, std::string>>;

BootReadout toReadout(const ReadoutPairs& info) {
    BootReadout readout;
    readout.reserve(info.size());
    for (const auto& [label, value] : info) {
        readout.push_back(ReadoutLine{.label = label, .value = value});
    }
    return readout;
}

bool playBoot(OverlayWindow& window, const ReadoutPairs& info) {
    return window.playBoot(toReadout(info));
}

void replayBoot(OverlayWindow& window, const ReadoutPairs& info) {
    window.replayBoot(toReadout(info));
}

// --- interactive HUD (ADR-0024): data in, events out --------------------------

/// dict[key] cast to T; KeyError when the key is missing, so a malformed
/// page from Python fails loudly instead of rendering blanks.
template <class T>
T field(const py::dict& dict, const char* key) {
    return dict.attr("__getitem__")(key).cast<T>();
}

SettingKind settingKindFrom(std::string_view name) {
    if (name == "choice") {
        return SettingKind::Choice;
    }
    if (name == "toggle") {
        return SettingKind::Toggle;
    }
    if (name == "slider") {
        return SettingKind::Slider;
    }
    if (name == "action") {
        return SettingKind::Action;
    }
    if (name == "colour") {
        return SettingKind::Colour;
    }
    throw py::value_error("unknown setting kind '" + std::string{name} +
                          "'; expected choice, toggle, slider, action or colour");
}

SettingItem settingFrom(const py::dict& item) {
    SettingItem setting{.key = field<std::string>(item, "key"),
                        .label = field<std::string>(item, "label"),
                        .section = field<std::string>(item, "section"),
                        .kind = settingKindFrom(field<std::string>(item, "kind")),
                        .choices = {},
                        .value = field<std::string>(item, "value"),
                        .minimum = field<float>(item, "minimum"),
                        .maximum = field<float>(item, "maximum"),
                        .step = field<float>(item, "step"),
                        .restartRequired = field<bool>(item, "restart_required"),
                        .note = field<std::string>(item, "note")};
    for (const auto& [value, label] :
         field<std::vector<std::pair<std::string, std::string>>>(item, "choices")) {
        setting.choices.push_back(SettingChoice{.value = value, .label = label});
    }
    return setting;
}

void setSettings(OverlayWindow& window, const py::dict& page) {
    SettingsPage settings{.items = {},
                          .restartPending = field<bool>(page, "restart_pending"),
                          .notice = field<std::string>(page, "notice")};
    for (const py::handle item : field<py::list>(page, "items")) {
        settings.items.push_back(settingFrom(item.cast<py::dict>()));
    }
    window.setSettings(std::move(settings));
}

HistoryRole historyRoleFrom(std::string_view name) {
    if (name == "user") {
        return HistoryRole::User;
    }
    if (name == "jarvis") {
        return HistoryRole::Jarvis;
    }
    if (name == "tool") {
        return HistoryRole::Tool;
    }
    if (name == "notice") {
        return HistoryRole::Notice;
    }
    throw py::value_error("unknown history role '" + std::string{name} +
                          "'; expected user, jarvis, tool or notice");
}

template <class Enum>
Enum parsedOrThrow(std::optional<Enum> parsed, std::string_view name, std::string_view what) {
    if (!parsed.has_value()) {
        throw py::value_error("unknown " + std::string{what} + " '" + std::string{name} + "'");
    }
    return *parsed;
}

void setAppearance(OverlayWindow& window, std::string_view theme, std::string_view intensity,
                   std::string_view size, std::string_view accent) {
    window.setAppearance(Appearance{
        .theme = parsedOrThrow(parseThemeName(theme), theme, "theme"),
        .accent = accentFrom(accent),
        .intensity = parsedOrThrow(anim::parseIntensity(intensity), intensity, "intensity"),
        .size = parsedOrThrow(parsePanelSize(size), size, "panel size")});
}

/// One UiEvent as a dict: {"kind": "submit_text", "text": ...}.
py::dict eventToDict(const UiEvent& event) {
    using py::literals::operator""_a;
    const std::string kind{kindOf(event)};
    if (const auto* submit = std::get_if<SubmitText>(&event)) {
        return py::dict("kind"_a = kind, "text"_a = submit->text);
    }
    if (const auto* expand = std::get_if<SetExpanded>(&event)) {
        return py::dict("kind"_a = kind, "expanded"_a = expand->expanded);
    }
    if (const auto* setting = std::get_if<SettingChanged>(&event)) {
        return py::dict("kind"_a = kind, "key"_a = setting->key, "value"_a = setting->value);
    }
    if (const auto* moved = std::get_if<PlacementChanged>(&event)) {
        return py::dict("kind"_a = kind, "expanded"_a = moved->expanded, "custom"_a = moved->custom,
                        "monitor"_a = moved->monitor, "x"_a = moved->x, "y"_a = moved->y,
                        "width"_a = moved->width, "height"_a = moved->height, "orb"_a = moved->orb);
    }
    if (const auto* minimized = std::get_if<Minimized>(&event)) {
        return py::dict("kind"_a = kind, "automatic"_a = minimized->automatic);
    }
    if (const auto* restored = std::get_if<Restored>(&event)) {
        return py::dict("kind"_a = kind, "automatic"_a = restored->automatic);
    }
    return py::dict("kind"_a = kind);
}

/// Inverse of eventToDict for the test hook.
UiEvent eventFromDict(const py::dict& event) {
    const auto kind = field<std::string>(event, "kind");
    if (kind == "submit_text") {
        return SubmitText{.text = field<std::string>(event, "text")};
    }
    if (kind == "set_expanded") {
        return SetExpanded{.expanded = field<bool>(event, "expanded")};
    }
    if (kind == "setting_changed") {
        return SettingChanged{.key = field<std::string>(event, "key"),
                              .value = field<std::string>(event, "value")};
    }
    if (kind == "minimize") {
        return Minimized{.automatic = field<bool>(event, "automatic")};
    }
    if (kind == "restore") {
        return Restored{.automatic = field<bool>(event, "automatic")};
    }
    const std::vector<std::pair<std::string_view, UiEvent>> simple{
        {"close", CloseRequested{}},     {"quit", QuitRequested{}},
        {"toggle_mute", ToggleMute{}},   {"open_log", OpenLog{}},
        {"restart", RestartRequested{}}, {"settings_opened", SettingsOpened{}}};
    for (const auto& [name, value] : simple) {
        if (name == kind) {
            return value;
        }
    }
    throw py::value_error("cannot inject UI event of kind '" + kind + "'");
}

py::list pollUiEvents(OverlayWindow& window) {
    py::list events;
    for (const UiEvent& event : window.pollUiEvents()) {
        events.append(eventToDict(event));
    }
    return events;
}

py::dict hudSnapshot(const OverlayWindow& window) {
    const HudData data = window.hudData();
    py::list history;
    for (const HistoryEntry& entry : *data.history) {
        history.append(py::make_tuple(static_cast<int>(entry.role), entry.text, entry.time));
    }
    py::list keys;
    for (const SettingItem& item : data.settings->items) {
        keys.append(item.key);
    }
    using py::literals::operator""_a;
    return py::dict("muted"_a = data.muted, "history"_a = history, "setting_keys"_a = keys,
                    "restart_pending"_a = data.settings->restartPending,
                    "theme"_a = std::string{toString(data.appearance.theme)},
                    "accent"_a = toHex(data.appearance.accent),
                    "intensity"_a = std::string{anim::toString(data.appearance.intensity)},
                    "size"_a = std::string{toString(data.appearance.size)},
                    "minimized"_a = window.minimized(),
                    "auto_minimize_seconds"_a = data.autoMinimizeSeconds,
                    "hide_in_fullscreen"_a = data.hideInFullscreen);
}

void setDemo(OverlayWindow& window, bool hover, std::optional<std::string> input) {
    window.setDemo(DemoOverrides{.hover = hover, .input = std::move(input)});
}

using WindowClass = py::class_<OverlayWindow>;

void defineWindowHud(WindowClass& window) {
    window
        .def("poll_ui_events", &pollUiEvents,
             "Everything the user did since the last call, oldest first, as dicts\n"
             "with a 'kind' key (submit_text, close, quit, toggle_mute, set_expanded,\n"
             "setting_changed, open_log, restart, settings_opened, placement_changed,\n"
             "minimize, restore).")
        .def("dropped_ui_events", &OverlayWindow::droppedUiEvents,
             "Events lost to the queue's drop-oldest overflow policy.")
        .def(
            "inject_ui_event",
            [](OverlayWindow& self, const py::dict& event) {
                self.injectUiEvent(eventFromDict(event));
            },
            py::arg("event"), "Test hook: enqueue an event as if the user had caused it.")
        .def("set_muted", &OverlayWindow::setMuted, py::arg("muted"))
        .def(
            "add_history",
            [](OverlayWindow& self, std::string_view role, std::string text, std::string time) {
                self.addHistory(HistoryEntry{.role = historyRoleFrom(role),
                                             .text = std::move(text),
                                             .time = std::move(time)});
            },
            py::arg("role"), py::arg("text"), py::arg("time"),
            "Append a conversation line; role is user, jarvis, tool or notice.")
        .def("clear_history", &OverlayWindow::clearHistory)
        .def("set_settings", &setSettings, py::arg("page"),
             "Replace the settings page: {'items': [...], 'restart_pending', 'notice'}.")
        .def("set_appearance", &setAppearance, py::arg("theme"), py::arg("intensity"),
             py::arg("size"), py::arg("accent") = "",
             "Switch theme, intensity and size live (settings spellings); accent is\n"
             "the CUSTOM theme's '#rrggbb', empty for the default.")
        .def("set_expanded", &OverlayWindow::setExpanded, py::arg("expanded"))
        .def("expanded", &OverlayWindow::expanded)
        .def("hud_snapshot", &hudSnapshot, "The HUD data pushed so far; for tests.")
        .def("set_demo", &setDemo, py::arg("hover") = false, py::arg("input") = py::none(),
             "Demo hook: show the hover controls and/or text in the field without\n"
             "moving the mouse or taking the keyboard focus.");
}

void defineWindowMinimize(WindowClass& window) {
    window
        .def("set_minimized", &OverlayWindow::setMinimized, py::arg("minimized"),
             "Shrink the HUD into the orb (True) or grow it back (False), animated.")
        .def("minimized", &OverlayWindow::minimized,
             "Whether the HUD is (or is animating towards) the orb.")
        .def("set_auto_minimize", &OverlayWindow::setAutoMinimize, py::arg("seconds"),
             "Shrink to the orb after this many quiet seconds; 0 switches it off.")
        .def("set_hide_in_fullscreen", &OverlayWindow::setHideInFullscreen, py::arg("hide"))
        .def("fullscreen_hidden", &OverlayWindow::fullscreenHidden,
             "True while hidden because a full-screen application is in front.");
}

void defineWindowAnimation(WindowClass& window) {
    window
        .def("wake_pulse", &OverlayWindow::flashWakeWord,
             "Pulse a ripple outwards from the ring to acknowledge the wake word.")
        .def("flash_wake_word", &OverlayWindow::flashWakeWord, "Alias of wake_pulse().")
        .def("play_boot", &playBoot, py::arg("info") = ReadoutPairs{},
             "Start the boot sequence with (label, value) readout lines; non-blocking,\n"
             "idempotent. True if this call started it.")
        .def("replay_boot", &replayBoot, py::arg("info") = ReadoutPairs{},
             "Play the whole boot sequence again with a fresh readout, even after the\n"
             "first one (a relaunch while running); skipped at once unless IDLE.")
        .def("skip_boot", &OverlayWindow::skipBoot,
             "Cut the boot sequence short; it still glides into the corner.")
        .def("transparent", &OverlayWindow::transparent,
             "True when the window has per-pixel transparency.")
        .def("fallback_reason", &OverlayWindow::fallbackReason,
             "Why the whole-window-alpha fallback is used; empty when transparent.")
        .def("bloom", &OverlayWindow::bloom, "True while the GPU bloom pipeline runs.")
        .def("bloom_reason", &OverlayWindow::bloomReason,
             "Why there is no bloom; empty when bloom() is True.")
        .def("frame_stats", &OverlayWindow::frameStats)
        .def("reset_frame_stats", &OverlayWindow::resetFrameStats);
}

void defineWindow(const py::module_& module) {
    WindowClass window(module, "OverlayWindow",
                       "Always-on-top status panel rendered on a native UI thread.");
    defineWindowAnimation(window);
    defineWindowHud(window);
    defineWindowMinimize(window);
    window
        .def(py::init<OverlayConfig>(), py::arg("config") = OverlayConfig{},
             py::call_guard<py::gil_scoped_release>(),
             "Create the window and wait until it is on screen. Raises OverlayError.")
        .def("set_state", py::overload_cast<AssistantState>(&OverlayWindow::setState),
             py::arg("state"))
        .def("set_state", &setStateByName, py::arg("name"),
             "Same as above, by member name (\"LISTENING\"); ValueError if unknown.")
        .def("push_level", &OverlayWindow::pushLevel, py::arg("level"),
             "Append one level sample in [0, 1] to the waveform.")
        .def("set_transcript", &OverlayWindow::setTranscript, py::arg("text"))
        .def("set_response", &OverlayWindow::setResponse, py::arg("text"))
        .def("set_visible", &OverlayWindow::setVisible, py::arg("visible"))
        .def("visible", &OverlayWindow::visible)
        .def("state", &OverlayWindow::state)
        .def("running", &OverlayWindow::running)
        .def("snapshot", &OverlayWindow::snapshot)
        .def("config", &OverlayWindow::config, py::return_value_policy::copy)
        .def("close", &OverlayWindow::close, py::call_guard<py::gil_scoped_release>(),
             "Close the window and join the UI thread. Idempotent.")
        .def("__enter__", [](OverlayWindow& self) -> OverlayWindow& { return self; })
        .def(
            "__exit__",
            [](OverlayWindow& self, const py::object& /*type*/, const py::object& /*value*/,
               const py::object& /*traceback*/) {
                const py::gil_scoped_release release;
                self.close();
            },
            py::arg("type"), py::arg("value"), py::arg("traceback"));
}

}  // namespace

PYBIND11_MODULE(jarvis_overlay, module) {
    module.doc() =
        "Status overlay: always-on-top Dear ImGui window with waveform, state "
        "indicator and transcript. See native/overlay/README.md.";
    jarvis::bindings::defineModuleInfo(module, "implemented in step 13");

    module.attr("BOOT_DURATION_SECONDS") = anim::BootTimeline{}.duration();
    module.def("default_microphone_name", &defaultMicrophoneName,
               "Friendly name of the default capture device, or None.");
    module.def("palette", &paletteOf, py::arg("theme"), py::arg("accent") = "",
               "The colours a theme resolves to, {role: (r, g, b, a)} plus 'bloom'; for\n"
               "the theme sheet and tests.");

    defineExceptions(module);
    defineEnums(module);
    definePlacement(module);
    defineConfig(module);
    defineSnapshot(module);
    defineFrameStats(module);
    defineWindow(module);
}
