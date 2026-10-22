// The interactive HUD on a real window (ADR-0024). Nothing here moves the
// mouse, presses keys or takes the keyboard focus: the user may be working
// at this desktop, so input is simulated through the test and demo hooks.

#include "jarvis/overlay/errors.hpp"
#include "jarvis/overlay/hud_data.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/placement.hpp"
#include "jarvis/overlay/ui_events.hpp"
#include "jarvis/overlay/window.hpp"

#include <catch2/catch_test_macros.hpp>

#include <windows.h>

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <variant>
#include <vector>

using jarvis::overlay::Appearance;
using jarvis::overlay::Corner;
using jarvis::overlay::DemoOverrides;
using jarvis::overlay::HistoryEntry;
using jarvis::overlay::HistoryRole;
using jarvis::overlay::OverlayConfig;
using jarvis::overlay::OverlayError;
using jarvis::overlay::OverlayWindow;
using jarvis::overlay::PanelSize;
using jarvis::overlay::SavedPlacement;
using jarvis::overlay::SettingItem;
using jarvis::overlay::SettingKind;
using jarvis::overlay::SettingsPage;
using jarvis::overlay::SubmitText;
using jarvis::overlay::ThemeName;
using jarvis::overlay::UiEvent;

namespace {

constexpr auto kSettle = std::chrono::milliseconds(150);

std::unique_ptr<OverlayWindow> openOrSkip(OverlayConfig config = {}) {
    config.corner = Corner::TopLeft;
    config.title = "Jarvis interactive test overlay";
    config.activeFps = 120;
    try {
        return std::make_unique<OverlayWindow>(config);
    } catch (const OverlayError& error) {
        SKIP("overlay window unavailable here: " << error.what());
    }
    return nullptr;
}

}  // namespace

TEST_CASE("an injected SubmitText comes back from pollUiEvents intact", "[overlay][interactive]") {
    const auto window = openOrSkip();
    window->injectUiEvent(SubmitText{.text = "vad \xC3\xA4r klockan?"});
    window->injectUiEvent(jarvis::overlay::ToggleMute{});

    const std::vector<UiEvent> events = window->pollUiEvents();
    REQUIRE(events.size() == 2);
    REQUIRE(std::get<SubmitText>(events.at(0)).text == "vad \xC3\xA4r klockan?");
    REQUIRE(std::holds_alternative<jarvis::overlay::ToggleMute>(events.at(1)));
    REQUIRE(window->pollUiEvents().empty());
    REQUIRE(window->droppedUiEvents() == 0);
}

TEST_CASE("showing the HUD never takes the keyboard focus", "[overlay][interactive]") {
    const auto window = openOrSkip();
    window->setDemo(DemoOverrides{.hover = true, .input = std::string{"typed by the demo"}});
    window->setExpanded(true);
    std::this_thread::sleep_for(kSettle * 4);
    // Compared against the HUD itself, not the previous foreground window:
    // the person at this desktop may switch windows while the test runs.
    HWND hud = FindWindowW(nullptr, L"Jarvis interactive test overlay");
    REQUIRE(hud != nullptr);
    REQUIRE(GetForegroundWindow() != hud);
    REQUIRE(window->running());
}

TEST_CASE("pushed HUD data reaches the model", "[overlay][interactive]") {
    const auto window = openOrSkip();
    window->setMuted(true);
    window->addHistory(HistoryEntry{.role = HistoryRole::User, .text = "hej", .time = "10:00"});
    window->setSettings(SettingsPage{
        .items = {SettingItem{
            .key = "overlay_theme",
            .label = "Theme",
            .section = "OVERLAY",
            .kind = SettingKind::Choice,
            .choices = {{.value = "cyan", .label = "Cyan"}, {.value = "amber", .label = "Amber"}},
            .value = "cyan"}},
        .restartPending = false,
        .notice = {}});

    const auto data = window->hudData();
    REQUIRE(data.muted);
    REQUIRE(data.history->front().text == "hej");
    REQUIRE(data.settings->items.front().choices.size() == 2);
}

TEST_CASE("the HUD keeps rendering through live appearance changes", "[overlay][interactive]") {
    const auto window = openOrSkip();
    window->setAppearance(Appearance{.theme = ThemeName::Amber,
                                     .intensity = jarvis::overlay::anim::Intensity::Max,
                                     .size = PanelSize::Compact});
    std::this_thread::sleep_for(kSettle);
    window->setAppearance(Appearance{.theme = ThemeName::Custom,
                                     .accent = {.r = 1.0F, .g = 1.0F, .b = 1.0F, .a = 1.0F}});
    std::this_thread::sleep_for(kSettle);
    window->resetFrameStats();
    window->setAppearance(Appearance{});
    std::this_thread::sleep_for(kSettle);
    REQUIRE(window->running());
    REQUIRE(window->frameStats().frames > 0);
    REQUIRE(window->hudData().appearance == Appearance{});
}

TEST_CASE("expanding and collapsing renders the morph and the panel", "[overlay][interactive]") {
    const auto window = openOrSkip();
    for (int i = 0; i < 5; ++i) {
        window->addHistory(
            HistoryEntry{.role = i % 2 == 0 ? HistoryRole::User : HistoryRole::Jarvis,
                         .text = "line " + std::to_string(i),
                         .time = "10:0" + std::to_string(i)});
    }
    window->setExpanded(true);
    REQUIRE(window->expanded());
    std::this_thread::sleep_for(kSettle * 4);  // the morph takes about 0.4 s
    window->resetFrameStats();
    std::this_thread::sleep_for(kSettle);
    REQUIRE(window->frameStats().frames > 0);
    window->setExpanded(false);
    std::this_thread::sleep_for(kSettle * 4);
    REQUIRE_FALSE(window->expanded());
    REQUIRE(window->running());
}

TEST_CASE("a remembered position on an unknown monitor still opens", "[overlay][interactive]") {
    OverlayConfig config;
    config.position = SavedPlacement{.monitor = "NO-SUCH-DISPLAY", .x = 50.0F, .y = 60.0F};
    config.expandedBounds = SavedPlacement{
        .monitor = "NO-SUCH-DISPLAY", .x = 0.0F, .y = 0.0F, .width = 100.0F, .height = 80.0F};
    const auto window = openOrSkip(config);
    window->setExpanded(true);
    std::this_thread::sleep_for(kSettle * 4);
    REQUIRE(window->running());
}

TEST_CASE("a display-only HUD still renders and reports no events", "[overlay][interactive]") {
    OverlayConfig config;
    config.interactive = false;
    const auto window = openOrSkip(config);
    std::this_thread::sleep_for(kSettle);
    REQUIRE(window->running());
    REQUIRE(window->pollUiEvents().empty());
}
