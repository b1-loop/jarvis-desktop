// Minimizing the real window to the orb (ADR-0024, update). Like the other
// window tests, nothing here moves the mouse, presses keys or takes focus:
// minimize and restore go through the same setters Python uses.

#include "jarvis/overlay/errors.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/ui_events.hpp"
#include "jarvis/overlay/window.hpp"

#include <catch2/catch_test_macros.hpp>

#include <windows.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <variant>
#include <vector>

using jarvis::overlay::AssistantState;
using jarvis::overlay::Corner;
using jarvis::overlay::Minimized;
using jarvis::overlay::OverlayConfig;
using jarvis::overlay::OverlayError;
using jarvis::overlay::OverlayWindow;
using jarvis::overlay::Restored;
using jarvis::overlay::UiEvent;

namespace {

constexpr auto kMorph = std::chrono::milliseconds(900);        // 0.5 s morph plus slack
constexpr auto kBootStarted = std::chrono::milliseconds(200);  // a few frames into act 1
constexpr const wchar_t* kTitle = L"Jarvis minimize test overlay";
/// Wider than any orb window (orb plus bleed at up to 200 % DPI), narrower
/// than any HUD.
constexpr int kOrbWindowLimit = 300;

std::unique_ptr<OverlayWindow> openOrSkip(OverlayConfig config = {}) {
    config.corner = Corner::TopLeft;
    config.title = "Jarvis minimize test overlay";
    config.activeFps = 120;
    try {
        return std::make_unique<OverlayWindow>(config);
    } catch (const OverlayError& error) {
        SKIP("overlay window unavailable here: " << error.what());
    }
    return nullptr;
}

RECT windowRect() {
    RECT rect{};
    HWND window = FindWindowW(nullptr, kTitle);
    if (window != nullptr) {
        GetWindowRect(window, &rect);
    }
    return rect;
}

long widthOf(const RECT& rect) {
    return rect.right - rect.left;
}

template <class Event>
bool contains(const std::vector<UiEvent>& events, bool automatic) {
    return std::ranges::any_of(events, [automatic](const UiEvent& event) {
        const auto* found = std::get_if<Event>(&event);
        return found != nullptr && found->automatic == automatic;
    });
}

}  // namespace

TEST_CASE("minimizing shrinks the window to the orb and restoring grows it back",
          "[overlay][orb][window]") {
    const auto window = openOrSkip();
    std::this_thread::sleep_for(kMorph);
    const long hudWidth = widthOf(windowRect());
    REQUIRE(hudWidth > kOrbWindowLimit);

    window->setMinimized(true);
    REQUIRE(window->minimized());
    std::this_thread::sleep_for(kMorph);
    REQUIRE(widthOf(windowRect()) < kOrbWindowLimit);
    REQUIRE(window->running());

    window->setMinimized(false);
    REQUIRE_FALSE(window->minimized());
    std::this_thread::sleep_for(kMorph);
    REQUIRE(widthOf(windowRect()) == hudWidth);
    // Requests from Python are not echoed back as events.
    REQUIRE(window->pollUiEvents().empty());
}

TEST_CASE("a HUD started minimized opens as the orb", "[overlay][orb][window]") {
    OverlayConfig config;
    config.startMinimized = true;
    const auto window = openOrSkip(config);
    REQUIRE(window->minimized());
    std::this_thread::sleep_for(kMorph);
    REQUIRE(widthOf(windowRect()) < kOrbWindowLimit);
}

TEST_CASE("auto-minimize shrinks when quiet and pops back for a turn", "[overlay][orb][window]") {
    OverlayConfig config;
    config.autoMinimizeSeconds = 0.3F;
    const auto window = openOrSkip(config);
    std::this_thread::sleep_for(kMorph);
    REQUIRE(window->minimized());
    REQUIRE(contains<Minimized>(window->pollUiEvents(), true));

    window->setState(AssistantState::Listening);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    REQUIRE_FALSE(window->minimized());
    REQUIRE(contains<Restored>(window->pollUiEvents(), true));

    window->setState(AssistantState::Idle);
    std::this_thread::sleep_for(kMorph);
    REQUIRE(window->minimized());
}

TEST_CASE("the idle orb drops to the idle frame rate", "[overlay][orb][window]") {
    OverlayConfig config;
    config.startMinimized = true;
    const auto window = openOrSkip(config);
    // Past the morph and the second of active rate after the last change.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    POINT cursor{};
    const RECT orb = windowRect();
    if (GetCursorPos(&cursor) != 0 && PtInRect(&orb, cursor) != 0) {
        SKIP("the cursor rests on the orb, which keeps it at the active rate");
    }
    window->resetFrameStats();
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    const auto stats = window->frameStats();
    REQUIRE(stats.frames > 0);
    // 15 fps is a 66.7 ms interval; allow for timer jitter.
    REQUIRE(stats.intervalMs > 55.0);
}

TEST_CASE("minimize options can change while the HUD runs", "[overlay][orb][window]") {
    const auto window = openOrSkip();
    window->setAutoMinimize(120.0F);
    window->setHideInFullscreen(false);
    const auto data = window->hudData();
    REQUIRE(data.autoMinimizeSeconds == 120.0F);
    REQUIRE_FALSE(data.hideInFullscreen);
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    REQUIRE_FALSE(window->fullscreenHidden());
    REQUIRE_FALSE(window->minimized());
}

TEST_CASE("a boot over the orb plays in the full HUD, then shrinks back",
          "[overlay][orb][window]") {
    OverlayConfig config;
    config.startMinimized = true;
    const auto window = openOrSkip(config);
    std::this_thread::sleep_for(kMorph);
    REQUIRE(widthOf(windowRect()) < kOrbWindowLimit);

    // A relaunch while running: the boot spans the work area again.
    window->replayBoot({{.label = "MODE", .value = "CLOUD"}});
    std::this_thread::sleep_for(kBootStarted);
    REQUIRE(widthOf(windowRect()) > kOrbWindowLimit);
    REQUIRE(window->minimized());  // the user's orb is kept, not forgotten

    REQUIRE(window->skipBoot());
    std::this_thread::sleep_for(kMorph + kMorph);  // the hand-off, then the orb morph
    REQUIRE(widthOf(windowRect()) < kOrbWindowLimit);
    REQUIRE(window->pollUiEvents().empty());  // nothing the user did
}
