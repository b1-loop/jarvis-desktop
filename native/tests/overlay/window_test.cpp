#include "jarvis/overlay/window.hpp"

#include "jarvis/overlay/errors.hpp"
#include "jarvis/overlay/model.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using jarvis::overlay::AssistantState;
using jarvis::overlay::Corner;
using jarvis::overlay::OverlayConfig;
using jarvis::overlay::OverlayError;
using jarvis::overlay::OverlayWindow;

namespace {

/// A single Python call must never be noticeable; 1 ms per call is already
/// a hundred times slower than reality and still far below that bar.
constexpr int kSetterCalls = 2000;
constexpr auto kSetterBudget = std::chrono::milliseconds(kSetterCalls);

OverlayConfig testConfig() {
    OverlayConfig config;
    config.corner = Corner::TopLeft;
    config.title = "Jarvis test overlay";
    config.activeFps = 120;
    return config;
}

/// Build the real window, or SKIP where no desktop or Direct3D is available
/// (session 0 services, some headless runners).
std::unique_ptr<OverlayWindow> openWindowOrSkip() {
    try {
        return std::make_unique<OverlayWindow>(testConfig());
    } catch (const OverlayError& error) {
        SKIP("overlay window unavailable here: " << error.what());
    }
    return nullptr;
}

}  // namespace

TEST_CASE("the window comes up running and visible, and closes cleanly", "[overlay][window]") {
    const auto window = openWindowOrSkip();

    REQUIRE(window->running());
    REQUIRE(window->visible());
    REQUIRE(window->config().corner == Corner::TopLeft);

    window->close();
    REQUIRE_FALSE(window->running());
    window->close();  // idempotent
}

TEST_CASE("destruction joins the UI thread without an explicit close", "[overlay][window]") {
    auto window = openWindowOrSkip();
    window->setState(AssistantState::Speaking);
    window.reset();
    SUCCEED("no hang, no crash");
}

TEST_CASE("setters update the snapshot the UI thread will draw", "[overlay][window]") {
    const auto window = openWindowOrSkip();

    window->setState(AssistantState::Listening);
    window->pushLevel(0.7F);
    window->setTranscript("open the terminal");
    window->setResponse("Opening the terminal.");
    window->flashWakeWord();
    window->setVisible(false);

    const auto snapshot = window->snapshot();
    REQUIRE(snapshot.state == AssistantState::Listening);
    REQUIRE(window->state() == AssistantState::Listening);
    REQUIRE(snapshot.levels.back() == 0.7F);
    REQUIRE(snapshot.transcript == "open the terminal");
    REQUIRE(snapshot.response == "Opening the terminal.");
    REQUIRE(snapshot.wakeFlash > 0.0F);
    REQUIRE_FALSE(snapshot.visible);
    REQUIRE_FALSE(window->visible());
}

TEST_CASE("setters never block on the renderer", "[overlay][window]") {
    const auto window = openWindowOrSkip();
    window->setState(AssistantState::Listening);  // keeps the renderer at the active rate

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kSetterCalls; ++i) {
        window->pushLevel(static_cast<float>(i % 100) / 100.0F);
        window->setTranscript("call number " + std::to_string(i));
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;

    REQUIRE(elapsed < kSetterBudget);
    REQUIRE(window->running());
}

TEST_CASE("setters are safe from several threads at once", "[overlay][window]") {
    const auto window = openWindowOrSkip();
    constexpr int kThreads = 4;
    std::vector<std::jthread> callers;
    callers.reserve(kThreads);
    for (int thread = 0; thread < kThreads; ++thread) {
        callers.emplace_back([&window, thread] {
            for (int i = 0; i < 500; ++i) {
                window->pushLevel(0.5F);
                window->setResponse("thread " + std::to_string(thread));
                window->setState(i % 2 == 0 ? AssistantState::Thinking : AssistantState::Speaking);
            }
        });
    }
    callers.clear();  // joins

    REQUIRE(window->running());
    REQUIRE(window->snapshot().response.starts_with("thread "));
}

TEST_CASE("a hidden window keeps accepting updates and can be shown again", "[overlay][window]") {
    const auto window = openWindowOrSkip();
    window->setVisible(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));  // let the UI thread hide it

    window->setTranscript("still listening");
    window->setVisible(true);

    REQUIRE(window->visible());
    REQUIRE(window->snapshot().transcript == "still listening");
}
