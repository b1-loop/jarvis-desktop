#include "jarvis/overlay/window.hpp"

#include "jarvis/overlay/errors.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/system_info.hpp"

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

TEST_CASE("the window reports which surface it chose", "[overlay][window]") {
    const auto window = openWindowOrSkip();
    // Either per-pixel transparency works, or the reason it does not is known.
    REQUIRE(window->transparent() == window->fallbackReason().empty());
}

TEST_CASE("the layered fallback can be forced and still renders", "[overlay][window]") {
    OverlayConfig config = testConfig();
    config.transparent = false;
    std::unique_ptr<OverlayWindow> window;
    try {
        window = std::make_unique<OverlayWindow>(config);
    } catch (const OverlayError& error) {
        SKIP("overlay window unavailable here: " << error.what());
    }
    REQUIRE_FALSE(window->transparent());
    REQUIRE_FALSE(window->fallbackReason().empty());
    window->playBoot();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    REQUIRE(window->running());
    REQUIRE(window->frameStats().frames > 0);
}

TEST_CASE("the boot plays once and frames are measured while it runs", "[overlay][window]") {
    const auto window = openWindowOrSkip();
    window->resetFrameStats();
    REQUIRE(window->playBoot());
    REQUIRE_FALSE(window->playBoot());  // idempotent
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    const auto stats = window->frameStats();
    REQUIRE(stats.frames > 3);
    REQUIRE(stats.renderMs > 0.0);
    REQUIRE(stats.intervalMs > 0.0);
    REQUIRE(window->snapshot().bootStartedAt.has_value());
    REQUIRE(window->skipBoot());
    REQUIRE_FALSE(window->skipBoot());
}

TEST_CASE("the window reports whether bloom runs, and why not", "[overlay][window][bloom]") {
    const auto window = openWindowOrSkip();
    REQUIRE(window->bloom() == window->bloomReason().empty());

    OverlayConfig config = testConfig();
    config.bloom = false;
    std::unique_ptr<OverlayWindow> plain;
    window->close();  // Dear ImGui supports one context, so one window at a time
    try {
        plain = std::make_unique<OverlayWindow>(config);
    } catch (const OverlayError& error) {
        SKIP("overlay window unavailable here: " << error.what());
    }
    REQUIRE_FALSE(plain->bloom());
    REQUIRE_FALSE(plain->bloomReason().empty());
    plain->playBoot();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    REQUIRE(plain->running());
    REQUIRE(plain->frameStats().frames > 0);
}

TEST_CASE("the GPU time of frames is measured with timestamp queries", "[overlay][window][bloom]") {
    const auto window = openWindowOrSkip();
    window->resetFrameStats();
    REQUIRE(window->playBoot({{.label = "MODEL", .value = "test"}}));
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    const auto stats = window->frameStats();
    REQUIRE(stats.gpuFrames > 0);
    REQUIRE(stats.gpuMs > 0.0);
    REQUIRE(window->snapshot().bootReadout->front().label == "MODEL");
}

TEST_CASE("every intensity renders", "[overlay][window]") {
    for (const auto intensity :
         {jarvis::overlay::anim::Intensity::Calm, jarvis::overlay::anim::Intensity::Max}) {
        OverlayConfig config = testConfig();
        config.intensity = intensity;
        std::unique_ptr<OverlayWindow> window;
        try {
            window = std::make_unique<OverlayWindow>(config);
        } catch (const OverlayError& error) {
            SKIP("overlay window unavailable here: " << error.what());
        }
        window->setState(AssistantState::Speaking);
        window->pushLevel(0.8F);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        REQUIRE(window->running());
        REQUIRE(window->frameStats().frames > 0);
    }
}

TEST_CASE("the default microphone has a name or none at all", "[overlay][system]") {
    const auto name = jarvis::overlay::defaultMicrophoneName();
    if (!name.has_value()) {
        SKIP("no default capture device in this session");
    }
    REQUIRE_FALSE(name.value_or(std::string{}).empty());
}
