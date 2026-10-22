#include "jarvis/overlay/minimize_policy.hpp"

#include "jarvis/overlay/layout.hpp"

#include <catch2/catch_test_macros.hpp>

#include <optional>

using jarvis::overlay::AutoAction;
using jarvis::overlay::AutoMinimizeInput;
using jarvis::overlay::AutoMinimizer;
using jarvis::overlay::coversMonitor;
using jarvis::overlay::ForegroundWindow;
using jarvis::overlay::hidesHud;
using jarvis::overlay::isFullscreen;
using jarvis::overlay::PixelRect;

namespace {

/// A fake clock and the HUD's state, driven one frame at a time.
class Harness {
public:
    explicit Harness(float delay) : policy_(delay) {}

    /// Advance the clock by @p seconds in frames of 1/15 s (the idle rate),
    /// applying every action; returns the last non-None action.
    AutoAction run(double seconds) {
        AutoAction last = AutoAction::None;
        const double end = now_ + seconds;
        while (now_ < end) {
            now_ += kFrame;
            const AutoAction action = step();
            if (action != AutoAction::None) {
                last = action;
            }
        }
        return last;
    }

    AutoAction step() {
        const AutoAction action = policy_.update(AutoMinimizeInput{
            .now = now_, .active = active_, .engaged = engaged_, .minimized = minimized_});
        if (action == AutoAction::Minimize) {
            minimized_ = true;
        } else if (action == AutoAction::Restore) {
            minimized_ = false;
        }
        return action;
    }

    AutoMinimizer& policy() { return policy_; }
    void setActive(bool active) { active_ = active; }
    void setEngaged(bool engaged) { engaged_ = engaged; }
    void setMinimized(bool minimized) { minimized_ = minimized; }
    [[nodiscard]] bool minimized() const { return minimized_; }

private:
    bool active_{false};
    bool engaged_{false};
    bool minimized_{false};
    static constexpr double kFrame = 1.0 / 15.0;
    AutoMinimizer policy_;
    double now_{0.0};
};

constexpr PixelRect kMonitor{.left = 0, .top = 0, .right = 1920, .bottom = 1080};
constexpr PixelRect kOther{.left = 1920, .top = 0, .right = 4480, .bottom = 1440};

ForegroundWindow window(PixelRect rect, PixelRect monitor = kMonitor) {
    return ForegroundWindow{.window = rect, .monitor = monitor, .captioned = false, .shell = false};
}

}  // namespace

TEST_CASE("a delay of zero never minimizes", "[overlay][autominimize]") {
    Harness hud(0.0F);
    REQUIRE(hud.run(600.0) == AutoAction::None);
    REQUIRE_FALSE(hud.minimized());
}

TEST_CASE("an idle, untouched HUD shrinks after the delay", "[overlay][autominimize]") {
    Harness hud(10.0F);
    REQUIRE(hud.run(9.8) == AutoAction::None);
    REQUIRE(hud.run(0.4) == AutoAction::Minimize);
    REQUIRE(hud.minimized());
    REQUIRE(hud.policy().ownsOrb());
    REQUIRE(hud.run(60.0) == AutoAction::None);  // it stays an orb
}

TEST_CASE("hovering or using the HUD restarts the countdown", "[overlay][autominimize]") {
    Harness hud(5.0F);
    hud.run(4.0);
    hud.setEngaged(true);
    REQUIRE(hud.run(30.0) == AutoAction::None);
    hud.setEngaged(false);
    REQUIRE(hud.run(4.5) == AutoAction::None);
    REQUIRE(hud.run(1.0) == AutoAction::Minimize);
}

TEST_CASE("a turn pops the orb back and it returns after the turn plus the delay",
          "[overlay][autominimize]") {
    Harness hud(5.0F);
    hud.run(6.0);
    REQUIRE(hud.minimized());
    // The wake word fires: LISTENING, then THINKING and SPEAKING.
    hud.setActive(true);
    REQUIRE(hud.step() == AutoAction::Restore);
    REQUIRE_FALSE(hud.minimized());
    REQUIRE(hud.run(20.0) == AutoAction::None);  // a long answer never shrinks
    hud.setActive(false);
    REQUIRE(hud.run(4.8) == AutoAction::None);
    REQUIRE(hud.run(0.4) == AutoAction::Minimize);
}

TEST_CASE("an orb the user asked for is never popped back", "[overlay][autominimize]") {
    Harness hud(5.0F);
    hud.setMinimized(true);
    hud.policy().userChanged();
    hud.setActive(true);
    REQUIRE(hud.run(10.0) == AutoAction::None);
    REQUIRE(hud.minimized());
}

TEST_CASE("a manual restore gets the full delay again", "[overlay][autominimize]") {
    Harness hud(5.0F);
    hud.run(6.0);
    REQUIRE(hud.minimized());
    hud.setMinimized(false);
    hud.policy().userChanged();
    REQUIRE_FALSE(hud.policy().ownsOrb());
    REQUIRE(hud.run(4.8) == AutoAction::None);
    REQUIRE(hud.run(0.4) == AutoAction::Minimize);
}

TEST_CASE("switching the policy off gives back the orb it created", "[overlay][autominimize]") {
    Harness hud(5.0F);
    hud.run(6.0);
    REQUIRE(hud.minimized());
    hud.policy().setDelay(0.0F);
    REQUIRE(hud.step() == AutoAction::Restore);
    REQUIRE(hud.run(60.0) == AutoAction::None);
}

TEST_CASE("an orb restored elsewhere is no longer the policy's", "[overlay][autominimize]") {
    Harness hud(5.0F);
    hud.run(6.0);
    hud.setMinimized(false);  // e.g. restored from the tray
    REQUIRE(hud.step() == AutoAction::None);
    REQUIRE_FALSE(hud.policy().ownsOrb());
}

TEST_CASE("a negative delay is treated as off", "[overlay][autominimize]") {
    const AutoMinimizer policy(-3.0F);
    REQUIRE(policy.delay() == 0.0F);
}

TEST_CASE("a window covers a monitor only when it reaches every edge", "[overlay][fullscreen]") {
    REQUIRE(coversMonitor(kMonitor, kMonitor));
    REQUIRE(
        coversMonitor(PixelRect{.left = -8, .top = -8, .right = 1928, .bottom = 1088}, kMonitor));
    REQUIRE_FALSE(coversMonitor(PixelRect{.left = 0, .top = 0, .right = 1920, .bottom = 1040},
                                kMonitor));  // stops at the taskbar
    REQUIRE_FALSE(coversMonitor(kMonitor, PixelRect{}));
}

TEST_CASE("full screen means covering, captionless and not the shell", "[overlay][fullscreen]") {
    REQUIRE(isFullscreen(window(kMonitor)));
    ForegroundWindow maximized =
        window(PixelRect{.left = -8, .top = -8, .right = 1928, .bottom = 1088});
    maximized.captioned = true;
    REQUIRE_FALSE(isFullscreen(maximized));
    ForegroundWindow desktop = window(kMonitor);
    desktop.shell = true;
    REQUIRE_FALSE(isFullscreen(desktop));
    REQUIRE_FALSE(
        isFullscreen(window(PixelRect{.left = 100, .top = 100, .right = 900, .bottom = 700})));
}

TEST_CASE("only a full-screen window on the HUD's monitor hides it", "[overlay][fullscreen]") {
    REQUIRE(hidesHud(window(kMonitor), kMonitor));
    REQUIRE_FALSE(hidesHud(window(kOther, kOther), kMonitor));
    REQUIRE_FALSE(hidesHud(std::nullopt, kMonitor));
}
