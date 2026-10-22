/// @file minimize_policy.hpp
/// @brief When the HUD gets out of the way by itself: auto-minimize after
/// a quiet spell, and hiding while another application is full-screen.
///
/// Both are decisions about time and rectangles, so they are pure and
/// unit-tested here with a fake clock and made-up windows; the window code
/// only feeds them (once per frame, and twice a second for the foreground
/// window) and applies what they return.

#pragma once

#include "jarvis/overlay/layout.hpp"

#include <cstdint>
#include <optional>

namespace jarvis::overlay {

// --- auto-minimize ------------------------------------------------------------------

/// What the HUD looks like to the auto-minimizer on one frame.
struct AutoMinimizeInput {
    /// Seconds on any monotonic clock.
    double now{0.0};
    /// Jarvis is doing something: not Idle, the wake word just fired, or
    /// the boot sequence is playing.
    bool active{false};
    /// The user is using the HUD: the cursor is over it, a drag, typing,
    /// the quit confirmation or the expanded panel.
    bool engaged{false};
    /// The HUD is (or is morphing into) the orb, whoever asked for it.
    bool minimized{false};
};

/// What the auto-minimizer wants done this frame.
enum class AutoAction : std::uint8_t { None, Minimize, Restore };

/// The `hud_auto_minimize_seconds` policy as a small state machine.
///
/// With a delay above 0, the full HUD shrinks to the orb after that many
/// seconds of Jarvis being idle with the user not touching it; when
/// Jarvis becomes active again (the wake word, a turn) an orb that this
/// policy created pops back to the full HUD, and the countdown starts
/// again once the turn is over. An orb the user asked for is the user's:
/// the policy never restores it. A delay of 0 switches the policy off
/// (and gives back an orb it created).
class AutoMinimizer {
public:
    explicit AutoMinimizer(float delaySeconds = 0.0F) noexcept;

    /// Change the delay; the current quiet spell keeps counting.
    void setDelay(float delaySeconds) noexcept;
    [[nodiscard]] float delay() const noexcept { return delay_; }

    /// Advance to @p input and return what to do now. Applying the action
    /// is the caller's job; the next input should reflect it.
    [[nodiscard]] AutoAction update(const AutoMinimizeInput& input) noexcept;

    /// The user minimized or restored by hand: whatever the state is now
    /// is theirs, and a restore restarts the countdown from that moment.
    void userChanged() noexcept;

    /// True while the orb on screen is one this policy created.
    [[nodiscard]] bool ownsOrb() const noexcept { return ownsOrb_; }

private:
    float delay_;
    std::optional<double> quietSince_;
    bool ownsOrb_{false};
};

// --- full-screen detection -------------------------------------------------------------

/// The foreground window as the full-screen check sees it.
struct ForegroundWindow {
    /// Its rectangle (GetWindowRect), physical pixels.
    PixelRect window;
    /// The full rectangle (not the work area) of the monitor it is on.
    PixelRect monitor;
    /// It has a caption (title bar): a maximized ordinary window, not a
    /// full-screen one, even where it covers the whole monitor.
    bool captioned{false};
    /// The desktop or shell itself (Progman, WorkerW): it always covers
    /// the monitor and never means "full-screen application".
    bool shell{false};
};

/// True when @p window covers all of @p monitor (and @p monitor has an area).
[[nodiscard]] bool coversMonitor(PixelRect window, PixelRect monitor) noexcept;

/// True when @p foreground is a full-screen application: it covers its
/// monitor, has no caption and is not the shell (a game, a video, a
/// presentation).
[[nodiscard]] bool isFullscreen(const ForegroundWindow& foreground) noexcept;

/// True when the HUD, on the monitor whose full rectangle is
/// @p hudMonitor, should hide because of @p foreground; a full-screen
/// window on another monitor leaves it alone. No foreground window
/// (a locked desktop) hides nothing.
[[nodiscard]] bool hidesHud(const std::optional<ForegroundWindow>& foreground,
                            PixelRect hudMonitor) noexcept;

}  // namespace jarvis::overlay
