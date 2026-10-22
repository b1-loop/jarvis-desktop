#include "jarvis/overlay/minimize_policy.hpp"

#include "jarvis/overlay/layout.hpp"

#include <algorithm>
#include <optional>

namespace jarvis::overlay {

// --- auto-minimize ------------------------------------------------------------------

AutoMinimizer::AutoMinimizer(float delaySeconds) noexcept : delay_(std::max(delaySeconds, 0.0F)) {}

void AutoMinimizer::setDelay(float delaySeconds) noexcept {
    delay_ = std::max(delaySeconds, 0.0F);
}

AutoAction AutoMinimizer::update(const AutoMinimizeInput& input) noexcept {
    if (ownsOrb_ && !input.minimized) {
        ownsOrb_ = false;  // restored by someone else since the last frame
    }
    if (ownsOrb_ && (input.active || delay_ <= 0.0F)) {
        ownsOrb_ = false;
        quietSince_ = input.now;
        return AutoAction::Restore;
    }
    if (input.minimized || input.active || input.engaged || !quietSince_.has_value()) {
        quietSince_ = input.now;
        return AutoAction::None;
    }
    if (delay_ > 0.0F && input.now - *quietSince_ >= static_cast<double>(delay_)) {
        ownsOrb_ = true;
        return AutoAction::Minimize;
    }
    return AutoAction::None;
}

void AutoMinimizer::userChanged() noexcept {
    ownsOrb_ = false;
    quietSince_.reset();
}

// --- full-screen detection -------------------------------------------------------------

bool coversMonitor(PixelRect window, PixelRect monitor) noexcept {
    const bool hasArea = monitor.right > monitor.left && monitor.bottom > monitor.top;
    return hasArea && window.left <= monitor.left && window.top <= monitor.top &&
           window.right >= monitor.right && window.bottom >= monitor.bottom;
}

bool isFullscreen(const ForegroundWindow& foreground) noexcept {
    return !foreground.shell && !foreground.captioned &&
           coversMonitor(foreground.window, foreground.monitor);
}

bool hidesHud(const std::optional<ForegroundWindow>& foreground, PixelRect hudMonitor) noexcept {
    return foreground.has_value() && foreground->monitor == hudMonitor && isFullscreen(*foreground);
}

}  // namespace jarvis::overlay

// Design notes:
// - The quiet spell is measured from the last frame on which Jarvis was
//   active, the user engaged or the HUD minimized; any of those restarts
//   it. So the orb comes back "after the turn plus the delay", and a user
//   reading the HUD with the cursor on it never has it shrink under them.
// - The policy remembers whether the orb is its own. Popping the HUD back
//   on the wake word only makes sense for an orb the user did not ask for;
//   a user who minimized by hand still sees the state on the orb itself.
// - The first frame only starts the clock (quietSince_ unset), so a HUD
//   that was just shown, restored or reconfigured always gets the full
//   delay before it shrinks.
// - Full screen means "covers the monitor and has no caption": a maximized
//   window with an auto-hiding taskbar also covers the monitor (by its
//   invisible resize borders) but keeps its title bar; games, videos and
//   slide shows drop it. The desktop window covers the monitor too and is
//   excluded by class name in the window code.
