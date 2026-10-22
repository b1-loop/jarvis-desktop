#include "jarvis/overlay/hud_data.hpp"
#include "jarvis/overlay/hud_layout.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/minimize_policy.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/orb.hpp"
#include "jarvis/overlay/placement.hpp"
#include "jarvis/overlay/ui_events.hpp"
#include "jarvis/overlay/window.hpp"

#include "window_impl.hpp"
#include <windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <optional>
#include <string_view>

namespace jarvis::overlay {

namespace {

/// How often the foreground window is checked for full screen: often
/// enough that the HUD is gone before a video's first seconds are over,
/// rare enough to cost nothing.
constexpr std::chrono::milliseconds kForegroundPoll{500};
/// Class names of the desktop itself, which always covers its monitor.
constexpr std::array<std::wstring_view, 2> kShellClasses{L"Progman", L"WorkerW"};
constexpr int kClassNameChars = 64;

bool isShellWindow(HWND window) {
    std::array<wchar_t, kClassNameChars> name{};
    const int length = GetClassNameW(window, name.data(), static_cast<int>(name.size()));
    const std::wstring_view className{name.data(), static_cast<std::size_t>(std::max(length, 0))};
    return std::ranges::find(kShellClasses, className) != kShellClasses.end();
}

PixelRect toPixelRect(const RECT& rect) noexcept {
    return PixelRect{
        .left = rect.left, .top = rect.top, .right = rect.right, .bottom = rect.bottom};
}

/// The foreground window as minimize_policy.hpp needs it; none when there
/// is none, when it is this overlay (the user is typing into it) or when
/// its rectangle cannot be read.
std::optional<ForegroundWindow> foregroundWindow(HWND self) {
    HWND window = GetForegroundWindow();
    RECT rect{};
    if (window == nullptr || window == self || GetWindowRect(window, &rect) == 0) {
        return std::nullopt;
    }
    MONITORINFO monitor{};
    monitor.cbSize = sizeof(monitor);
    if (GetMonitorInfoW(MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST), &monitor) == 0) {
        return std::nullopt;
    }
    // NOLINTBEGIN(bugprone-signed-bitwise,hicpp-signed-bitwise) - Win32 style flags are signed
    const LONG_PTR style = GetWindowLongPtrW(window, GWL_STYLE);
    const bool captioned = (style & WS_CAPTION) == WS_CAPTION;
    // NOLINTEND(bugprone-signed-bitwise,hicpp-signed-bitwise)
    return ForegroundWindow{.window = toPixelRect(rect),
                            .monitor = toPixelRect(monitor.rcMonitor),
                            .captioned = captioned,
                            .shell = isShellWindow(window)};
}

}  // namespace

// --- minimize and restore ----------------------------------------------------------

void OverlayWindow::Impl::applyMinimizeRequest(const HudData& data) {
    if (data.minimizeRequestId == seenMinimizeRequest) {
        return;
    }
    seenMinimizeRequest = data.minimizeRequestId;
    if (data.minimizeRequest.has_value()) {
        setOrbTarget(*data.minimizeRequest, false, false);
    }
}

void OverlayWindow::Impl::setOrbTarget(bool minimized, bool automatic, bool report) {
    if (!automatic) {
        autoMinimizer.userChanged();
    }
    if (minimized == orbTarget) {
        return;
    }
    orbTarget = minimized;
    minimizedNow.store(minimized);
    if (minimized) {
        // The orb is reached through the corner HUD: collapse first.
        interaction.setExpanded(false, true);
        interaction.closeInput();
        releaseFocus();
        orbPanel = orbHome();
    }
    if (report) {
        events.push(minimized ? UiEvent{Minimized{.automatic = automatic}}
                              : UiEvent{Restored{.automatic = automatic}});
    }
}

void OverlayWindow::Impl::runAutoMinimize(const OverlaySnapshot& snapshot, const HudData& data,
                                          bool booting, Clock::time_point now) {
    autoMinimizer.setDelay(data.autoMinimizeSeconds);
    const bool engaged = demoHover || hoverKind != HitKind::None || drag.mode != DragMode::None ||
                         keyboardFocus || interaction.inputOpen() || interaction.confirming() ||
                         interaction.expanded();
    const bool active =
        booting || snapshot.state != AssistantState::Idle || snapshot.wakeFlash > 0.0F;
    const AutoAction action = autoMinimizer.update(
        AutoMinimizeInput{.now = std::chrono::duration<double>(now.time_since_epoch()).count(),
                          .active = active,
                          .engaged = engaged,
                          .minimized = orbTarget});
    if (action != AutoAction::None) {
        setOrbTarget(action == AutoAction::Minimize, true, true);
    }
}

void OverlayWindow::Impl::advanceOrb(float dt, bool booting) {
    if (orbTarget && interaction.expanded()) {
        // Python (or the keyboard) expanded the HUD: that wins over the orb.
        setOrbTarget(false, false, true);
    }
    // The orb waits for the boot's hand-off into the corner and for the
    // expanded panel to collapse, then the corner HUD shrinks into it.
    orbGoal = orbTarget && !booting && expandLinear <= 0.0F;
    orbLinear = approach(orbLinear, orbGoal ? 1.0F : 0.0F, dt, kOrbMorphSeconds);
}

bool OverlayWindow::Impl::orbMorphing() const noexcept {
    return orbLinear != (orbGoal ? 1.0F : 0.0F);
}

Placement OverlayWindow::Impl::orbHome() const {
    if (orbSaved.has_value()) {
        return restorePlacement(*orbSaved, monitors, kOrbDiameter, kOrbDiameter);
    }
    const HudGeometry ring =
        panelGeometry(toRectF(cornerPanel), kPanelPadding * metrics.textScale * dpiScale);
    return dockOrb(cornerPanel, ring.centreX, ring.centreY, monitorOf(cornerPanel));
}

void OverlayWindow::Impl::syncWindowRegion(const FrameLayout& layout) {
    if (transparent.load()) {
        return;  // per-pixel alpha already makes everything around the orb invisible
    }
    auto* const window = hwnd.load();
    if (layout.orb <= 0.0F) {
        if (regionSize.has_value()) {
            SetWindowRgn(window, nullptr, TRUE);
            regionSize.reset();
        }
        return;
    }
    const SIZE size{.cx = layout.window.width, .cy = layout.window.height};
    if (regionSize.has_value() && regionSize->cx == size.cx && regionSize->cy == size.cy) {
        return;
    }
    // On success the system owns the region; on failure it is ours to free.
    HRGN region = CreateEllipticRgn(0, 0, size.cx + 1, size.cy + 1);
    if (region != nullptr && SetWindowRgn(window, region, TRUE) == 0) {
        DeleteObject(region);
    }
    regionSize = size;
}

// --- full screen -----------------------------------------------------------------------

void OverlayWindow::Impl::checkFullscreen(Clock::time_point now, bool enabled) {
    if (now < nextForegroundCheck) {
        return;
    }
    nextForegroundCheck = now + kForegroundPoll;
    if (!enabled || monitors.empty()) {
        fullscreenHidden = false;
    } else {
        const HudPanel panel = orbTarget ? HudPanel::Orb : activePanel();
        const PixelRect hudMonitor = monitorOf(panelRect(panel)).bounds;
        fullscreenHidden = hidesHud(foregroundWindow(hwnd.load()), hudMonitor);
    }
    fullscreenNow.store(fullscreenHidden);
}

}  // namespace jarvis::overlay

// Design notes:
// - The minimized state lives here, on the UI thread, not in
//   HudInteraction: it changes the window's size, its hit map, what a drag
//   moves and what a click means, which are all the window's business.
//   The interaction layer only reports a click on its minimize control.
// - Requests from Python (tray, hotkey) are applied without queuing an
//   event back, like set_expanded; the control, a click on the orb and
//   auto-minimize queue Minimized/Restored so Python can remember the
//   user's choice and keep the tray's label right. Automatic changes are
//   flagged so they are never remembered as the user's.
// - Full screen is polled twice a second rather than hooked
//   (SetWinEventHook): an application can go full screen without the
//   foreground changing (F11 in a browser), and the poll is a handful of
//   cheap calls. It runs while hidden too, which is how the HUD returns.
// - The opaque fallback cuts its window to a circle with SetWindowRgn
//   while the orb shows or forms, so it never shows a dark square either.
