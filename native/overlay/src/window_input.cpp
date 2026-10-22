#include "jarvis/overlay/hud_layout.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/placement.hpp"
#include "jarvis/overlay/window.hpp"

#include <imgui.h>
#include <imgui_impl_win32.h>

#include "hud_interact.hpp"
#include "window_impl.hpp"
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>

// The backend keeps this declaration behind '#if 0' in its header so the
// header stays free of <windows.h>; applications declare it themselves.
// NOLINTNEXTLINE(readability-identifier-naming) - symbol name is Dear ImGui's
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam,
                                                             LPARAM lParam);

namespace jarvis::overlay {

namespace {

constexpr std::uint64_t kWordMask = 0xFFFF;
constexpr unsigned kWordBits = 16;
constexpr int kMaxOpacity = 255;
/// The column's scrim reaches this far beyond the text, 96-DPI pixels.
constexpr float kScrimFeather = 28.0F;

UINT lowWord(LPARAM value) noexcept {
    return static_cast<UINT>(static_cast<std::uint64_t>(value) & kWordMask);
}

UINT highWord(LPARAM value) noexcept {
    return static_cast<UINT>((static_cast<std::uint64_t>(value) >> kWordBits) & kWordMask);
}

/// The signed client point packed in a mouse message's lParam.
POINT pointOf(LPARAM value) noexcept {
    return POINT{.x = static_cast<short>(lowWord(value)), .y = static_cast<short>(highWord(value))};
}

bool contains(const RectF& rect, POINT point) noexcept {
    const auto x = static_cast<float>(point.x);
    const auto y = static_cast<float>(point.y);
    return x >= rect.x && x < rect.right() && y >= rect.y && y < rect.bottom();
}

bool altHeld() noexcept {
    // Asynchronous state: the HUD has the mouse capture, not the keyboard.
    // The most significant bit (a negative SHORT) means "down now".
    return GetAsyncKeyState(VK_MENU) < 0;
}

}  // namespace

// --- click-through and hit-testing --------------------------------------------

void OverlayWindow::Impl::setClickThrough(bool on) const {
    auto* const window = hwnd.load();
    const LONG_PTR current = GetWindowLongPtrW(window, GWL_EXSTYLE);
    // NOLINTBEGIN(bugprone-signed-bitwise,hicpp-signed-bitwise) - Win32 style flags are signed
    constexpr LONG_PTR kPassThrough = WS_EX_TRANSPARENT | WS_EX_LAYERED;
    // The fallback is always layered (that is its transparency); the
    // composition window only needs WS_EX_LAYERED for click-through, and an
    // interactive one keeps it so hover can toggle WS_EX_TRANSPARENT alone.
    const bool layeredSurface = (current & WS_EX_NOREDIRECTIONBITMAP) == 0;
    LONG_PTR style = on ? (current | kPassThrough) : (current & ~WS_EX_TRANSPARENT);
    if (!on && !layeredSurface && !config.interactive) {
        style &= ~WS_EX_LAYERED;
    }
    // NOLINTEND(bugprone-signed-bitwise,hicpp-signed-bitwise)
    SetWindowLongPtrW(window, GWL_EXSTYLE, style);
    if (on && !layeredSurface) {
        // A layered window stays invisible until its attributes are set;
        // full alpha leaves the composition visual's per-pixel alpha alone.
        SetLayeredWindowAttributes(window, 0, kMaxOpacity, LWA_ALPHA);
    }
}

void OverlayWindow::Impl::syncBootClickThrough(bool booting) {
    // The boot canvas sits in the middle of the screen; it must never eat a
    // click there, even when the panel itself is configured to take them.
    // An interactive HUD has an empty hit map while booting instead.
    if (config.interactive || config.clickThrough || booting == forcedClickThrough) {
        return;
    }
    forcedClickThrough = booting;
    setClickThrough(booting);
}

void OverlayWindow::Impl::updateHitMap(const FrameLayout& layout, bool booting) {
    hitMap.clear();
    inputRect.reset();
    hitRing = layout.geometry;
    if (!config.interactive || booting) {
        return;
    }
    if (layout.orb > 0.0F) {
        // Only the finished orb is clickable; mid-morph nothing is.
        if (layout.orb >= 1.0F) {
            const HudGeometry& orb = layout.geometry;
            orbHitMap(hitMap, orb.centreX, orb.centreY, orb.radius);
        }
        return;
    }
    if (layout.expand <= 0.0F) {
        const CollapsedHits open{.inputOpen = interaction.inputOpen(),
                                 .confirming = interaction.confirming()};
        collapsedHitMap(hitMap, layout.geometry, layout.panelText, layout.collapsed, open,
                        kScrimFeather * dpiScale);
        if (open.inputOpen) {
            inputRect = layout.collapsed.input;
        }
    } else if (layout.expand >= 1.0F) {
        expandedHitMap(hitMap, layout.expanded);
        inputRect = layout.expanded.input;
    }
}

HitKind OverlayWindow::Impl::hitAtClient(POINT client) const {
    if (drag.mode != DragMode::None) {
        return drag.mode == DragMode::Resize ? HitKind::Resize : HitKind::Drag;
    }
    return hitMap.test(static_cast<float>(client.x), static_cast<float>(client.y));
}

HitKind OverlayWindow::Impl::hitAt(LPARAM screenPoint) const {
    POINT point = pointOf(screenPoint);
    ScreenToClient(hwnd.load(), &point);
    return hitAtClient(point);
}

void OverlayWindow::Impl::pollCursor() {
    if (!config.interactive) {
        return;
    }
    POINT cursor{};
    if (GetCursorPos(&cursor) == 0) {
        hoverKind = HitKind::None;  // e.g. a locked desktop: nothing to hover
    } else {
        ScreenToClient(hwnd.load(), &cursor);
        hoverKind = hitAtClient(cursor);
    }
    syncMouseThrough();
}

void OverlayWindow::Impl::syncMouseThrough() {
    const bool wanted = hoverKind != HitKind::None || drag.mode != DragMode::None;
    if (wanted == mouseOpen) {
        return;
    }
    mouseOpen = wanted;
    auto* const window = hwnd.load();
    const LONG_PTR current = GetWindowLongPtrW(window, GWL_EXSTYLE);
    // NOLINTBEGIN(bugprone-signed-bitwise,hicpp-signed-bitwise) - Win32 style flags are signed
    const LONG_PTR style = wanted ? (current & ~WS_EX_TRANSPARENT) : (current | WS_EX_TRANSPARENT);
    // NOLINTEND(bugprone-signed-bitwise,hicpp-signed-bitwise)
    SetWindowLongPtrW(window, GWL_EXSTYLE, style);
}

// --- keyboard focus ---------------------------------------------------------------

void OverlayWindow::Impl::takeFocus() {
    auto* const window = hwnd.load();
    if (keyboardFocus || window == nullptr) {
        return;
    }
    HWND foreground = GetForegroundWindow();
    previousForeground = foreground != window ? foreground : nullptr;
    // WS_EX_NOACTIVATE only stops clicks from activating the window; an
    // explicit request is allowed because this process got the last input.
    SetForegroundWindow(window);
    SetFocus(window);
    keyboardFocus = GetForegroundWindow() == window;
}

void OverlayWindow::Impl::releaseFocus() {
    if (!keyboardFocus) {
        return;
    }
    keyboardFocus = false;
    if (previousForeground != nullptr && IsWindow(previousForeground) != 0) {
        SetForegroundWindow(previousForeground);
    }
    previousForeground = nullptr;
}

// --- dragging -------------------------------------------------------------------------

bool OverlayWindow::Impl::beginDrag(POINT client, bool doubleClick) {
    const HitKind kind = hitAtClient(client);
    if (kind != HitKind::Drag && kind != HitKind::Resize) {
        return false;
    }
    const HudPanel panel = activePanel();
    const float dx = static_cast<float>(client.x) - hitRing.centreX;
    const float dy = static_cast<float>(client.y) - hitRing.centreY;
    // A double-click on the ring resets the HUD's place; on the orb it is
    // simply another click (the first one already restores).
    if (doubleClick && kind == HitKind::Drag && panel != HudPanel::Orb &&
        std::hypot(dx, dy) <= hitRing.radius) {
        resetPlacement(panel == HudPanel::Expanded);
        return true;
    }
    POINT screen = client;
    ClientToScreen(hwnd.load(), &screen);
    drag = DragState{.mode = kind == HitKind::Resize ? DragMode::Resize : DragMode::Move,
                     .panel = panel,
                     .anchor = {},
                     .start = screen,
                     .startRect = panelRect(panel),
                     .moved = panel != HudPanel::Orb};
    drag.anchor = anchorFor(drag.startRect, screen.x, screen.y);
    SetCapture(hwnd.load());
    return true;
}

bool OverlayWindow::Impl::leftDragRectangle(POINT cursor) const noexcept {
    return std::abs(cursor.x - drag.start.x) > GetSystemMetrics(SM_CXDRAG) ||
           std::abs(cursor.y - drag.start.y) > GetSystemMetrics(SM_CYDRAG);
}

void OverlayWindow::Impl::continueDrag() {
    POINT cursor{};
    if (GetCursorPos(&cursor) == 0) {
        return;
    }
    if (!drag.moved) {
        if (!leftDragRectangle(cursor)) {
            return;  // still a click on the orb
        }
        drag.moved = true;
    }
    Placement& rect = panelRect(drag.panel);
    if (drag.mode == DragMode::Move) {
        const Placement moved = dragTo(drag.anchor, cursor.x, cursor.y, rect.width, rect.height);
        const MonitorArea& monitor = monitorOf(moved);
        // Alt held disables snapping, for pixel-exact placement.
        const Placement snapped =
            altHeld() ? moved
                      : snapToEdges(moved, monitor.work, scaled(kSnapDistance, monitor.scale));
        rect = clampInto(snapped, monitor.work);
        return;
    }
    const MonitorArea& monitor = monitorOf(drag.startRect);
    Placement grown =
        resizeFrom(drag.startRect, cursor.x - drag.start.x, cursor.y - drag.start.y,
                   scaled(kExpandedMinWidth, dpiScale), scaled(kExpandedMinHeight, dpiScale));
    grown.width = std::min(grown.width, monitor.work.right - grown.x);
    grown.height = std::min(grown.height, monitor.work.bottom - grown.y);
    rect = grown;
}

void OverlayWindow::Impl::endDrag(bool released) {
    if (drag.mode == DragMode::None) {
        return;
    }
    const DragState ended = drag;
    // Cleared first: ReleaseCapture sends WM_CAPTURECHANGED synchronously.
    drag.mode = DragMode::None;
    if (GetCapture() == hwnd.load()) {
        ReleaseCapture();
    }
    if (ended.moved) {
        reportPlacement(ended.panel);
    } else if (released) {
        setOrbTarget(false, false, true);  // a click on the orb restores the HUD
    }
}

bool OverlayWindow::Impl::onSetCursor() const {
    POINT cursor{};
    GetCursorPos(&cursor);
    ScreenToClient(hwnd.load(), &cursor);
    const HitKind kind = hitAtClient(cursor);
    LPCSTR shape = IDC_ARROW;
    if (kind == HitKind::Drag) {
        // The orb's main action is a click (restore); the HUD's is a drag.
        shape = activePanel() == HudPanel::Orb ? IDC_HAND : IDC_SIZEALL;
    } else if (kind == HitKind::Resize) {
        shape = IDC_SIZENWSE;
    } else if (kind == HitKind::Control && inputRect.has_value() && contains(*inputRect, cursor)) {
        shape = IDC_IBEAM;
    }
    SetCursor(LoadCursorA(nullptr, shape));
    return true;
}

// --- window procedure ---------------------------------------------------------------

std::optional<LRESULT> OverlayWindow::Impl::handleInput(UINT message, WPARAM wParam,
                                                        LPARAM lParam) {
    switch (message) {
        case WM_NCHITTEST:
            if (hitAt(lParam) == HitKind::None) {
                // Close the window to the mouse at once; the next click
                // falls through instead of waiting for the next frame.
                hoverKind = HitKind::None;
                syncMouseThrough();
                return HTTRANSPARENT;
            }
            return HTCLIENT;
        case WM_SETCURSOR:
            if (lowWord(lParam) == HTCLIENT && onSetCursor()) {
                return TRUE;
            }
            return std::nullopt;
        case WM_ACTIVATE:
            if (lowWord(static_cast<LPARAM>(wParam)) == WA_INACTIVE) {
                // The user clicked elsewhere: they took the focus back.
                keyboardFocus = false;
                previousForeground = nullptr;
            }
            return std::nullopt;
        default:
            return handleMouse(message, lParam);
    }
}

std::optional<LRESULT> OverlayWindow::Impl::handleMouse(UINT message, LPARAM lParam) {
    switch (message) {
        case WM_LBUTTONDOWN:
        case WM_LBUTTONDBLCLK: {
            const POINT point = pointOf(lParam);
            if (beginDrag(point, message == WM_LBUTTONDBLCLK)) {
                return 0;
            }
            if (inputRect.has_value() && contains(*inputRect, point)) {
                takeFocus();
            }
            return std::nullopt;  // a control: ImGui handles the click
        }
        case WM_MOUSEMOVE:
            if (drag.mode != DragMode::None) {
                continueDrag();
                return 0;
            }
            return std::nullopt;
        case WM_LBUTTONUP:
            if (drag.mode != DragMode::None) {
                endDrag(true);
                return 0;
            }
            return std::nullopt;
        case WM_CAPTURECHANGED:
            endDrag(false);
            return std::nullopt;
        default:
            return std::nullopt;
    }
}

void OverlayWindow::Impl::onDpiChanged(WPARAM wParam) {
    const float next = static_cast<float>(highWord(static_cast<LPARAM>(wParam))) / kBaseDpi;
    if (drag.mode != DragMode::None && dpiScale > 0.0F) {
        // Mid-drag onto another monitor: keep the logical size.
        Placement& rect = panelRect(drag.panel);
        const float ratio = next / dpiScale;
        rect.width = static_cast<int>(std::lround(static_cast<float>(rect.width) * ratio));
        rect.height = static_cast<int>(std::lround(static_cast<float>(rect.height) * ratio));
    } else {
        placementDirty = true;
    }
    dpiScale = next;
    if (ImGui::GetCurrentContext() != nullptr && hud != nullptr) {
        hud::applyHudStyle(hud->theme(), dpiScale);
    }
}

LRESULT CALLBACK OverlayWindow::Impl::wndProc(HWND window, UINT message, WPARAM wParam,
                                              LPARAM lParam) {
    if (message == WM_NCCREATE) {
        // Win32 passes the CREATESTRUCT and stores user data as integers.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
        const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lParam);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) - GWLP_USERDATA is a LONG_PTR
        const auto userData = reinterpret_cast<LONG_PTR>(create->lpCreateParams);
        SetWindowLongPtrW(window, GWLP_USERDATA, userData);
    }
    const LONG_PTR userData = GetWindowLongPtrW(window, GWLP_USERDATA);
    // Inverse of the store above.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    auto* self = reinterpret_cast<Impl*>(userData);
    if (self == nullptr) {
        return DefWindowProcW(window, message, wParam, lParam);
    }
    return self->handleMessage(window, message, wParam, lParam);
}

LRESULT OverlayWindow::Impl::handleMessage(HWND window, UINT message, WPARAM wParam,
                                           LPARAM lParam) {
    if (config.interactive && window == hwnd.load()) {
        if (const std::optional<LRESULT> handled = handleInput(message, wParam, lParam)) {
            return *handled;
        }
    }
    if (ImGui::GetCurrentContext() != nullptr &&
        ImGui_ImplWin32_WndProcHandler(window, message, wParam, lParam) != 0) {
        return 1;
    }
    switch (message) {
        case WM_SIZE:
            if (wParam != SIZE_MINIMIZED && surface != nullptr) {
                surface->resize(lowWord(lParam), highWord(lParam));
            }
            return 0;
        case WM_DPICHANGED:
            onDpiChanged(wParam);
            return 0;  // the next frame's layout uses the new scale
        case WM_DISPLAYCHANGE:
        case WM_SETTINGCHANGE:
            refreshMonitors();  // a monitor, the work area or the resolution changed
            return 0;
        case WM_MOUSEACTIVATE:
            return MA_NOACTIVATE;
        case WM_DESTROY:
            // Only the live window ends the loop; the composition attempt
            // that failed and was destroyed during creation must not.
            if (window == hwnd.load()) {
                PostQuitMessage(0);
            }
            return 0;
        case WM_NCDESTROY:
            if (window == hwnd.load()) {
                hwnd.store(nullptr);
            }
            return 0;
        default:
            return DefWindowProcW(window, message, wParam, lParam);
    }
}

}  // namespace jarvis::overlay

// Design notes:
// - HTTRANSPARENT alone cannot make transparent pixels click-through:
//   Windows only passes such a click to windows of the same thread. The
//   window is therefore WS_EX_TRANSPARENT (truly click-through) except
//   while the cursor is over one of the HUD's shapes, polled once per frame
//   and re-checked on every WM_NCHITTEST; HTTRANSPARENT covers the moment
//   in between.
// - Dragging is done here with SetCapture, not by answering HTCAPTION:
//   DefWindowProc's modal move loop would stop this thread's frame loop
//   for the whole drag (a frozen HUD), and snapping, Alt, the DPI-safe
//   anchor and clamping are simpler on raw cursor positions anyway.
// - The window never activates on a click (WM_MOUSEACTIVATE returns
//   MA_NOACTIVATE, WS_EX_NOACTIVATE); it takes the keyboard explicitly when
//   the text field is clicked or the keyboard control used, remembers the
//   window it took it from and gives it back on submit or Escape.
// - Snapping uses the monitor under most of the panel, with its own DPI,
//   then clamps onto that monitor; crossing to the next monitor therefore
//   happens once most of the panel is over it.
