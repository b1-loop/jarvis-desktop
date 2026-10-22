#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/hud_layout.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/orb.hpp"
#include "jarvis/overlay/placement.hpp"
#include "jarvis/overlay/ui_events.hpp"
#include "jarvis/overlay/window.hpp"

#include "window_impl.hpp"
#include <shellscalingapi.h>
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace jarvis::overlay {

namespace {

/// Smallest placement containing both @p a and @p b.
Placement unionOf(const Placement& a, const Placement& b) noexcept {
    const int left = std::min(a.x, b.x);
    const int top = std::min(a.y, b.y);
    const int right = std::max(a.x + a.width, b.x + b.width);
    const int bottom = std::max(a.y + a.height, b.y + b.height);
    return Placement{.x = left, .y = top, .width = right - left, .height = bottom - top};
}

float monitorScale(HMONITOR monitor) noexcept {
    UINT dpiX = 0;
    UINT dpiY = 0;
    if (FAILED(GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &dpiX, &dpiY)) || dpiX == 0) {
        return 1.0F;
    }
    return static_cast<float>(dpiX) / kBaseDpi;
}

MonitorArea describeMonitor(HMONITOR monitor) {
    MONITORINFOEXW info{};
    info.cbSize = sizeof(info);
    GetMonitorInfoW(monitor, &info);
    return MonitorArea{
        .name = toUtf8(std::wstring_view{static_cast<const wchar_t*>(info.szDevice)}),
        .work = PixelRect{.left = info.rcWork.left,
                          .top = info.rcWork.top,
                          .right = info.rcWork.right,
                          .bottom = info.rcWork.bottom},
        .scale = monitorScale(monitor),
        .primary = (info.dwFlags & MONITORINFOF_PRIMARY) != 0,
        .bounds = PixelRect{.left = info.rcMonitor.left,
                            .top = info.rcMonitor.top,
                            .right = info.rcMonitor.right,
                            .bottom = info.rcMonitor.bottom}};
}

BOOL CALLBACK collectMonitor(HMONITOR monitor, HDC /*dc*/, LPRECT /*rect*/, LPARAM data) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    auto* const monitors = reinterpret_cast<std::vector<MonitorArea>*>(data);
    monitors->push_back(describeMonitor(monitor));
    return TRUE;
}

/// One axis of the default expanded placement: aligned with the corner
/// HUD's edge on the side of the monitor it sits on.
int alignedStart(int start, int length, int panelLength, int low, int high) noexcept {
    const int centre = start + (length / 2);
    return centre > (low + high) / 2 ? start + length - panelLength : start;
}

}  // namespace

void OverlayWindow::Impl::refreshMonitors() {
    std::vector<MonitorArea> found;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) - Win32 callback context
    EnumDisplayMonitors(nullptr, nullptr, &collectMonitor, reinterpret_cast<LPARAM>(&found));
    if (found.empty()) {
        found.push_back(
            describeMonitor(MonitorFromPoint(POINT{.x = 0, .y = 0}, MONITOR_DEFAULTTOPRIMARY)));
        found.back().primary = true;
    }
    monitors = std::move(found);
    workArea = monitors.at(monitorNamed(monitors, "")).work;
    placementDirty = true;
}

void OverlayWindow::Impl::resolvePlacements() {
    cornerPanel = cornerSaved.has_value()
                      ? restorePlacement(*cornerSaved, monitors, metrics.width, metrics.height)
                      : defaultCorner();
    if (expandedSaved.has_value()) {
        const int width =
            std::max(static_cast<int>(std::lround(expandedSaved->width)), kExpandedMinWidth);
        const int height =
            std::max(static_cast<int>(std::lround(expandedSaved->height)), kExpandedMinHeight);
        expandedPanel = restorePlacement(*expandedSaved, monitors, width, height);
    } else {
        expandedPanel = defaultExpanded();
    }
    orbPanel = orbHome();
    placementDirty = false;
}

Placement OverlayWindow::Impl::defaultCorner() const {
    return anchorToCorner(config.corner, workArea, scaled(metrics.width, dpiScale),
                          scaled(metrics.height, dpiScale), scaled(config.margin, dpiScale));
}

Placement OverlayWindow::Impl::defaultExpanded() const {
    // Grows out of the corner HUD: shares its outer corner, then is pushed
    // fully onto the monitor the HUD is on.
    const MonitorArea& monitor = monitorOf(cornerPanel);
    const int width = scaled(kExpandedWidth, monitor.scale);
    const int height = scaled(kExpandedHeight, monitor.scale);
    const PixelRect& work = monitor.work;
    const Placement panel{
        .x = alignedStart(cornerPanel.x, cornerPanel.width, width, work.left, work.right),
        .y = alignedStart(cornerPanel.y, cornerPanel.height, height, work.top, work.bottom),
        .width = width,
        .height = height};
    return clampInto(panel, work);
}

const MonitorArea& OverlayWindow::Impl::monitorOf(const Placement& rect) const {
    return monitors.at(monitorFor(rect, monitors));
}

float OverlayWindow::Impl::expandEased() const noexcept {
    return anim::easeInOutCubic(expandLinear);
}

FrameLayout OverlayWindow::Impl::layoutFor(const anim::AnimationFrame& frame) const {
    const RectF panel = toRectF(cornerPanel);
    const HudGeometry panelHud = panelGeometry(panel, kPanelPadding * metrics.textScale * dpiScale);
    FrameLayout layout{.window = cornerPanel,
                       .geometry = panelHud,
                       .panel = panel,
                       .panelText = panelHud.text,
                       .canvas = panel,
                       .frame = panelHud.text};
    if (frame.boot.has_value()) {
        placeBoot(layout, *frame.boot, panelHud);
    } else if (orbLinear > 0.0F) {
        placeOrb(layout, panelHud);
    } else if (expandLinear > 0.0F) {
        placeExpanded(layout, panelHud);
    } else if (transparent.load()) {
        layout.window = inflated(cornerPanel, scaled(kBleed, dpiScale));
    }
    const auto dx = static_cast<float>(-layout.window.x);
    const auto dy = static_cast<float>(-layout.window.y);
    layout.geometry = translated(layout.geometry, dx, dy);
    layout.panel = translated(layout.panel, dx, dy);
    layout.panelText = translated(layout.panelText, dx, dy);
    layout.canvas = translated(layout.canvas, dx, dy);
    layout.frame = translated(layout.frame, dx, dy);
    layout.collapsed = collapsedLayout(layout.panelText, dpiScale, metrics.textScale);
    layout.expanded = expandedLayout(translated(toRectF(expandedPanel), dx, dy), dpiScale);
    return layout;
}

void OverlayWindow::Impl::placeBoot(FrameLayout& layout, const anim::BootFrame& boot,
                                    const HudGeometry& panelHud) const {
    const int side = scaled(kBootCanvas, dpiScale);
    const RectF canvas = toRectF(centredIn(workArea, side, side));
    const HudGeometry bootHud = bootGeometry(canvas);
    layout.canvas = lerpRect(canvas, layout.panel, boot.handoff);
    layout.geometry = lerpGeometry(bootHud, panelHud, boot.handoff);
    const float dx = panelHud.centreX - bootHud.centreX;
    const float dy = panelHud.centreY - bootHud.centreY;
    const float distance = std::hypot(dx, dy);
    if (distance > 0.0F) {
        layout.handoffX = dx / distance;
        layout.handoffY = dy / distance;
    }
    // Transparent: one work-area-sized window, so the move is pure drawing.
    // Fallback: the opaque window itself follows the canvas.
    layout.window = transparent.load() ? Placement{.x = workArea.left,
                                                   .y = workArea.top,
                                                   .width = workArea.right - workArea.left,
                                                   .height = workArea.bottom - workArea.top}
                                       : enclosingPlacement(layout.canvas);
}

void OverlayWindow::Impl::placeExpanded(FrameLayout& layout, const HudGeometry& panelHud) const {
    const float eased = expandEased();
    const RectF target = toRectF(expandedPanel);
    HudGeometry ring = expandedLayout(target, dpiScale).ring;
    ring.text = panelHud.text;
    layout.expand = expandLinear >= 1.0F ? 1.0F : eased;
    layout.geometry = lerpGeometry(panelHud, ring, eased);
    // The frame unfolds out of the corner HUD's text column.
    layout.frame = lerpRect(panelHud.text, target, eased);
    if (transparent.load()) {
        const int bleed = scaled(kBleed, dpiScale);
        const Placement expanded = inflated(expandedPanel, bleed);
        // Both rectangles for the whole morph: drawing, not resizing.
        layout.window =
            expandLinear >= 1.0F ? expanded : unionOf(inflated(cornerPanel, bleed), expanded);
    } else {
        layout.window = expandLinear >= 1.0F
                            ? expandedPanel
                            : enclosingPlacement(lerpRect(layout.panel, target, eased));
    }
}

void OverlayWindow::Impl::placeOrb(FrameLayout& layout, const HudGeometry& panelHud) const {
    const OrbMorph morph = orbMorphAt(orbLinear);
    const RectF orb = toRectF(orbPanel);
    const HudGeometry orbRing{.centreX = orb.x + (orb.width * 0.5F),
                              .centreY = orb.y + (orb.height * 0.5F),
                              .radius = orb.width * 0.5F,
                              .text = panelHud.text};
    layout.orb = orbLinear;
    layout.geometry = lerpGeometry(panelHud, orbRing, morph.travel);
    if (transparent.load()) {
        const Placement orbWindow = inflated(orbPanel, scaled(kOrbBleed, dpiScale));
        // Both rectangles for the whole morph, like the expand morph.
        layout.window = orbLinear >= 1.0F
                            ? orbWindow
                            : unionOf(inflated(cornerPanel, scaled(kBleed, dpiScale)), orbWindow);
        return;
    }
    // The opaque fallback has no invisible pixels: the window is the
    // shrinking ring's square, cut to a circle (syncWindowRegion).
    const int side =
        std::max(static_cast<int>(std::lround(layout.geometry.radius * 2.0F)), orbPanel.width);
    layout.window = squareAt(layout.geometry.centreX, layout.geometry.centreY, side);
}

void OverlayWindow::Impl::applyPlacement(const Placement& target) {
    if (target == placement) {
        return;
    }
    placement = target;
    // WM_SIZE arrives synchronously and resizes the swap chain before the
    // frame is drawn, so the new size is used immediately.
    SetWindowPos(hwnd.load(), HWND_TOPMOST, target.x, target.y, target.width, target.height,
                 SWP_NOACTIVATE);
}

void OverlayWindow::Impl::resetPlacement(bool expanded) {
    if (expanded) {
        expandedSaved.reset();
    } else {
        cornerSaved.reset();
    }
    placementDirty = true;
    events.push(PlacementChanged{.expanded = expanded, .custom = false});
}

void OverlayWindow::Impl::reportPlacement(HudPanel panel) {
    const Placement& rect = panelRect(panel);
    const SavedPlacement saved = savePlacement(rect, monitorOf(rect));
    switch (panel) {
        case HudPanel::Expanded:
            expandedSaved = saved;
            break;
        case HudPanel::Orb:
            orbSaved = saved;
            break;
        case HudPanel::Corner:
            cornerSaved = saved;
            forgetOrbPlacement();  // the orb docks next to the HUD's new place
            break;
    }
    events.push(PlacementChanged{.expanded = panel == HudPanel::Expanded,
                                 .custom = true,
                                 .monitor = saved.monitor,
                                 .x = saved.x,
                                 .y = saved.y,
                                 .width = saved.width,
                                 .height = saved.height,
                                 .orb = panel == HudPanel::Orb});
}

void OverlayWindow::Impl::forgetOrbPlacement() {
    if (!orbSaved.has_value()) {
        return;
    }
    orbSaved.reset();
    events.push(PlacementChanged{.expanded = false, .custom = false, .orb = true});
}

Placement& OverlayWindow::Impl::panelRect(HudPanel panel) noexcept {
    switch (panel) {
        case HudPanel::Expanded:
            return expandedPanel;
        case HudPanel::Orb:
            return orbPanel;
        case HudPanel::Corner:
            break;
    }
    return cornerPanel;
}

HudPanel OverlayWindow::Impl::activePanel() const noexcept {
    if (orbLinear >= 1.0F) {
        return HudPanel::Orb;
    }
    return expandLinear >= 1.0F ? HudPanel::Expanded : HudPanel::Corner;
}

}  // namespace jarvis::overlay

// Design notes:
// - Two remembered rectangles, the corner HUD's and the expanded panel's,
//   each either "default" (anchored to the configured corner, or grown out
//   of the HUD) or saved as monitor + logical offset (placement.hpp). They
//   are re-resolved after a display change, so an unplugged monitor sends
//   the HUD to the primary one, fully on screen.
// - The morph keeps the transparent window at the union of both panels
//   for its duration (one resize at each end); the opaque fallback has no
//   invisible pixels to spare and follows the interpolated rectangle.
// - The orb is a third remembered rectangle. Unless the user dragged the
//   orb itself, it docks next to wherever the HUD is (orb.hpp); dragging
//   the HUD forgets a dragged orb, so the two never end up on opposite
//   sides of the desktop by accident.
// - The default expanded panel is aligned with the corner the HUD sits in,
//   so expanding from the bottom-right corner grows up and to the left and
//   the ring's flight into the header stays short.
