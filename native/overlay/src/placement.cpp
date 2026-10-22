#include "jarvis/overlay/placement.hpp"

#include "jarvis/overlay/layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <span>
#include <string_view>

namespace jarvis::overlay {

namespace {

/// One axis of snapToEdges: @p start/@p length against [@p low, @p high).
int snapAxis(int start, int length, int low, int high, int threshold) noexcept {
    if (std::abs(start - low) <= threshold) {
        return low;
    }
    const int end = start + length;
    if (std::abs(end - high) <= threshold) {
        return high - length;
    }
    return start;
}

/// One axis of clampInto.
int clampAxis(int start, int length, int low, int high) noexcept {
    if (length >= high - low) {
        return low;
    }
    return std::clamp(start, low, high - length);
}

long long overlapArea(Placement window, PixelRect area) noexcept {
    const int width = std::min(window.x + window.width, area.right) - std::max(window.x, area.left);
    const int height =
        std::min(window.y + window.height, area.bottom) - std::max(window.y, area.top);
    if (width <= 0 || height <= 0) {
        return 0;
    }
    return static_cast<long long>(width) * height;
}

double centreDistance(Placement window, PixelRect area) noexcept {
    const double dx = (window.x + (window.width / 2.0)) - ((area.left + area.right) / 2.0);
    const double dy = (window.y + (window.height / 2.0)) - ((area.top + area.bottom) / 2.0);
    return std::hypot(dx, dy);
}

std::size_t nearestMonitor(Placement window, std::span<const MonitorArea> monitors) noexcept {
    std::size_t best = 0;
    double bestDistance = std::numeric_limits<double>::max();
    for (std::size_t i = 0; const MonitorArea& monitor : monitors) {
        const double distance = centreDistance(window, monitor.work);
        if (distance < bestDistance) {
            best = i;
            bestDistance = distance;
        }
        ++i;
    }
    return best;
}

}  // namespace

Placement snapToEdges(Placement window, PixelRect area, int threshold) noexcept {
    window.x = snapAxis(window.x, window.width, area.left, area.right, threshold);
    window.y = snapAxis(window.y, window.height, area.top, area.bottom, threshold);
    return window;
}

Placement clampInto(Placement window, PixelRect area) noexcept {
    window.x = clampAxis(window.x, window.width, area.left, area.right);
    window.y = clampAxis(window.y, window.height, area.top, area.bottom);
    return window;
}

std::size_t monitorFor(Placement window, std::span<const MonitorArea> monitors) noexcept {
    std::size_t best = 0;
    long long bestArea = 0;
    for (std::size_t i = 0; const MonitorArea& monitor : monitors) {
        const long long area = overlapArea(window, monitor.work);
        if (area > bestArea) {
            best = i;
            bestArea = area;
        }
        ++i;
    }
    return bestArea > 0 ? best : nearestMonitor(window, monitors);
}

std::size_t monitorNamed(std::span<const MonitorArea> monitors, std::string_view name) noexcept {
    std::size_t primary = 0;
    for (std::size_t i = 0; const MonitorArea& monitor : monitors) {
        if (monitor.name == name) {
            return i;
        }
        if (monitor.primary) {
            primary = i;
        }
        ++i;
    }
    return primary;
}

Placement restorePlacement(const SavedPlacement& saved, std::span<const MonitorArea> monitors,
                           int logicalWidth, int logicalHeight) {
    const auto index = static_cast<std::ptrdiff_t>(monitorNamed(monitors, saved.monitor));
    const MonitorArea& monitor = *std::next(monitors.begin(), index);
    const Placement window{
        .x = monitor.work.left + static_cast<int>(std::lround(saved.x * monitor.scale)),
        .y = monitor.work.top + static_cast<int>(std::lround(saved.y * monitor.scale)),
        .width = scaled(logicalWidth, monitor.scale),
        .height = scaled(logicalHeight, monitor.scale)};
    return clampInto(window, monitor.work);
}

SavedPlacement savePlacement(Placement window, const MonitorArea& monitor) {
    const float scale = monitor.scale > 0.0F ? monitor.scale : 1.0F;
    return SavedPlacement{.monitor = monitor.name,
                          .x = static_cast<float>(window.x - monitor.work.left) / scale,
                          .y = static_cast<float>(window.y - monitor.work.top) / scale,
                          .width = static_cast<float>(window.width) / scale,
                          .height = static_cast<float>(window.height) / scale};
}

DragAnchor anchorFor(Placement window, int cursorX, int cursorY) noexcept {
    const auto share = [](int offset, int length) {
        return length > 0
                   ? std::clamp(static_cast<float>(offset) / static_cast<float>(length), 0.0F, 1.0F)
                   : 0.0F;
    };
    return DragAnchor{.x = share(cursorX - window.x, window.width),
                      .y = share(cursorY - window.y, window.height)};
}

Placement dragTo(DragAnchor anchor, int cursorX, int cursorY, int width, int height) noexcept {
    return Placement{
        .x = cursorX - static_cast<int>(std::lround(anchor.x * static_cast<float>(width))),
        .y = cursorY - static_cast<int>(std::lround(anchor.y * static_cast<float>(height))),
        .width = width,
        .height = height};
}

Placement resizeFrom(Placement start, int dx, int dy, int minWidth, int minHeight) noexcept {
    start.width = std::max(start.width + dx, minWidth);
    start.height = std::max(start.height + dy, minHeight);
    return start;
}

}  // namespace jarvis::overlay

// Design notes:
// - Positions are saved relative to a named monitor's work area, in
//   logical pixels: the same number means the same place after a DPI or
//   resolution change, and when that monitor is gone the panel lands at
//   the same offset on the primary monitor and is then clamped on screen.
// - Snapping runs before clamping in the window code, so an edge pulled a
//   few pixels past the screen snaps back onto it instead of being pushed.
// - The drag anchor is a share of the window size rather than a pixel
//   offset: when the cursor crosses onto a monitor with another DPI the
//   window is resized and the grab point must stay under the cursor.
