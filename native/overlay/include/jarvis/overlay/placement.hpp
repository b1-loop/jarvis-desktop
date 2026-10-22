/// @file placement.hpp
/// @brief Where a movable HUD goes: snapping, clamping, monitors, drags.
///
/// The corner HUD and the expanded panel can be dragged anywhere on any
/// monitor (ADR-0024). Every decision about the resulting rectangle (snap
/// to an edge, stay fully on screen, survive a monitor being unplugged,
/// keep its size across DPI changes) is a pure function here, so the maths
/// is unit-tested without a window. All rectangles are the visible panel,
/// never the transparent bleed around it.

#pragma once

#include "jarvis/overlay/layout.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace jarvis::overlay {

/// Distance (96-DPI pixels) within which an edge snaps to the work area's.
inline constexpr int kSnapDistance = 16;

/// One monitor as the window sees it.
struct MonitorArea {
    /// Device name (MONITORINFOEX::szDevice), stable while it stays plugged in.
    std::string name;
    /// Work area in physical pixels (virtual-screen coordinates).
    PixelRect work;
    /// DPI scale of the monitor (1 at 96 DPI).
    float scale{1.0F};
    bool primary{false};
    /// The whole monitor (MONITORINFO::rcMonitor), taskbar included; what a
    /// full-screen window covers.
    PixelRect bounds;

    [[nodiscard]] bool operator==(const MonitorArea&) const = default;
};

/// A remembered position: logical pixels from the top-left of a monitor's
/// work area, plus a logical size (used by the expanded panel only).
struct SavedPlacement {
    std::string monitor;
    float x{0.0F};
    float y{0.0F};
    float width{0.0F};
    float height{0.0F};

    [[nodiscard]] bool operator==(const SavedPlacement&) const = default;
};

/// @p window with each edge that lies within @p threshold of the matching
/// edge of @p area moved onto it (left before right, top before bottom, so
/// a window as large as the area snaps to its top-left). Snapping both
/// axes at once is what makes corners snap. The size never changes.
[[nodiscard]] Placement snapToEdges(Placement window, PixelRect area, int threshold) noexcept;

/// @p window moved (never resized) the least distance that puts it fully
/// inside @p area; a window larger than the area is aligned to its
/// top-left so the controls in its header stay reachable.
[[nodiscard]] Placement clampInto(Placement window, PixelRect area) noexcept;

/// The monitor @p window overlaps most; with no overlap (a monitor was
/// unplugged) the one whose work area centre is nearest; with an empty
/// list 0. Ties go to the earlier monitor.
[[nodiscard]] std::size_t monitorFor(Placement window,
                                     std::span<const MonitorArea> monitors) noexcept;

/// Index of the monitor called @p name, else of the primary monitor, else 0.
[[nodiscard]] std::size_t monitorNamed(std::span<const MonitorArea> monitors,
                                       std::string_view name) noexcept;

/// Physical placement of a @p logicalWidth x @p logicalHeight panel saved
/// as @p saved: on its monitor if that is still connected (otherwise the
/// primary), scaled by that monitor's DPI, and clamped fully inside its
/// work area. @p monitors must not be empty.
[[nodiscard]] Placement restorePlacement(const SavedPlacement& saved,
                                         std::span<const MonitorArea> monitors, int logicalWidth,
                                         int logicalHeight);

/// What to remember for @p window on @p monitor: the offset from the work
/// area's top-left and the size, both divided by the monitor's scale.
[[nodiscard]] SavedPlacement savePlacement(Placement window, const MonitorArea& monitor);

/// Where in a window the user grabbed it, as a share of its size, so the
/// grab point stays under the cursor even when a DPI change resizes the
/// window mid-drag.
struct DragAnchor {
    float x{0.0F};
    float y{0.0F};

    [[nodiscard]] bool operator==(const DragAnchor&) const noexcept = default;
};

/// The anchor for a drag that starts at screen point (@p cursorX, @p cursorY).
[[nodiscard]] DragAnchor anchorFor(Placement window, int cursorX, int cursorY) noexcept;

/// A @p width x @p height window placed so @p anchor is under the cursor.
[[nodiscard]] Placement dragTo(DragAnchor anchor, int cursorX, int cursorY, int width,
                               int height) noexcept;

/// @p start grown by (@p dx, @p dy) from its bottom-right corner, never
/// smaller than @p minWidth x @p minHeight.
[[nodiscard]] Placement resizeFrom(Placement start, int dx, int dy, int minWidth,
                                   int minHeight) noexcept;

}  // namespace jarvis::overlay
