/// @file screenshot.hpp
/// @brief Screen capture encoded as PNG.

#pragma once

#include "jarvis/os/window_manager.hpp"

#include <cstdint>
#include <vector>

namespace jarvis::os {

/// Interface for capturing the screen.
///
/// Results are PNG files in memory rather than raw pixels because every
/// consumer (vision model, dashboard, "save a screenshot") wants a file,
/// and encoding natively avoids pushing tens of megabytes of raw BGRA
/// across the Python boundary.
class Screenshotter {
public:
    Screenshotter() = default;
    Screenshotter(const Screenshotter&) = delete;
    Screenshotter& operator=(const Screenshotter&) = delete;
    Screenshotter(Screenshotter&&) = delete;
    Screenshotter& operator=(Screenshotter&&) = delete;
    virtual ~Screenshotter() = default;

    /// Capture the primary monitor, or the whole virtual desktop with
    /// @p allMonitors. Throws OsError if there is no interactive desktop.
    [[nodiscard]] virtual std::vector<std::uint8_t> captureScreen(bool allMonitors) const = 0;

    /// Capture the on-screen rectangle of @p handle (what the user sees,
    /// including anything overlapping it). Throws WindowNotFoundError.
    [[nodiscard]] virtual std::vector<std::uint8_t> captureWindow(WindowHandle handle) const = 0;
};

/// GDI BitBlt into a 32-bit DIB, encoded with the Windows Imaging Component.
/// The capturing thread is made per-monitor DPI aware for the duration of the
/// call so a 4K display yields 4K pixels even from a DPI-unaware host process.
class GdiScreenshotter final : public Screenshotter {
public:
    [[nodiscard]] std::vector<std::uint8_t> captureScreen(bool allMonitors) const override;
    [[nodiscard]] std::vector<std::uint8_t> captureWindow(WindowHandle handle) const override;
};

}  // namespace jarvis::os
