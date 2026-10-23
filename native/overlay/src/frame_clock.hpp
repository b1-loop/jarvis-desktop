/// @file frame_clock.hpp
/// @brief The display's vertical-blank grid as DWM reports it (private).
///
/// The overlay is composed by DWM, which shows a new frame only on a
/// vertical blank. Pacing frames on a free-running timer therefore beats
/// against the refresh: on a 240 Hz panel a "60 fps" timer frame stays on
/// screen for 4 or 5 refreshes depending on a fraction of a millisecond of
/// timer jitter. window.cpp uses this grid to wake just after a vblank
/// instead (pacing.hpp has the maths), and to measure how evenly frames
/// actually reached the screen.

#pragma once

#include "jarvis/overlay/model.hpp"

#include <chrono>
#include <optional>

namespace jarvis::overlay {

/// One vertical blank and the refresh period, on OverlayModel's clock.
struct VBlankGrid {
    OverlayModel::Clock::time_point vblank;
    std::chrono::nanoseconds period{0};
};

/// The composition clock of DWM (DwmGetCompositionTimingInfo), converted
/// from QPC ticks to OverlayModel::Clock; nullopt when DWM does not report
/// one (composition off, a remote session that has no vblank).
[[nodiscard]] std::optional<VBlankGrid> compositionGrid() noexcept;

}  // namespace jarvis::overlay
