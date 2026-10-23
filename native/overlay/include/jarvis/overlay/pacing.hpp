/// @file pacing.hpp
/// @brief When the UI thread draws its next frame, aligned to the display's
/// vertical blanks.
///
/// DWM shows a composed window's new frame only on a vertical blank. A
/// free-running timer at "60 fps" therefore beats against the refresh: on a
/// 240 Hz panel a frame stays up for 4 or 5 refreshes depending on a few
/// hundred microseconds of timer jitter, and on a 60 Hz panel the timer's
/// phase drifts across the vblank every few seconds and drops or repeats a
/// frame there. Both read as micro-stutter even with every frame on time.
///
/// The cure is to give every frame a whole number of refreshes and to wake
/// just after a vblank (a "slot"), so the frame is presented well before
/// the next one whatever the jitter. These are the pure decisions; the
/// window measures the grid (frame_clock.hpp) and sleeps. Times are seconds
/// on any one steady clock.

#pragma once

#include <optional>

namespace jarvis::overlay::pacing {

/// Where in a refresh interval the loop wakes: this share of a period after
/// the slot's vblank. Late enough that a timer firing early never lands
/// before the vblank, early enough to leave most of the interval for the
/// frame (the HUD needs 1-3 ms; a 240 Hz refresh is 4.2 ms).
inline constexpr double kWakePhase = 0.15;

/// Refreshes each frame stays on screen for a target of @p fps on a display
/// refreshing at @p refreshHz: the nearest whole number, at least 1. A
/// 60 fps target is 1 refresh at 60 Hz, 2 at 144 Hz (72 fps) and 4 at
/// 240 Hz; a 30 fps idle target is 2, 5 and 8.
[[nodiscard]] int refreshesPerFrame(double refreshHz, int fps) noexcept;

/// The vblank grid as last measured: one vblank and the period.
struct Grid {
    double vblank{0.0};
    double period{0.0};
};

/// The slot (a vblank time on @p grid) the next frame wakes after.
///
/// Normally @p previousSlot plus @p refreshes periods, re-snapped onto the
/// grid (the measured grid may have drifted a little since). Without a
/// previous slot (the first frame, after a hidden spell, or a wake-up
/// because something changed) or when that slot's wake time has already
/// passed (a stall), it is the first slot whose wake time is not before
/// @p now, so a late loop resynchronises instead of trying to catch up.
[[nodiscard]] double nextSlot(const Grid& grid, std::optional<double> previousSlot, int refreshes,
                              double now) noexcept;

/// When to wake for @p slot: kWakePhase of a period after it.
[[nodiscard]] double wakeTime(const Grid& grid, double slot) noexcept;

}  // namespace jarvis::overlay::pacing
