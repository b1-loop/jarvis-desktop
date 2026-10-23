#include "jarvis/overlay/pacing.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

namespace jarvis::overlay::pacing {

int refreshesPerFrame(double refreshHz, int fps) noexcept {
    if (refreshHz <= 0.0 || fps <= 0) {
        return 1;
    }
    return std::max(static_cast<int>(std::lround(refreshHz / static_cast<double>(fps))), 1);
}

double nextSlot(const Grid& grid, std::optional<double> previousSlot, int refreshes,
                double now) noexcept {
    if (grid.period <= 0.0) {
        return now;
    }
    // The first slot whose wake time is not in the past.
    const double earliest =
        grid.vblank + (std::ceil(((now - grid.vblank) / grid.period) - kWakePhase) * grid.period);
    if (!previousSlot.has_value()) {
        return earliest;
    }
    const double planned =
        *previousSlot + (static_cast<double>(std::max(refreshes, 1)) * grid.period);
    const double snapped =
        grid.vblank + (std::round((planned - grid.vblank) / grid.period) * grid.period);
    return std::max(snapped, earliest);
}

double wakeTime(const Grid& grid, double slot) noexcept {
    return slot + (kWakePhase * std::max(grid.period, 0.0));
}

}  // namespace jarvis::overlay::pacing

// Design notes:
// - Slots are re-snapped onto the freshly measured grid every frame rather
//   than accumulated, so a refresh period DWM reports slightly off (it is a
//   measurement) or a dynamic refresh rate changing it cannot make the
//   wake-ups drift across a vblank.
// - A stall resynchronises to the next slot instead of running frames back
//   to back to catch up: every animation is a function of time, so a
//   skipped slot loses nothing but that frame.
