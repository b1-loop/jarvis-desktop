#include "frame_clock.hpp"

#include "jarvis/overlay/model.hpp"

#include <dwmapi.h>
#include <windows.h>

#include <chrono>
#include <optional>

namespace jarvis::overlay {

namespace {

using Clock = OverlayModel::Clock;

/// QPC ticks to nanoseconds without overflowing for days of uptime.
std::chrono::nanoseconds ticksToNanoseconds(long long ticks, long long frequency) noexcept {
    constexpr long long kNanosPerSecond = 1'000'000'000;
    const long long seconds = ticks / frequency;
    const long long remainder = ticks % frequency;
    return std::chrono::nanoseconds{(seconds * kNanosPerSecond) +
                                    ((remainder * kNanosPerSecond) / frequency)};
}

/// The nominal refresh period (rateRefresh) when DWM reports one: the
/// measured qpcRefreshPeriod wanders by a millisecond or more on panels
/// with a variable or dynamic refresh rate, and a wandering period puts
/// slots off the real grid.
std::chrono::nanoseconds refreshPeriod(const DWM_TIMING_INFO& timing,
                                       const LARGE_INTEGER& frequency) noexcept {
    constexpr long long kNanosPerSecond = 1'000'000'000;
    const UNSIGNED_RATIO& rate = timing.rateRefresh;
    if (rate.uiNumerator != 0 && rate.uiDenominator != 0) {
        return std::chrono::nanoseconds{(kNanosPerSecond * rate.uiDenominator) / rate.uiNumerator};
    }
    return ticksToNanoseconds(static_cast<long long>(timing.qpcRefreshPeriod), frequency.QuadPart);
}

}  // namespace

std::optional<VBlankGrid> compositionGrid() noexcept {
    DWM_TIMING_INFO timing{};
    timing.cbSize = sizeof(timing);
    LARGE_INTEGER frequency{};
    LARGE_INTEGER now{};
    // hwnd must be null since Windows 8.1: the timing is the compositor's.
    if (FAILED(DwmGetCompositionTimingInfo(nullptr, &timing)) ||
        QueryPerformanceFrequency(&frequency) == 0 || frequency.QuadPart <= 0 ||
        timing.qpcRefreshPeriod == 0) {
        return std::nullopt;
    }
    const Clock::time_point clockNow = Clock::now();
    QueryPerformanceCounter(&now);
    const long long sinceVblank =
        now.QuadPart - static_cast<long long>(timing.qpcVBlank);  // may be negative (next vblank)
    const auto offset = std::chrono::duration_cast<Clock::duration>(
        ticksToNanoseconds(sinceVblank, frequency.QuadPart));
    return VBlankGrid{.vblank = clockNow - offset, .period = refreshPeriod(timing, frequency)};
}

}  // namespace jarvis::overlay

// Design notes:
// - DWM's own timing instead of IDXGIOutput::WaitForVBlank: waiting would
//   block the UI thread (and only knows the adapter's primary output),
//   while the composition clock is what actually decides when a frame of a
//   composed window reaches the screen, on whichever monitor it is.
// - The QPC reading and Clock::now() are taken back to back; on MSVC the
//   steady clock is QPC itself, so the conversion error is the few hundred
//   nanoseconds between the two calls.
