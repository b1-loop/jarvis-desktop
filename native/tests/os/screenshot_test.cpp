#include "jarvis/os/screenshot.hpp"

#include "jarvis/os/errors.hpp"
#include "jarvis/os/window_manager.hpp"

#include <catch2/catch_test_macros.hpp>

#include "os/desktop.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

using jarvis::os::GdiScreenshotter;
using jarvis::os::testing::requireDesktop;

namespace {

constexpr std::array<std::uint8_t, 8> kPngSignature{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};

bool isPng(const std::vector<std::uint8_t>& bytes) {
    return bytes.size() > kPngSignature.size() &&
           std::equal(kPngSignature.begin(), kPngSignature.end(), bytes.begin());
}

/// Width and height from the IHDR chunk, which always follows the signature.
std::pair<std::uint32_t, std::uint32_t> pngDimensions(const std::vector<std::uint8_t>& png) {
    constexpr std::size_t kWidthOffset = 16;
    const auto readBigEndian = [&png](std::size_t offset) {
        std::uint32_t value = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            value = (value << 8U) | static_cast<std::uint32_t>(png.at(offset + i));
        }
        return value;
    };
    return {readBigEndian(kWidthOffset), readBigEndian(kWidthOffset + 4)};
}

}  // namespace

TEST_CASE("the primary screen encodes to a PNG with its dimensions", "[os][screenshot]") {
    requireDesktop();
    const GdiScreenshotter screenshotter;

    const auto png = screenshotter.captureScreen(false);

    REQUIRE(isPng(png));
    const auto [width, height] = pngDimensions(png);
    REQUIRE(width > 0);
    REQUIRE(height > 0);
}

TEST_CASE("the virtual desktop is at least as large as the primary screen", "[os][screenshot]") {
    requireDesktop();
    const GdiScreenshotter screenshotter;

    const auto [primaryWidth, primaryHeight] = pngDimensions(screenshotter.captureScreen(false));
    const auto [allWidth, allHeight] = pngDimensions(screenshotter.captureScreen(true));

    REQUIRE(allWidth >= primaryWidth);
    REQUIRE(allHeight >= primaryHeight);
}

TEST_CASE("capturing a stale window raises WindowNotFoundError", "[os][screenshot]") {
    requireDesktop();
    const GdiScreenshotter screenshotter;

    REQUIRE_THROWS_AS(screenshotter.captureWindow(0xDEADBEEF), jarvis::os::WindowNotFoundError);
}

TEST_CASE("the active window, if any, captures as a PNG", "[os][screenshot]") {
    requireDesktop();
    const jarvis::os::Win32WindowManager windows;
    const auto active = windows.activeWindow();
    if (!active.has_value()) {
        SKIP("no active window");
        return;  // SKIP throws; the return keeps the optional access provably checked
    }
    const GdiScreenshotter screenshotter;

    REQUIRE(isPng(screenshotter.captureWindow(active->handle)));
}
