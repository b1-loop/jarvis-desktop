#include "jarvis/os/system_controls.hpp"

#include "jarvis/os/errors.hpp"

#include <catch2/catch_test_macros.hpp>

#include "os/desktop.hpp"

#include <optional>

using jarvis::os::OsError;
using jarvis::os::UnsupportedError;
using jarvis::os::Win32SystemControls;
using jarvis::os::testing::requireStateChangesAllowed;

namespace {

/// Volume reads need an audio endpoint, which a CI runner may lack.
std::optional<int> readVolume(const Win32SystemControls& controls) {
    try {
        return controls.volume();
    } catch (const OsError&) {
        return std::nullopt;
    }
}

int volumeOrSkip(const Win32SystemControls& controls) {
    const std::optional<int> volume = readVolume(controls);
    if (!volume.has_value()) {
        SKIP("no audio endpoint on this machine");
    }
    return volume.value_or(0);
}

}  // namespace

TEST_CASE("volume reads as a percentage", "[os][system]") {
    const Win32SystemControls controls;

    const int volume = volumeOrSkip(controls);

    REQUIRE(volume >= 0);
    REQUIRE(volume <= 100);
    REQUIRE_NOTHROW(controls.muted());
}

TEST_CASE("brightness is a percentage or reported unsupported", "[os][system]") {
    const Win32SystemControls controls;

    try {
        const int brightness = controls.brightness();
        REQUIRE(brightness >= 0);
        REQUIRE(brightness <= 100);
    } catch (const UnsupportedError&) {
        SUCCEED("no software-brightness display on this machine");
    }
}

TEST_CASE("volume survives a set/get round trip", "[os][system][state]") {
    requireStateChangesAllowed();
    Win32SystemControls controls;
    const int original = volumeOrSkip(controls);
    const bool wasMuted = controls.muted();

    controls.setVolume(37);
    REQUIRE(controls.volume() == 37);
    controls.setMuted(wasMuted);

    controls.setVolume(original);
    REQUIRE(controls.volume() == original);
}

TEST_CASE("volume values are clamped to 0-100", "[os][system][state]") {
    requireStateChangesAllowed();
    Win32SystemControls controls;
    const int original = volumeOrSkip(controls);

    controls.setVolume(250);
    REQUIRE(controls.volume() == 100);
    controls.setVolume(-5);
    REQUIRE(controls.volume() == 0);

    controls.setVolume(original);
}
