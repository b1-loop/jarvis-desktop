#include "jarvis/os/process_control.hpp"

#include "jarvis/os/errors.hpp"

#include <catch2/catch_test_macros.hpp>

#include "os/desktop.hpp"

using jarvis::os::ProcessNotFoundError;
using jarvis::os::Win32ProcessControl;

TEST_CASE("closing a pid that does not exist raises ProcessNotFoundError", "[os][process]") {
    Win32ProcessControl control;
    // Pids are multiples of four on Windows; this one is far above anything
    // a desktop session allocates and not a multiple of four either.
    constexpr std::uint32_t kBogusPid = 0x7FFFFFFD;

    REQUIRE_THROWS_AS(control.close(kBogusPid, false), ProcessNotFoundError);
    REQUIRE_THROWS_AS(control.close(kBogusPid, true), ProcessNotFoundError);
}

TEST_CASE("closing by an unknown name raises ProcessNotFoundError", "[os][process]") {
    Win32ProcessControl control;

    REQUIRE_THROWS_AS(control.closeByName("jarvis-no-such-process-xyz", false),
                      ProcessNotFoundError);
}

TEST_CASE("launching a target the shell cannot find raises OsError", "[os][process]") {
    jarvis::os::testing::requireDesktop();
    Win32ProcessControl control;

    REQUIRE_THROWS_AS(control.launch("jarvis-no-such-program-xyz.exe", ""), jarvis::os::OsError);
}
