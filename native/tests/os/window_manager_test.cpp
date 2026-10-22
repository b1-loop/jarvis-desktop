#include "jarvis/os/window_manager.hpp"

#include "jarvis/os/errors.hpp"

#include <catch2/catch_test_macros.hpp>

#include "os/desktop.hpp"

#include <algorithm>

using jarvis::os::Win32WindowManager;
using jarvis::os::WindowInfo;
using jarvis::os::WindowNotFoundError;
using jarvis::os::testing::requireDesktop;

TEST_CASE("listed windows are titled and belong to a process", "[os][windows]") {
    requireDesktop();
    const Win32WindowManager manager;

    const auto windows = manager.listWindows();

    for (const WindowInfo& window : windows) {
        REQUIRE_FALSE(window.title.empty());
        REQUIRE(window.handle != 0);
        REQUIRE(window.pid != 0);
    }
}

TEST_CASE("describe agrees with the listing", "[os][windows]") {
    requireDesktop();
    const Win32WindowManager manager;
    const auto windows = manager.listWindows();
    if (windows.empty()) {
        SKIP("no top-level windows on this desktop");
    }

    const WindowInfo described = manager.describe(windows.front().handle);

    REQUIRE(described.handle == windows.front().handle);
    REQUIRE(described.pid == windows.front().pid);
}

TEST_CASE("the active window, if any, is described without throwing", "[os][windows]") {
    requireDesktop();
    const Win32WindowManager manager;

    const auto active = manager.activeWindow();

    if (active.has_value()) {
        REQUIRE(active->handle != 0);
        REQUIRE(active->pid != 0);
    }
}

TEST_CASE("a stale handle raises WindowNotFoundError", "[os][windows]") {
    requireDesktop();
    Win32WindowManager manager;
    constexpr jarvis::os::WindowHandle kBogus = 0xDEADBEEF;

    REQUIRE_THROWS_AS(manager.describe(kBogus), WindowNotFoundError);
    REQUIRE_THROWS_AS(manager.focus(kBogus), WindowNotFoundError);
    REQUIRE_THROWS_AS(manager.minimize(kBogus), WindowNotFoundError);
    REQUIRE_THROWS_AS(manager.close(kBogus), WindowNotFoundError);
}

TEST_CASE("an unmatched title raises WindowNotFoundError", "[os][windows]") {
    requireDesktop();
    Win32WindowManager manager;

    REQUIRE_THROWS_AS(manager.focusByTitle("jarvis-no-such-window-7f3a"), WindowNotFoundError);
}
