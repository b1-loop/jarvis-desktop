#include "jarvis/os/errors.hpp"
#include "jarvis/os/hotkey.hpp"

#include <catch2/catch_test_macros.hpp>

#include "os/desktop.hpp"

using jarvis::os::HotkeyManager;
using jarvis::os::parseHotkey;
using jarvis::os::testing::requireDesktop;

// ctrl+alt+shift+f24: a combination no real keyboard or application uses,
// so registering it briefly cannot collide with the developer's own hotkeys.
namespace {
constexpr const char* kUnusedCombination = "ctrl+alt+shift+f24";
}

TEST_CASE("a manager starts with nothing registered and no events", "[os][hotkey]") {
    requireDesktop();
    HotkeyManager manager;

    REQUIRE(manager.registeredCount() == 0);
    REQUIRE(manager.pollEvents().empty());
}

TEST_CASE("register returns distinct ids and unregister releases them", "[os][hotkey]") {
    requireDesktop();
    HotkeyManager manager;

    const int first = manager.registerHotkey(parseHotkey(kUnusedCombination));
    const int second = manager.registerHotkey(parseHotkey("ctrl+alt+shift+f23"));

    REQUIRE(first != second);
    REQUIRE(manager.registeredCount() == 2);

    manager.unregisterHotkey(first);
    REQUIRE(manager.registeredCount() == 1);
    manager.unregisterHotkey(first);  // unknown ids are ignored
    REQUIRE(manager.registeredCount() == 1);
}

TEST_CASE("registering the same combination twice fails with OsError", "[os][hotkey]") {
    requireDesktop();
    HotkeyManager manager;
    manager.registerHotkey(parseHotkey(kUnusedCombination));

    REQUIRE_THROWS_AS(manager.registerHotkey(parseHotkey(kUnusedCombination)), jarvis::os::OsError);
    REQUIRE(manager.registeredCount() == 1);
}

TEST_CASE("a combination is free again after its manager is destroyed", "[os][hotkey]") {
    requireDesktop();
    {
        HotkeyManager manager;
        manager.registerHotkey(parseHotkey(kUnusedCombination));
    }
    HotkeyManager manager;

    REQUIRE_NOTHROW(manager.registerHotkey(parseHotkey(kUnusedCombination)));
}
