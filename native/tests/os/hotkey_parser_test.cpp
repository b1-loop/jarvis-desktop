#include "jarvis/os/hotkey.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

using jarvis::os::formatHotkey;
using jarvis::os::Hotkey;
using jarvis::os::HotkeyModifier;
using jarvis::os::parseHotkey;
using jarvis::os::toBits;

namespace {

constexpr std::uint32_t kVkJ = 0x4A;
constexpr std::uint32_t kVkF5 = 0x74;
constexpr std::uint32_t kVkSpace = 0x20;
constexpr std::uint32_t kVkF24 = 0x87;

}  // namespace

TEST_CASE("ctrl+alt+j parses into Win32 modifier bits and a virtual key", "[os][hotkey]") {
    const Hotkey hotkey = parseHotkey("ctrl+alt+j");

    REQUIRE(hotkey.modifiers == (toBits(HotkeyModifier::Ctrl) | toBits(HotkeyModifier::Alt)));
    REQUIRE(hotkey.virtualKey == kVkJ);
}

TEST_CASE("parsing ignores case and whitespace", "[os][hotkey]") {
    REQUIRE(parseHotkey(" Ctrl + Shift + F5 ") ==
            Hotkey{.modifiers = toBits(HotkeyModifier::Ctrl) | toBits(HotkeyModifier::Shift),
                   .virtualKey = kVkF5});
}

TEST_CASE("modifier aliases map to the same bits", "[os][hotkey]") {
    REQUIRE(parseHotkey("control+a") == parseHotkey("ctrl+a"));
    REQUIRE(parseHotkey("super+space") == parseHotkey("win+space"));
    REQUIRE(parseHotkey("cmd+space").modifiers == toBits(HotkeyModifier::Win));
    REQUIRE(parseHotkey("win+space").virtualKey == kVkSpace);
}

TEST_CASE("a bare key is a valid hotkey", "[os][hotkey]") {
    const Hotkey hotkey = parseHotkey("f24");

    REQUIRE(hotkey.modifiers == toBits(HotkeyModifier::None));
    REQUIRE(hotkey.virtualKey == kVkF24);
}

TEST_CASE("named keys and digits are recognised", "[os][hotkey]") {
    REQUIRE(parseHotkey("ctrl+enter").virtualKey == 0x0D);
    REQUIRE(parseHotkey("ctrl+return").virtualKey == 0x0D);
    REQUIRE(parseHotkey("alt+escape").virtualKey == 0x1B);
    REQUIRE(parseHotkey("shift+7").virtualKey == '7');
    REQUIRE(parseHotkey("ctrl+pageup").virtualKey == 0x21);
}

TEST_CASE("malformed hotkeys raise invalid_argument", "[os][hotkey]") {
    REQUIRE_THROWS_AS(parseHotkey(""), std::invalid_argument);
    REQUIRE_THROWS_AS(parseHotkey("ctrl+"), std::invalid_argument);
    REQUIRE_THROWS_AS(parseHotkey("ctrl+alt"), std::invalid_argument);
    REQUIRE_THROWS_AS(parseHotkey("ctrl+ctrl+j"), std::invalid_argument);
    REQUIRE_THROWS_AS(parseHotkey("ctrl+j+k"), std::invalid_argument);
    REQUIRE_THROWS_AS(parseHotkey("ctrl+f25"), std::invalid_argument);
    REQUIRE_THROWS_AS(parseHotkey("ctrl+bananas"), std::invalid_argument);
}

TEST_CASE("the error names the offending token", "[os][hotkey]") {
    REQUIRE_THROWS_WITH(parseHotkey("ctrl+bananas"), Catch::Matchers::ContainsSubstring("bananas"));
}

TEST_CASE("format produces the canonical spelling", "[os][hotkey]") {
    REQUIRE(formatHotkey(parseHotkey("J + Alt + Control")) == "ctrl+alt+j");
    REQUIRE(formatHotkey(parseHotkey("shift+win+F12")) == "shift+win+f12");
    REQUIRE(formatHotkey(parseHotkey("space")) == "space");
    REQUIRE(formatHotkey(parseHotkey("ctrl+3")) == "ctrl+3");
}

TEST_CASE("format and parse round-trip every canonical spelling", "[os][hotkey]") {
    for (const std::string spec : {"ctrl+alt+j", "alt+f4", "win+space", "ctrl+shift+esc"}) {
        REQUIRE(formatHotkey(parseHotkey(spec)) == spec);
    }
}

TEST_CASE("format rejects a virtual key without a name", "[os][hotkey]") {
    REQUIRE_THROWS_AS(formatHotkey(Hotkey{.modifiers = 0, .virtualKey = 0xFFFF}),
                      std::invalid_argument);
}
