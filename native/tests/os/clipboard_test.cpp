#include "jarvis/os/clipboard.hpp"

#include <catch2/catch_test_macros.hpp>

#include "os/desktop.hpp"

#include <string>

using jarvis::os::Win32Clipboard;
using jarvis::os::testing::requireDesktop;
using jarvis::os::testing::requireStateChangesAllowed;

TEST_CASE("reading the clipboard never throws on a desktop", "[os][clipboard]") {
    requireDesktop();
    const Win32Clipboard clipboard;

    REQUIRE_NOTHROW(clipboard.text());
}

TEST_CASE("text survives a set/get round trip", "[os][clipboard][state]") {
    requireStateChangesAllowed();
    Win32Clipboard clipboard;
    const std::string original = clipboard.text();
    const std::string sample = "Jarvis clipboard \xE2\x9C\x93 \xF0\x9F\x9A\x80\nline two";

    clipboard.setText(sample);
    REQUIRE(clipboard.text() == sample);

    clipboard.setText(original);
    REQUIRE(clipboard.text() == original);
}

TEST_CASE("an empty string clears the text", "[os][clipboard][state]") {
    requireStateChangesAllowed();
    Win32Clipboard clipboard;
    const std::string original = clipboard.text();

    clipboard.setText("");
    REQUIRE(clipboard.text().empty());

    clipboard.setText(original);
}
