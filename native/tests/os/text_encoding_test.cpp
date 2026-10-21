#include "jarvis/os/text_encoding.hpp"

#include "jarvis/os/errors.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

using jarvis::os::utf8ToWide;
using jarvis::os::wideToUtf8;

TEST_CASE("ascii converts unchanged in both directions", "[os][text]") {
    REQUIRE(utf8ToWide("hello") == L"hello");
    REQUIRE(wideToUtf8(L"hello") == "hello");
}

TEST_CASE("empty input yields empty output", "[os][text]") {
    REQUIRE(utf8ToWide("").empty());
    REQUIRE(wideToUtf8(L"").empty());
}

TEST_CASE("characters outside ascii round-trip", "[os][text]") {
    const std::string accented = "h\xC3\xA9llo";  // "héllo"
    REQUIRE(utf8ToWide(accented) == L"héllo");
    REQUIRE(wideToUtf8(utf8ToWide(accented)) == accented);
}

TEST_CASE("characters beyond the basic plane become surrogate pairs", "[os][text]") {
    const std::string rocket = "\xF0\x9F\x9A\x80";  // U+1F680

    const std::wstring wide = utf8ToWide(rocket);

    REQUIRE(wide.size() == 2);
    REQUIRE(wide.at(0) == 0xD83D);
    REQUIRE(wide.at(1) == 0xDE80);
    REQUIRE(wideToUtf8(wide) == rocket);
}

TEST_CASE("invalid utf-8 is rejected rather than replaced", "[os][text]") {
    REQUIRE_THROWS_AS(utf8ToWide("\xFF\xFE"), jarvis::os::OsError);
    REQUIRE_THROWS_AS(utf8ToWide("\xC3"), jarvis::os::OsError);  // truncated sequence
}

TEST_CASE("an unpaired surrogate is rejected", "[os][text]") {
    const std::wstring lone{static_cast<wchar_t>(0xD83D)};
    REQUIRE_THROWS_AS(wideToUtf8(lone), jarvis::os::OsError);
}

TEST_CASE("embedded nul characters survive", "[os][text]") {
    const std::string withNul{"a\0b", 3};

    const std::wstring wide = utf8ToWide(withNul);

    REQUIRE(wide.size() == 3);
    REQUIRE(wideToUtf8(wide) == withNul);
}
