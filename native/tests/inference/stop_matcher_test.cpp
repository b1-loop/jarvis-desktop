#include "jarvis/inference/stop_matcher.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <string_view>
#include <vector>

using jarvis::inference::completeUtf8Prefix;
using jarvis::inference::StopMatcher;

namespace {

/// Feed every piece and concatenate what the matcher releases.
std::string drain(StopMatcher& matcher, const std::vector<std::string>& pieces) {
    std::string out;
    for (const auto& piece : pieces) {
        out += matcher.feed(piece);
    }
    return out;
}

}  // namespace

TEST_CASE("ASCII text is always a complete UTF-8 prefix", "[inference][utf8]") {
    REQUIRE(completeUtf8Prefix("hello") == 5);
    REQUIRE(completeUtf8Prefix("") == 0);
}

TEST_CASE("a complete multi-byte character is kept", "[inference][utf8]") {
    const std::string_view cyrillic = "\xD0\x96";       // Ж
    const std::string_view emoji = "\xF0\x9F\x98\x80";  // grinning face
    REQUIRE(completeUtf8Prefix(cyrillic) == 2);
    REQUIRE(completeUtf8Prefix(emoji) == 4);
}

TEST_CASE("a truncated multi-byte character is held back", "[inference][utf8]") {
    REQUIRE(completeUtf8Prefix("ab\xD0") == 2);
    REQUIRE(completeUtf8Prefix("ab\xF0\x9F\x98") == 2);
    REQUIRE(completeUtf8Prefix("\xE2\x82") == 0);
}

TEST_CASE("without stop strings every piece passes through at once", "[inference][stop]") {
    StopMatcher matcher({});
    REQUIRE(matcher.feed("Hello") == "Hello");
    REQUIRE(matcher.feed(", world") == ", world");
    REQUIRE_FALSE(matcher.stopped());
    REQUIRE(matcher.flush().empty());
}

TEST_CASE("a stop string ends the stream and is never delivered", "[inference][stop]") {
    StopMatcher matcher({"</s>"});
    const std::string text = drain(matcher, {"Hi", " there", "</s>", "ignored"});
    REQUIRE(text == "Hi there");
    REQUIRE(matcher.stopped());
}

TEST_CASE("a stop string split across pieces is still caught", "[inference][stop]") {
    StopMatcher matcher({"User:"});
    REQUIRE(matcher.feed("Sure. Us") == "Sure. ");  // "Us" could start "User:"
    REQUIRE(matcher.feed("er").empty());
    REQUIRE(matcher.feed(": next").empty());
    REQUIRE(matcher.stopped());
}

TEST_CASE("held-back text is released when the stop string does not follow", "[inference][stop]") {
    StopMatcher matcher({"User:"});
    REQUIRE(matcher.feed("Use").empty());
    REQUIRE(matcher.feed("ful tip") == "Useful tip");
    REQUIRE_FALSE(matcher.stopped());
}

TEST_CASE("flush releases a pending partial match", "[inference][stop]") {
    StopMatcher matcher({"END"});
    REQUIRE(matcher.feed("the EN") == "the ");
    REQUIRE(matcher.flush() == "EN");
}

TEST_CASE("the longest of several stop strings decides how much is held", "[inference][stop]") {
    StopMatcher matcher({"\n\n", "Human:"});
    REQUIRE(matcher.feed("line\n") == "line");
    REQUIRE(matcher.feed("\n").empty());
    REQUIRE(matcher.stopped());
}

TEST_CASE("a multi-byte character split across pieces is reassembled", "[inference][stop]") {
    StopMatcher matcher({});
    REQUIRE(matcher.feed("\xD0").empty());
    REQUIRE(matcher.feed("\x96!") == "\xD0\x96!");
}

TEST_CASE("flush drops a trailing incomplete character", "[inference][stop]") {
    StopMatcher matcher({});
    REQUIRE(matcher.feed("ok\xF0\x9F") == "ok");
    REQUIRE(matcher.flush().empty());
}

TEST_CASE("empty stop strings are ignored", "[inference][stop]") {
    StopMatcher matcher({"", "X"});
    REQUIRE(matcher.feed("abc") == "abc");
    REQUIRE_FALSE(matcher.stopped());
}
