#include "jarvis/overlay/layout.hpp"

#include "jarvis/overlay/model.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using jarvis::overlay::anchorToCorner;
using jarvis::overlay::AssistantState;
using jarvis::overlay::charactersPerLine;
using jarvis::overlay::Corner;
using jarvis::overlay::PixelRect;
using jarvis::overlay::Placement;
using jarvis::overlay::Rgba;
using jarvis::overlay::scaled;
using jarvis::overlay::stateColour;
using jarvis::overlay::stateLabel;
using jarvis::overlay::Theme;
using jarvis::overlay::truncateUtf8Tail;
using jarvis::overlay::utf8Length;
using jarvis::overlay::waveformBars;
using jarvis::overlay::withAlpha;

namespace {

constexpr PixelRect kWorkArea{.left = 0, .top = 0, .right = 1920, .bottom = 1040};

}  // namespace

TEST_CASE("every state has its own colour and label", "[overlay][layout]") {
    const Theme theme;
    REQUIRE(stateColour(theme, AssistantState::Idle) == theme.idle);
    REQUIRE(stateColour(theme, AssistantState::Listening) == theme.listening);
    REQUIRE(stateColour(theme, AssistantState::Thinking) == theme.thinking);
    REQUIRE(stateColour(theme, AssistantState::Speaking) == theme.speaking);

    REQUIRE(stateLabel(AssistantState::Idle) == "Idle");
    REQUIRE(stateLabel(AssistantState::Listening) == "Listening");
    REQUIRE(stateLabel(AssistantState::Thinking) == "Thinking");
    REQUIRE(stateLabel(AssistantState::Speaking) == "Speaking");
}

TEST_CASE("withAlpha scales and clamps the alpha channel only", "[overlay][layout]") {
    const Rgba colour{.r = 0.1F, .g = 0.2F, .b = 0.3F, .a = 0.8F};

    const Rgba half = withAlpha(colour, 0.5F);
    REQUIRE_THAT(half.a, WithinAbs(0.4, 1e-6));
    REQUIRE(half.r == 0.1F);
    REQUIRE(withAlpha(colour, 5.0F).a == 1.0F);
    REQUIRE(withAlpha(colour, -1.0F).a == 0.0F);
}

TEST_CASE("utf8Length counts code points, not bytes", "[overlay][layout]") {
    REQUIRE(utf8Length("") == 0);
    REQUIRE(utf8Length("abc") == 3);
    REQUIRE(utf8Length("héllo") == 5);   // é is two bytes
    REQUIRE(utf8Length("日本語") == 3);  // three bytes each
    REQUIRE(utf8Length("…") == 1);
}

TEST_CASE("short text is returned untouched", "[overlay][layout]") {
    REQUIRE(truncateUtf8Tail("hello", 5) == "hello");
    REQUIRE(truncateUtf8Tail("hello", 50) == "hello");
    REQUIRE(truncateUtf8Tail("", 3).empty());
}

TEST_CASE("long text keeps its tail behind an ellipsis, exactly maxChars wide",
          "[overlay][layout]") {
    const std::string cut = truncateUtf8Tail("the quick brown fox", 8);

    REQUIRE(cut == "…own fox");
    REQUIRE(utf8Length(cut) == 8);
}

TEST_CASE("truncation never splits a multi-byte character", "[overlay][layout]") {
    const std::string cut = truncateUtf8Tail("ääääää", 3);

    REQUIRE(cut == "…ää");
    REQUIRE(utf8Length(cut) == 3);
}

TEST_CASE("a width of zero or one degrades gracefully", "[overlay][layout]") {
    REQUIRE(truncateUtf8Tail("hello", 0).empty());
    REQUIRE(truncateUtf8Tail("hello", 1) == "…");
}

TEST_CASE("charactersPerLine floors and refuses to divide by zero", "[overlay][layout]") {
    REQUIRE(charactersPerLine(100.0F, 7.0F) == 14);
    REQUIRE(charactersPerLine(100.0F, 0.0F) == 0);
    REQUIRE(charactersPerLine(-5.0F, 7.0F) == 0);
}

TEST_CASE("waveform bars pass levels through when the counts match", "[overlay][layout]") {
    const std::vector<float> levels{0.1F, 0.5F, 0.9F};

    REQUIRE(waveformBars(levels, 3, 0.0F) == levels);
}

TEST_CASE("waveform bars take the peak of each window when downsampling", "[overlay][layout]") {
    const std::vector<float> levels{0.1F, 0.9F, 0.2F, 0.3F, 0.0F, 0.6F};

    REQUIRE(waveformBars(levels, 3, 0.0F) == std::vector<float>{0.9F, 0.3F, 0.6F});
}

TEST_CASE("waveform bars never fall below the floor and handle empty input", "[overlay][layout]") {
    REQUIRE(waveformBars({}, 2, 0.05F) == std::vector<float>{0.05F, 0.05F});
    REQUIRE(waveformBars(std::vector<float>{0.0F, 0.5F}, 2, 0.1F) ==
            std::vector<float>{0.1F, 0.5F});
    REQUIRE(waveformBars(std::vector<float>{0.5F}, 0, 0.1F).empty());
}

TEST_CASE("more bars than levels repeats levels instead of leaving gaps", "[overlay][layout]") {
    const std::vector<float> bars = waveformBars(std::vector<float>{0.2F, 0.8F}, 4, 0.0F);

    REQUIRE(bars == std::vector<float>{0.2F, 0.2F, 0.8F, 0.8F});
}

TEST_CASE("scaled rounds logical pixels to the nearest physical pixel", "[overlay][layout]") {
    REQUIRE(scaled(100, 1.0F) == 100);
    REQUIRE(scaled(100, 1.5F) == 150);
    REQUIRE(scaled(33, 1.25F) == 41);
}

TEST_CASE("the panel is anchored inside the work area with a margin", "[overlay][layout]") {
    REQUIRE(anchorToCorner(Corner::TopLeft, kWorkArea, 300, 100, 20) ==
            Placement{.x = 20, .y = 20, .width = 300, .height = 100});
    REQUIRE(anchorToCorner(Corner::TopRight, kWorkArea, 300, 100, 20) ==
            Placement{.x = 1600, .y = 20, .width = 300, .height = 100});
    REQUIRE(anchorToCorner(Corner::BottomLeft, kWorkArea, 300, 100, 20) ==
            Placement{.x = 20, .y = 920, .width = 300, .height = 100});
    REQUIRE(anchorToCorner(Corner::BottomRight, kWorkArea, 300, 100, 20) ==
            Placement{.x = 1600, .y = 920, .width = 300, .height = 100});
}

TEST_CASE("placement respects a work area that does not start at the origin", "[overlay][layout]") {
    constexpr PixelRect kSecondMonitor{.left = 1920, .top = 40, .right = 3840, .bottom = 1080};

    REQUIRE(anchorToCorner(Corner::BottomRight, kSecondMonitor, 300, 100, 0) ==
            Placement{.x = 3540, .y = 980, .width = 300, .height = 100});
}
