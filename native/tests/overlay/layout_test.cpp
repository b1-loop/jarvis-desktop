#include "jarvis/overlay/layout.hpp"

#include "jarvis/overlay/model.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <ctime>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using jarvis::overlay::anchorToCorner;
using jarvis::overlay::AssistantState;
using jarvis::overlay::bootGeometry;
using jarvis::overlay::centredIn;
using jarvis::overlay::charactersPerLine;
using jarvis::overlay::Corner;
using jarvis::overlay::enclosingPlacement;
using jarvis::overlay::HudGeometry;
using jarvis::overlay::hudLabel;
using jarvis::overlay::inflated;
using jarvis::overlay::lerpGeometry;
using jarvis::overlay::lerpRect;
using jarvis::overlay::panelGeometry;
using jarvis::overlay::PixelRect;
using jarvis::overlay::Placement;
using jarvis::overlay::RectF;
using jarvis::overlay::Rgba;
using jarvis::overlay::scaled;
using jarvis::overlay::stateColour;
using jarvis::overlay::stateLabel;
using jarvis::overlay::Theme;
using jarvis::overlay::toRectF;
using jarvis::overlay::translated;
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

TEST_CASE("the HUD caption names every state", "[overlay][layout]") {
    REQUIRE(hudLabel(AssistantState::Idle) == "STANDBY");
    REQUIRE(hudLabel(AssistantState::Listening) == "LISTENING");
    REQUIRE(hudLabel(AssistantState::Thinking) == "PROCESSING");
    REQUIRE(hudLabel(AssistantState::Speaking) == "SPEAKING");
}

TEST_CASE("rectangles interpolate, translate and round outwards", "[overlay][layout]") {
    const RectF from{.x = 0.0F, .y = 0.0F, .width = 100.0F, .height = 50.0F};
    const RectF to{.x = 200.0F, .y = 100.0F, .width = 10.0F, .height = 10.0F};
    REQUIRE(lerpRect(from, to, 0.0F) == from);
    REQUIRE(lerpRect(from, to, 1.0F) == to);
    REQUIRE(lerpRect(from, to, 0.5F) ==
            RectF{.x = 100.0F, .y = 50.0F, .width = 55.0F, .height = 30.0F});
    REQUIRE(translated(to, -200.0F, -100.0F) ==
            RectF{.x = 0.0F, .y = 0.0F, .width = 10.0F, .height = 10.0F});

    const Placement rounded =
        enclosingPlacement(RectF{.x = 10.4F, .y = 20.6F, .width = 5.2F, .height = 4.0F});
    REQUIRE(rounded == Placement{.x = 10, .y = 20, .width = 6, .height = 5});
    REQUIRE(toRectF(rounded) == RectF{.x = 10.0F, .y = 20.0F, .width = 6.0F, .height = 5.0F});
    REQUIRE(inflated(rounded, 3) == Placement{.x = 7, .y = 17, .width = 12, .height = 11});
}

TEST_CASE("the boot canvas is centred in the work area", "[overlay][layout]") {
    REQUIRE(centredIn(kWorkArea, 440, 440) ==
            Placement{.x = 740, .y = 300, .width = 440, .height = 440});
}

TEST_CASE("the panel ring sits left of the text column, inside the panel", "[overlay][layout]") {
    const RectF panel{.x = 1000.0F, .y = 800.0F, .width = 380.0F, .height = 132.0F};
    const HudGeometry hud = panelGeometry(panel, 14.0F);
    REQUIRE(hud.centreY == 866.0F);
    REQUIRE(hud.centreX - hud.radius >= panel.x);
    REQUIRE(hud.centreY - hud.radius >= panel.y);
    REQUIRE(hud.centreY + hud.radius <= panel.bottom());
    REQUIRE(hud.text.x > hud.centreX + hud.radius);
    REQUIRE(hud.text.right() <= panel.right());
    REQUIRE(hud.text.width > 200.0F);
}

TEST_CASE("the boot ring is centred above its title", "[overlay][layout]") {
    const RectF canvas{.x = 100.0F, .y = 100.0F, .width = 440.0F, .height = 440.0F};
    const HudGeometry hud = bootGeometry(canvas);
    REQUIRE(hud.centreX == 320.0F);
    REQUIRE(hud.text.y > hud.centreY + hud.radius);
    REQUIRE(hud.text.bottom() <= canvas.bottom());
}

TEST_CASE("geometry interpolates for the handoff", "[overlay][layout]") {
    const HudGeometry boot =
        bootGeometry(RectF{.x = 0.0F, .y = 0.0F, .width = 400.0F, .height = 400.0F});
    const HudGeometry panel =
        panelGeometry(RectF{.x = 800.0F, .y = 600.0F, .width = 380.0F, .height = 132.0F}, 14.0F);
    REQUIRE(lerpGeometry(boot, panel, 0.0F) == boot);
    REQUIRE(lerpGeometry(boot, panel, 1.0F) == panel);
    const HudGeometry middle = lerpGeometry(boot, panel, 0.5F);
    REQUIRE_THAT(middle.radius, WithinAbs((boot.radius + panel.radius) * 0.5F, 1e-4));
    const HudGeometry moved = translated(panel, -800.0F, -600.0F);
    REQUIRE(moved.radius == panel.radius);
    REQUIRE(moved.text.x == panel.text.x - 800.0F);
}

TEST_CASE("head truncation keeps the start and ends in an ellipsis", "[overlay][layout]") {
    using jarvis::overlay::truncateUtf8Head;
    REQUIRE(truncateUtf8Head("qwen2.5:7b", 20) == "qwen2.5:7b");
    REQUIRE(truncateUtf8Head("qwen2.5:7b", 5) == "qwen…");
    REQUIRE(truncateUtf8Head("Mikrofon (Hörlurar)", 12) == "Mikrofon (H…");
    REQUIRE(truncateUtf8Head("Mikrofon (Hörlurar)", 13) == "Mikrofon (Hö…");
    REQUIRE(utf8Length(truncateUtf8Head("ääääää", 3)) == 3);
    REQUIRE(truncateUtf8Head("anything", 0).empty());
    REQUIRE(truncateUtf8Head("ab", 1) == "…");
}

TEST_CASE("utf8Prefix types text on by whole code points", "[overlay][layout]") {
    using jarvis::overlay::utf8Prefix;
    REQUIRE(utf8Prefix("Hörlurar", 0).empty());
    REQUIRE(utf8Prefix("Hörlurar", 1) == "H");
    REQUIRE(utf8Prefix("Hörlurar", 2) == "Hö");
    REQUIRE(utf8Prefix("Hörlurar", 3) == "Hör");
    REQUIRE(utf8Prefix("abc", 99) == "abc");
    REQUIRE(utf8Prefix("", 3).empty());
}

TEST_CASE("panel sizes grow from compact to large and parse by name", "[overlay][layout]") {
    using jarvis::overlay::PanelSize;
    const auto compact = jarvis::overlay::panelMetrics(PanelSize::Compact);
    const auto normal = jarvis::overlay::panelMetrics(PanelSize::Normal);
    const auto large = jarvis::overlay::panelMetrics(PanelSize::Large);
    REQUIRE(compact.width == 420);
    REQUIRE(compact.height == 132);
    REQUIRE(normal.width > compact.width);
    REQUIRE(large.width > normal.width);
    REQUIRE(large.height > normal.height);
    REQUIRE(large.textScale > normal.textScale);
    REQUIRE(normal.textScale > compact.textScale);
}

TEST_CASE("panel sizes parse by name and take explicit overrides", "[overlay][layout]") {
    using jarvis::overlay::PanelSize;
    REQUIRE(std::ranges::all_of(
        std::array{PanelSize::Compact, PanelSize::Normal, PanelSize::Large}, [](PanelSize size) {
            return jarvis::overlay::parsePanelSize(jarvis::overlay::toString(size)) == size;
        }));
    REQUIRE_FALSE(jarvis::overlay::parsePanelSize("huge").has_value());
    const auto large = jarvis::overlay::panelMetrics(PanelSize::Large);
    const auto custom = jarvis::overlay::withOverride(large, 500, 0);
    REQUIRE(custom.width == 500);
    REQUIRE(custom.height == large.height);
    // The large ring is a showpiece: about 280 px across at 96 DPI.
    const RectF panel{.x = 0.0F, .y = 0.0F, .width = 760.0F, .height = 300.0F};
    REQUIRE(panelGeometry(panel, 21.0F).radius * 2.0F > 260.0F);
}

TEST_CASE("the idle clock and date read like an instrument", "[overlay][layout]") {
    std::tm local{};
    local.tm_hour = 9;
    local.tm_min = 5;
    local.tm_sec = 7;
    local.tm_wday = 3;
    local.tm_mday = 22;
    local.tm_mon = 9;
    REQUIRE(jarvis::overlay::formatClock(local) == "09:05:07");
    REQUIRE(jarvis::overlay::formatDate(local) == "WED 22 OCT");
}

TEST_CASE("uptime is hours, minutes and seconds", "[overlay][layout]") {
    REQUIRE(jarvis::overlay::formatUptime(0.0F) == "00:00:00");
    REQUIRE(jarvis::overlay::formatUptime(3725.9F) == "01:02:05");
    REQUIRE(jarvis::overlay::formatUptime(-5.0F) == "00:00:00");
    REQUIRE(jarvis::overlay::formatUptime(360000.0F) == "100:00:00");
}
