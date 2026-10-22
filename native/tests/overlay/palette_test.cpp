#include "jarvis/overlay/palette.hpp"

#include "jarvis/overlay/layout.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using jarvis::overlay::allThemeNames;
using jarvis::overlay::contrastRatio;
using jarvis::overlay::deriveTheme;
using jarvis::overlay::fromOklch;
using jarvis::overlay::kDefaultAccent;
using jarvis::overlay::Oklch;
using jarvis::overlay::parseHexColour;
using jarvis::overlay::parseThemeName;
using jarvis::overlay::perceptualDistance;
using jarvis::overlay::Rgba;
using jarvis::overlay::Theme;
using jarvis::overlay::themeFor;
using jarvis::overlay::ThemeName;
using jarvis::overlay::toHex;
using jarvis::overlay::toOklch;
using jarvis::overlay::toString;

namespace {

// Legibility floors: body text at WCAG AAA, secondary text and the warning
// light at AA, all against the glass plate drawn behind them.
constexpr float kMinTextContrast = 7.0F;
constexpr float kMinSecondaryContrast = 4.5F;
// "Clearly different" in OKLab: a just noticeable difference is about 0.02.
constexpr float kMinStateDistance = 0.05F;
constexpr float kMinWarningDistance = 0.1F;
// Presets: ice and cyan share a hue family and differ mostly in chroma.
constexpr float kMinPresetDistance = 0.07F;
constexpr float kTolerance = 1e-3F;

constexpr Rgba rgb(float r, float g, float b) noexcept {
    return Rgba{.r = r, .g = g, .b = b, .a = 1.0F};
}

constexpr Rgba kWhite = rgb(1.0F, 1.0F, 1.0F);
constexpr Rgba kBlack = rgb(0.0F, 0.0F, 0.0F);

/// Every preset plus derived themes for awkward accents (white, black,
/// grey, pure primaries) and a sweep around the hue circle.
std::vector<Theme> themesUnderTest() {
    std::vector<Theme> themes;
    for (const ThemeName name : allThemeNames()) {
        themes.push_back(themeFor(name));
    }
    for (const Rgba accent :
         {kWhite, kBlack, rgb(0.5F, 0.5F, 0.5F), rgb(1.0F, 0.0F, 0.0F), rgb(0.0F, 1.0F, 0.0F),
          rgb(0.0F, 0.0F, 1.0F), rgb(1.0F, 1.0F, 0.0F)}) {
        themes.push_back(deriveTheme(accent));
    }
    constexpr int kHueStep = 15;
    for (int hue = 0; hue < 360; hue += kHueStep) {
        const Oklch vivid{.lightness = 0.75F, .chroma = 0.15F, .hue = static_cast<float>(hue)};
        themes.push_back(deriveTheme(fromOklch(vivid)));
    }
    return themes;
}

float closestStates(const Theme& theme) {
    const std::array states{theme.idle, theme.listening, theme.thinking, theme.speaking};
    float closest = 1.0F;
    for (std::size_t i = 0; i < states.size(); ++i) {
        for (std::size_t j = i + 1; j < states.size(); ++j) {
            closest = std::min(closest, perceptualDistance(states.at(i), states.at(j)));
        }
    }
    return closest;
}

bool inUnitRange(Rgba colour) {
    const auto unit = [](float c) { return c >= 0.0F && c <= 1.0F; };
    return unit(colour.r) && unit(colour.g) && unit(colour.b) && unit(colour.a);
}

/// The smallest pairwise distance between the accents of the presets
/// (Custom is left out: it defaults to the cyan accent).
float closestPresetAccents() {
    std::vector<Rgba> accents;
    for (const ThemeName name : allThemeNames()) {
        if (name != ThemeName::Custom) {
            accents.push_back(themeFor(name).accent);
        }
    }
    float closest = 1.0F;
    for (std::size_t i = 0; i < accents.size(); ++i) {
        for (std::size_t j = i + 1; j < accents.size(); ++j) {
            closest = std::min(closest, perceptualDistance(accents.at(i), accents.at(j)));
        }
    }
    return closest;
}

/// Every colour of @p theme lies in [0, 1].
bool allInUnitRange(const Theme& theme) {
    const std::array colours{theme.accent,     theme.idle,      theme.listening, theme.thinking,
                             theme.speaking,   theme.wakeFlash, theme.text,      theme.mutedText,
                             theme.background, theme.plate,     theme.warning};
    return std::ranges::all_of(colours, inUnitRange);
}

/// The lowest contrast of @p role against the plate over every theme under test.
template <class Role>
float worstContrast(Role role) {
    float worst = 21.0F;
    for (const Theme& theme : themesUnderTest()) {
        worst = std::min(worst, contrastRatio(role(theme), theme.plate));
    }
    return worst;
}

}  // namespace

TEST_CASE("every theme is selectable by its settings spelling", "[overlay][palette]") {
    REQUIRE(allThemeNames().size() == 8);
    REQUIRE(std::ranges::all_of(
        allThemeNames(), [](ThemeName name) { return parseThemeName(toString(name)) == name; }));
    REQUIRE(parseThemeName("red") == ThemeName::Red);
    REQUIRE(parseThemeName("custom") == ThemeName::Custom);
}

TEST_CASE("theme names are exact", "[overlay][palette]") {
    REQUIRE_FALSE(parseThemeName("Amber").has_value());
    REQUIRE_FALSE(parseThemeName("purple").has_value());
    REQUIRE_FALSE(parseThemeName("").has_value());
}

TEST_CASE("cyan and amber keep their hand-tuned palettes", "[overlay][palette]") {
    REQUIRE(themeFor(ThemeName::Cyan) == Theme{});
    const Theme amber = themeFor(ThemeName::Amber);
    REQUIRE(amber.accent.r > amber.accent.b);   // warm light
    REQUIRE(amber.plate.a == Theme{}.plate.a);  // same readability either way
}

TEST_CASE("presets are distinct colours and custom follows its accent", "[overlay][palette]") {
    REQUIRE(closestPresetAccents() > kMinPresetDistance);
    const Rgba orange = rgb(1.0F, 0.5F, 0.0F);
    REQUIRE(themeFor(ThemeName::Custom, orange) == deriveTheme(orange));
    REQUIRE(themeFor(ThemeName::Red).accent.r > themeFor(ThemeName::Red).accent.g);
    REQUIRE(themeFor(ThemeName::Green).accent.g > themeFor(ThemeName::Green).accent.r);
}

TEST_CASE("text stays legible on the plate for any accent", "[overlay][palette]") {
    // Against the plate's own colour (contrast ignores alpha): the worst case
    // is a dark desktop behind it, which the plate then matches.
    REQUIRE(worstContrast([](const Theme& t) { return t.text; }) >= kMinTextContrast);
    REQUIRE(worstContrast([](const Theme& t) { return t.mutedText; }) >= kMinSecondaryContrast);
    REQUIRE(worstContrast([](const Theme& t) { return t.warning; }) >= kMinSecondaryContrast);
}

TEST_CASE("the four states stay distinguishable for any accent", "[overlay][palette]") {
    for (const Theme& theme : themesUnderTest()) {
        CAPTURE(toHex(theme.accent));
        REQUIRE(closestStates(theme) >= kMinStateDistance);
        REQUIRE(perceptualDistance(theme.warning, theme.accent) >= kMinWarningDistance);
    }
}

TEST_CASE("derived colours are clamped into a band that glows", "[overlay][palette]") {
    const Oklch white = toOklch(deriveTheme(kWhite).accent);
    const Oklch black = toOklch(deriveTheme(kBlack).accent);
    const Oklch blue = toOklch(deriveTheme(rgb(0.0F, 0.0F, 1.0F)).accent);
    REQUIRE(white.lightness <= 0.86F + kTolerance);
    REQUIRE(black.lightness >= 0.64F - kTolerance);
    REQUIRE(white.chroma >= 0.05F);  // a tint, so states can swing in hue
    REQUIRE(blue.chroma <= 0.19F + kTolerance);
    for (const Theme& theme : themesUnderTest()) {
        CAPTURE(toHex(theme.accent));
        REQUIRE(allInUnitRange(theme));
    }
}

TEST_CASE("pale accents bloom less so white does not blow out", "[overlay][palette]") {
    REQUIRE(deriveTheme(kWhite).bloom < 0.85F);
    REQUIRE(themeFor(ThemeName::Ice).bloom < 0.85F);
    REQUIRE(themeFor(ThemeName::Red).bloom == 1.0F);
    for (const Theme& theme : themesUnderTest()) {
        REQUIRE(theme.bloom >= 0.6F);
        REQUIRE(theme.bloom <= 1.0F);
    }
}

TEST_CASE("OKLCH conversion round-trips sRGB colours", "[overlay][palette]") {
    for (const Rgba colour : {kDefaultAccent, rgb(1.0F, 0.231F, 0.188F), rgb(0.2F, 0.4F, 0.6F)}) {
        CAPTURE(toHex(colour));
        REQUIRE(perceptualDistance(fromOklch(toOklch(colour)), colour) < kTolerance);
    }
    REQUIRE_THAT(toOklch(kWhite).lightness, WithinAbs(1.0F, kTolerance));
    REQUIRE_THAT(toOklch(kWhite).chroma, WithinAbs(0.0F, kTolerance));
}

TEST_CASE("out-of-gamut colours keep lightness and hue", "[overlay][palette]") {
    const Oklch tooVivid{.lightness = 0.9F, .chroma = 0.4F, .hue = 264.0F};
    const Rgba mapped = fromOklch(tooVivid);
    REQUIRE(inUnitRange(mapped));
    REQUIRE_THAT(toOklch(mapped).lightness, WithinAbs(tooVivid.lightness, 0.01F));
    REQUIRE_THAT(toOklch(mapped).hue, WithinAbs(tooVivid.hue, 2.0F));
    REQUIRE_THAT(contrastRatio(kBlack, kWhite), WithinAbs(21.0F, 0.01F));
}

TEST_CASE("accents are read and written as #rrggbb", "[overlay][palette]") {
    const Rgba red = parseHexColour("#ff3b30").value_or(kBlack);
    REQUIRE_THAT(red.r, WithinAbs(1.0F, kTolerance));
    REQUIRE_THAT(red.g, WithinAbs(59.0F / 255.0F, kTolerance));
    REQUIRE_THAT(red.b, WithinAbs(48.0F / 255.0F, kTolerance));
    REQUIRE(parseHexColour("FF3B30") == red);
    REQUIRE(toHex(red) == "#ff3b30");
    REQUIRE(toHex(rgb(2.0F, -1.0F, 0.5F)) == "#ff0080");
}

TEST_CASE("anything but six hex digits is not an accent", "[overlay][palette]") {
    for (const char* bad : {"", "#", "#ff3b3", "#ff3b300", "#gg0000", "ff 3b30", "#-f3b30"}) {
        CAPTURE(std::string{bad});
        REQUIRE_FALSE(parseHexColour(bad).has_value());
    }
}
