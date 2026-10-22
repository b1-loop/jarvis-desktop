/// @file palette.hpp
/// @brief The HUD's named themes and the palette derived from one accent colour.
///
/// Every colour of a Theme (state colours, text, glass plate, warning, bloom
/// gain) follows from a single accent so a user can pick any colour and still
/// get a legible, coherent HUD. The derivation works in OKLCH (Björn
/// Ottosson's OKLab in polar form): lightness and chroma there track what the
/// eye sees, so "a bit brighter" or "the same colour, less saturated" means
/// the same thing for red as for cyan, which HSL does not. Everything here is
/// pure arithmetic and unit-tested without a window.

#pragma once

#include "jarvis/overlay/layout.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace jarvis::overlay {

/// A colour in OKLCH: perceptual lightness [0, 1], chroma (0 is grey, about
/// 0.37 the most saturated sRGB colour) and hue angle in degrees [0, 360).
struct Oklch {
    float lightness{0.0F};
    float chroma{0.0F};
    float hue{0.0F};

    [[nodiscard]] bool operator==(const Oklch&) const noexcept = default;
};

/// @p colour (sRGB-encoded, alpha ignored) in OKLCH. Greys get hue 0.
[[nodiscard]] Oklch toOklch(Rgba colour) noexcept;

/// The sRGB colour of @p colour with alpha @p alpha. Colours outside the sRGB
/// gamut keep their lightness and hue and lose chroma until they fit, so a
/// bright saturated request becomes the most saturated colour the screen can
/// show rather than a clipped, hue-shifted one.
[[nodiscard]] Rgba fromOklch(Oklch colour, float alpha = 1.0F) noexcept;

/// Euclidean distance in OKLab (delta E OK): about 0.02 is a just noticeable
/// difference, 0.1 an obvious one.
[[nodiscard]] float perceptualDistance(Rgba first, Rgba second) noexcept;

/// WCAG 2 contrast ratio of two opaque colours, from 1 (same) to 21.
[[nodiscard]] float contrastRatio(Rgba first, Rgba second) noexcept;

/// The full HUD palette for @p accent.
///
/// The accent's lightness and chroma are clamped to a band that glows on a
/// dark desktop without blowing out (near-white and near-black inputs get a
/// faint ice tint, so the states stay distinguishable), then every role is
/// placed relative to it: states by lightness and a hue swing, text and
/// plate at fixed lightness for contrast, a warning hue far from the accent,
/// and a bloom gain that drops for pale colours.
[[nodiscard]] Theme deriveTheme(Rgba accent) noexcept;

/// "#rrggbb" or "rrggbb" (any case) as an opaque colour; std::nullopt otherwise.
[[nodiscard]] std::optional<Rgba> parseHexColour(std::string_view text) noexcept;

/// @p colour as lower-case "#rrggbb" (alpha dropped, channels clamped).
[[nodiscard]] std::string toHex(Rgba colour);

/// Named palettes selectable from settings; Custom is derived from an accent.
enum class ThemeName : std::uint8_t { Cyan, Amber, Red, Green, Violet, Ice, Pink, Custom };

/// The accent of the original cyan theme; what Custom uses until one is picked.
inline constexpr Rgba kDefaultAccent = Theme{}.accent;

/// The palette for @p name. Cyan is Theme{} and Amber its hand-tuned warm
/// twin (both kept as they were); the other presets are deriveTheme() of a
/// fixed seed, and Custom is deriveTheme(@p customAccent).
[[nodiscard]] Theme themeFor(ThemeName name, Rgba customAccent = kDefaultAccent) noexcept;

/// Every ThemeName, in the order the settings page lists them.
[[nodiscard]] std::span<const ThemeName> allThemeNames() noexcept;

/// Lower-case name as used in settings ("cyan", "red", "custom").
[[nodiscard]] std::string_view toString(ThemeName name) noexcept;

/// Inverse of toString(ThemeName); std::nullopt for anything else.
[[nodiscard]] std::optional<ThemeName> parseThemeName(std::string_view name) noexcept;

}  // namespace jarvis::overlay
