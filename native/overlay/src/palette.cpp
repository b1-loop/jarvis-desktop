#include "jarvis/overlay/palette.hpp"

#include "jarvis/overlay/layout.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <format>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace jarvis::overlay {

namespace {

// --- sRGB <-> OKLab (Ottosson's published matrices) --------------------------

struct Vec3 {
    float x{0.0F};
    float y{0.0F};
    float z{0.0F};
};

using Matrix = std::array<Vec3, 3>;

constexpr Matrix kLinearToLms{{{.x = 0.4122214708F, .y = 0.5363325363F, .z = 0.0514459929F},
                               {.x = 0.2119034982F, .y = 0.6806995451F, .z = 0.1073969566F},
                               {.x = 0.0883024619F, .y = 0.2817188376F, .z = 0.6299787005F}}};
constexpr Matrix kLmsToLab{{{.x = 0.2104542553F, .y = 0.7936177850F, .z = -0.0040720468F},
                            {.x = 1.9779984951F, .y = -2.4285922050F, .z = 0.4505937099F},
                            {.x = 0.0259040371F, .y = 0.7827717662F, .z = -0.8086757660F}}};
constexpr Matrix kLabToLms{{{.x = 1.0F, .y = 0.3963377774F, .z = 0.2158037573F},
                            {.x = 1.0F, .y = -0.1055613458F, .z = -0.0638541728F},
                            {.x = 1.0F, .y = -0.0894841775F, .z = -1.2914855480F}}};
constexpr Matrix kLmsToLinear{{{.x = 4.0767416621F, .y = -3.3077115913F, .z = 0.2309699292F},
                               {.x = -1.2684380046F, .y = 2.6097574011F, .z = -0.3413193965F},
                               {.x = -0.0041960863F, .y = -0.7034186147F, .z = 1.7076373010F}}};

// sRGB transfer function (IEC 61966-2-1).
constexpr float kSrgbLinearLimit = 0.04045F;
constexpr float kLinearSrgbLimit = 0.0031308F;
constexpr float kSrgbSlope = 12.92F;
constexpr float kSrgbOffset = 0.055F;
constexpr float kSrgbScale = 1.055F;
constexpr float kSrgbGamma = 2.4F;
// WCAG relative luminance weights and flare term.
constexpr Vec3 kLuminance{.x = 0.2126F, .y = 0.7152F, .z = 0.0722F};
constexpr float kContrastFlare = 0.05F;

constexpr float kFullTurn = 360.0F;
constexpr float kGamutTolerance = 1e-4F;
constexpr int kGamutSearchSteps = 20;

Vec3 multiply(const Matrix& m, Vec3 v) noexcept {
    const auto row = [&](const Vec3& r) { return (r.x * v.x) + (r.y * v.y) + (r.z * v.z); };
    return Vec3{.x = row(m.at(0)), .y = row(m.at(1)), .z = row(m.at(2))};
}

float toLinear(float channel) noexcept {
    if (channel <= kSrgbLinearLimit) {
        return channel / kSrgbSlope;
    }
    return std::pow((channel + kSrgbOffset) / kSrgbScale, kSrgbGamma);
}

float toSrgb(float linear) noexcept {
    if (linear <= kLinearSrgbLimit) {
        return linear * kSrgbSlope;
    }
    return (kSrgbScale * std::pow(linear, 1.0F / kSrgbGamma)) - kSrgbOffset;
}

Vec3 linearOf(Rgba colour) noexcept {
    return Vec3{.x = toLinear(colour.r), .y = toLinear(colour.g), .z = toLinear(colour.b)};
}

Vec3 labOf(Rgba colour) noexcept {
    const Vec3 lms = multiply(kLinearToLms, linearOf(colour));
    return multiply(kLmsToLab,
                    Vec3{.x = std::cbrt(lms.x), .y = std::cbrt(lms.y), .z = std::cbrt(lms.z)});
}

float radians(float degrees) noexcept {
    return degrees * std::numbers::pi_v<float> / (kFullTurn / 2.0F);
}

Vec3 linearOf(Oklch colour) noexcept {
    const float hue = radians(colour.hue);
    const Vec3 lab{.x = colour.lightness,
                   .y = colour.chroma * std::cos(hue),
                   .z = colour.chroma * std::sin(hue)};
    const Vec3 root = multiply(kLabToLms, lab);
    return multiply(kLmsToLinear, Vec3{.x = root.x * root.x * root.x,
                                       .y = root.y * root.y * root.y,
                                       .z = root.z * root.z * root.z});
}

bool inGamut(Vec3 linear) noexcept {
    const auto fits = [](float c) { return c >= -kGamutTolerance && c <= 1.0F + kGamutTolerance; };
    return fits(linear.x) && fits(linear.y) && fits(linear.z);
}

/// The largest chroma at which @p colour's lightness and hue fit in sRGB.
float gamutChroma(Oklch colour) noexcept {
    float low = 0.0F;
    float high = colour.chroma;
    for (int step = 0; step < kGamutSearchSteps; ++step) {
        const float middle = (low + high) / 2.0F;
        colour.chroma = middle;
        if (inGamut(linearOf(colour))) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
}

float relativeLuminance(Rgba colour) noexcept {
    const Vec3 linear = linearOf(colour);
    return (kLuminance.x * linear.x) + (kLuminance.y * linear.y) + (kLuminance.z * linear.z);
}

// --- the derivation -----------------------------------------------------------

// Band the accent is clamped to: bright enough to glow on a dark desktop,
// dark enough that the lighter state colours still have room above it.
constexpr float kMinAccentLightness = 0.64F;
constexpr float kMaxAccentLightness = 0.86F;
// A floor on chroma keeps the hue swing between states visible for greys.
constexpr float kMinAccentChroma = 0.07F;
constexpr float kMaxAccentChroma = 0.19F;
// Below this an input has no meaningful hue; it gets a cool "ice" tint.
constexpr float kGreyChroma = 0.02F;
constexpr float kGreyHue = 230.0F;

// State colours relative to the accent: lightness steps and a hue swing.
constexpr float kIdleLightness = -0.13F;
constexpr float kIdleChroma = 0.8F;
constexpr float kListeningLightness = 0.08F;
constexpr float kThinkingLightness = -0.03F;
constexpr float kSpeakingLightness = 0.04F;
constexpr float kStateHueSwing = 30.0F;

// Fixed-lightness roles; their chroma is a share of the accent's, capped,
// so text stays nearly neutral and the plate a dark tint.
struct Role {
    float lightness;
    float chromaShare;
    float maxChroma;
};
constexpr Role kWakeFlash{.lightness = 0.97F, .chromaShare = 1.0F, .maxChroma = 0.05F};
constexpr Role kText{.lightness = 0.95F, .chromaShare = 0.25F, .maxChroma = 0.035F};
constexpr Role kMutedText{.lightness = 0.76F, .chromaShare = 0.45F, .maxChroma = 1.0F};
constexpr Role kBackground{.lightness = 0.14F, .chromaShare = 0.25F, .maxChroma = 0.03F};
constexpr Role kPlate{.lightness = 0.17F, .chromaShare = 0.3F, .maxChroma = 0.035F};
constexpr float kPlateAlpha = 0.55F;

// Warning: red-orange, the conventional alert colour, unless the accent is
// itself within kWarningClearance of it (red, orange, pink); then yellow.
constexpr float kWarningLightness = 0.70F;
constexpr float kWarningChroma = 0.19F;
constexpr float kRedWarningHue = 28.0F;
constexpr float kYellowWarningHue = 95.0F;
constexpr float kWarningClearance = 45.0F;

// Bloom gain rises from kPaleBloom (greys) to 1 at kFullBloomChroma.
constexpr float kPaleBloom = 0.6F;
constexpr float kFullBloomChroma = 0.14F;

float wrapHue(float hue) noexcept {
    const float wrapped = std::fmod(hue, kFullTurn);
    return wrapped < 0.0F ? wrapped + kFullTurn : wrapped;
}

/// Angle between two hues, 0 to 180 degrees.
float hueDistance(float first, float second) noexcept {
    return std::abs(std::remainder(first - second, kFullTurn));
}

Oklch clampedAccent(Rgba accent) noexcept {
    Oklch base = toOklch(accent);
    if (base.chroma < kGreyChroma) {
        base.hue = kGreyHue;
    }
    base.lightness = std::clamp(base.lightness, kMinAccentLightness, kMaxAccentLightness);
    base.chroma = std::clamp(base.chroma, kMinAccentChroma, kMaxAccentChroma);
    return base;
}

Rgba shifted(Oklch base, float lightness, float chromaFactor, float hueSwing) noexcept {
    return fromOklch(Oklch{.lightness = base.lightness + lightness,
                           .chroma = base.chroma * chromaFactor,
                           .hue = base.hue + hueSwing});
}

Rgba role(Oklch base, Role spec, float alpha = 1.0F) noexcept {
    const float chroma = std::min(base.chroma * spec.chromaShare, spec.maxChroma);
    return fromOklch(Oklch{.lightness = spec.lightness, .chroma = chroma, .hue = base.hue}, alpha);
}

Rgba warningFor(Oklch base) noexcept {
    const bool nearRed = hueDistance(base.hue, kRedWarningHue) < kWarningClearance;
    const float hue = nearRed ? kYellowWarningHue : kRedWarningHue;
    return fromOklch(Oklch{.lightness = kWarningLightness, .chroma = kWarningChroma, .hue = hue});
}

float bloomGain(Oklch accent) noexcept {
    return std::clamp(kPaleBloom + ((1.0F - kPaleBloom) * accent.chroma / kFullBloomChroma),
                      kPaleBloom, 1.0F);
}

// --- named themes ---------------------------------------------------------------

constexpr Rgba rgba(float r, float g, float b, float a = 1.0F) noexcept {
    return Rgba{.r = r, .g = g, .b = b, .a = a};
}

/// The cyan HUD in warm light: same structure, amber hues (hand-tuned).
Theme amberTheme() noexcept {
    Theme theme;
    theme.background = rgba(0.08F, 0.04F, 0.01F);
    theme.text = rgba(1.00F, 0.95F, 0.85F);
    theme.mutedText = rgba(0.85F, 0.65F, 0.40F);
    theme.idle = rgba(0.80F, 0.50F, 0.18F);
    theme.listening = rgba(1.00F, 0.78F, 0.30F);
    theme.thinking = rgba(1.00F, 0.62F, 0.35F);
    theme.speaking = rgba(1.00F, 0.88F, 0.45F);
    theme.wakeFlash = rgba(1.00F, 0.97F, 0.85F);
    theme.accent = rgba(1.00F, 0.68F, 0.22F);
    theme.plate = rgba(0.10F, 0.05F, 0.01F, 0.55F);
    return theme;
}

struct NamedTheme {
    ThemeName name;
    std::string_view spelling;
    /// Seed accent of a derived preset (unused for Cyan, Amber and Custom).
    Rgba seed;
};

// Seeds are what each preset is named after: Iron Man red (#ff3b30), a
// phosphor green, violet, ice white and a hot pink.
constexpr std::array<NamedTheme, 8> kThemes{{
    {.name = ThemeName::Cyan, .spelling = "cyan", .seed = kDefaultAccent},
    {.name = ThemeName::Amber, .spelling = "amber", .seed = kDefaultAccent},
    {.name = ThemeName::Red, .spelling = "red", .seed = rgba(1.00F, 0.231F, 0.188F)},
    {.name = ThemeName::Green, .spelling = "green", .seed = rgba(0.251F, 1.00F, 0.549F)},
    {.name = ThemeName::Violet, .spelling = "violet", .seed = rgba(0.678F, 0.451F, 1.00F)},
    {.name = ThemeName::Ice, .spelling = "ice", .seed = rgba(0.851F, 0.929F, 1.00F)},
    {.name = ThemeName::Pink, .spelling = "pink", .seed = rgba(1.00F, 0.40F, 0.749F)},
    {.name = ThemeName::Custom, .spelling = "custom", .seed = kDefaultAccent},
}};

constexpr std::array<ThemeName, kThemes.size()> kThemeOrder = [] {
    std::array<ThemeName, kThemes.size()> order{};
    std::ranges::transform(kThemes, order.begin(), &NamedTheme::name);
    return order;
}();

const NamedTheme& entryFor(ThemeName name) noexcept {
    const auto found = std::ranges::find(kThemes, name, &NamedTheme::name);
    return found != kThemes.end() ? *found : kThemes.front();
}

constexpr int kHexDigits = 6;
constexpr int kHexBase = 16;
constexpr float kChannelMax = 255.0F;

std::optional<float> hexChannel(std::string_view pair) noexcept {
    unsigned value = 0;
    const auto [end, error] =
        std::from_chars(pair.data(), pair.data() + pair.size(), value, kHexBase);
    if (error != std::errc{} || end != pair.data() + pair.size()) {
        return std::nullopt;
    }
    return static_cast<float>(value) / kChannelMax;
}

unsigned byteOf(float channel) noexcept {
    return static_cast<unsigned>(std::lround(std::clamp(channel, 0.0F, 1.0F) * kChannelMax));
}

}  // namespace

Oklch toOklch(Rgba colour) noexcept {
    const Vec3 lab = labOf(colour);
    const float chroma = std::hypot(lab.y, lab.z);
    const float hue =
        chroma > 0.0F ? std::atan2(lab.z, lab.y) * (kFullTurn / 2.0F) / std::numbers::pi_v<float>
                      : 0.0F;
    return Oklch{.lightness = lab.x, .chroma = chroma, .hue = wrapHue(hue)};
}

Rgba fromOklch(Oklch colour, float alpha) noexcept {
    colour.lightness = std::clamp(colour.lightness, 0.0F, 1.0F);
    colour.chroma = std::max(colour.chroma, 0.0F);
    colour.hue = wrapHue(colour.hue);
    if (!inGamut(linearOf(colour))) {
        colour.chroma = gamutChroma(colour);
    }
    const Vec3 linear = linearOf(colour);
    const auto encode = [](float c) {
        return std::clamp(toSrgb(std::clamp(c, 0.0F, 1.0F)), 0.0F, 1.0F);
    };
    return Rgba{.r = encode(linear.x), .g = encode(linear.y), .b = encode(linear.z), .a = alpha};
}

float perceptualDistance(Rgba first, Rgba second) noexcept {
    const Vec3 a = labOf(first);
    const Vec3 b = labOf(second);
    return std::sqrt(((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y)) +
                     ((a.z - b.z) * (a.z - b.z)));
}

float contrastRatio(Rgba first, Rgba second) noexcept {
    const float a = relativeLuminance(first);
    const float b = relativeLuminance(second);
    return (std::max(a, b) + kContrastFlare) / (std::min(a, b) + kContrastFlare);
}

Theme deriveTheme(Rgba accent) noexcept {
    const Oklch base = clampedAccent(accent);
    Theme theme;
    theme.accent = fromOklch(base);
    theme.idle = shifted(base, kIdleLightness, kIdleChroma, 0.0F);
    theme.listening = shifted(base, kListeningLightness, 1.0F, 0.0F);
    theme.thinking = shifted(base, kThinkingLightness, 1.0F, kStateHueSwing);
    theme.speaking = shifted(base, kSpeakingLightness, 1.0F, -kStateHueSwing);
    theme.wakeFlash = role(base, kWakeFlash);
    theme.text = role(base, kText);
    theme.mutedText = role(base, kMutedText);
    theme.background = role(base, kBackground);
    theme.plate = role(base, kPlate, kPlateAlpha);
    theme.warning = warningFor(base);
    theme.bloom = bloomGain(base);
    return theme;
}

std::optional<Rgba> parseHexColour(std::string_view text) noexcept {
    if (text.starts_with('#')) {
        text.remove_prefix(1);
    }
    if (text.size() != kHexDigits) {
        return std::nullopt;
    }
    const auto red = hexChannel(text.substr(0, 2));
    const auto green = hexChannel(text.substr(2, 2));
    const auto blue = hexChannel(text.substr(4, 2));
    if (!red || !green || !blue) {
        return std::nullopt;
    }
    return Rgba{.r = *red, .g = *green, .b = *blue, .a = 1.0F};
}

std::string toHex(Rgba colour) {
    return std::format("#{:02x}{:02x}{:02x}", byteOf(colour.r), byteOf(colour.g), byteOf(colour.b));
}

Theme themeFor(ThemeName name, Rgba customAccent) noexcept {
    switch (name) {
        case ThemeName::Cyan:
            return Theme{};
        case ThemeName::Amber:
            return amberTheme();
        case ThemeName::Custom:
            return deriveTheme(customAccent);
        case ThemeName::Red:
        case ThemeName::Green:
        case ThemeName::Violet:
        case ThemeName::Ice:
        case ThemeName::Pink:
            break;
    }
    return deriveTheme(entryFor(name).seed);
}

std::span<const ThemeName> allThemeNames() noexcept {
    return kThemeOrder;
}

std::string_view toString(ThemeName name) noexcept {
    return entryFor(name).spelling;
}

std::optional<ThemeName> parseThemeName(std::string_view name) noexcept {
    const auto found = std::ranges::find(kThemes, name, &NamedTheme::spelling);
    if (found == kThemes.end()) {
        return std::nullopt;
    }
    return found->name;
}

}  // namespace jarvis::overlay

// Design notes:
// - OKLCH rather than HSL: HSL "lightness 60 %" is a pale yellow but a deep
//   blue, so a palette built by HSL offsets looks right for one hue and
//   wrong for the next. In OKLCH the same offsets give the same perceived
//   steps for every hue, which is what lets a single tested rule serve any
//   colour a user can pick.
// - Out-of-gamut colours are mapped by reducing chroma (binary search) at
//   constant lightness and hue, the CSS Color 4 approach without its
//   delta-E refinement; clipping each channel instead would shift the hue
//   (a bright blue clipped per channel turns purple).
// - Cyan and Amber are kept hand-tuned and bit-identical to the original
//   palettes so nobody's HUD changes on upgrade; both satisfy the same
//   contrast and distinctness invariants the derived themes are tested
//   against (palette_test.cpp).
// - The bloom gain is driven by chroma, not luminance: what blows out under
//   bloom is light present in all three channels (pale and white accents),
//   and chroma is exactly the measure of how far from that a colour is.
