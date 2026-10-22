#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/orb.hpp"

#include <imgui.h>

#include "hud.hpp"
#include "hud_draw.hpp"
#include "hud_util.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jarvis::overlay::hud {

namespace {

constexpr std::array<AssistantState, 4> kStates{AssistantState::Idle, AssistantState::Listening,
                                                AssistantState::Thinking, AssistantState::Speaking};

// Panel typography, in 96-DPI pixels before the size preset's text scale.
constexpr float kCaptionSize = 15.0F;
constexpr float kCaptionTracking = 2.6F;
constexpr float kBrandSize = 10.0F;
constexpr float kBrandTracking = 1.8F;
constexpr float kPrefixSize = 10.5F;
constexpr float kPrefixTracking = 1.4F;
constexpr float kBodySize = 14.5F;
constexpr float kPrefixGap = 8.0F;
constexpr float kClockSize = 24.0F;
constexpr float kInfoSize = 11.0F;
constexpr float kInfoTracking = 1.2F;
constexpr float kDividerDrop = 1.75F;
constexpr float kInfoLine = 1.7F;
/// Values of the idle info lines start this far right of the column (96-DPI px).
constexpr float kInfoLabelWidth = 62.0F;
constexpr std::size_t kTelemetryBars = 48;
constexpr float kTelemetryHeight = 30.0F;
constexpr float kRowGap = 0.9F;
/// Transcript rows slide in from the right by this much.
constexpr float kSlide = 24.0F;

// The scrims: soft darkness under the ring and the text, never a box.
constexpr float kRingScrimRadius = 1.55F;
constexpr float kRingScrimAlpha = 0.72F;
constexpr int kFeatherLayers = 7;
constexpr float kFeatherStep = 4.0F;
constexpr float kFeatherAlpha = 0.14F;
constexpr float kScrimRounding = 14.0F;
constexpr float kBracketLength = 14.0F;
constexpr float kBracketInset = 6.0F;

constexpr std::string_view kBrand = "J.A.R.V.I.S";
constexpr std::string_view kWakeCaption = "WAKE WORD";
constexpr std::string_view kYouPrefix = "YOU";
constexpr std::string_view kJarvisPrefix = "JARVIS";
constexpr std::string_view kModelLabel = "MODEL";
constexpr std::string_view kModeLabel = "MODE";
constexpr std::string_view kMicLabel = "MIC";
constexpr std::string_view kUptimeLabel = "UPTIME";
constexpr std::string_view kSeparator = "  \xC2\xB7  ";  // a middle dot between values

/// The value of readout line @p label, or empty.
std::string_view readoutValue(const OverlaySnapshot& snapshot, std::string_view label) {
    if (snapshot.bootReadout == nullptr) {
        return {};
    }
    for (const ReadoutLine& line : *snapshot.bootReadout) {
        if (line.label == label) {
            return line.value;
        }
    }
    return {};
}

float textSize(const HudScene& scene, float logical) noexcept {
    return logical * scene.scale * scene.textScale;
}

float bootAlphaOf(const HudScene& scene) noexcept {
    return scene.frame.boot.has_value() ? scene.frame.boot->handoff : 1.0F;
}

/// The corner panel's own alpha: in during the boot hand-off, out during
/// the first half of the expand morph (the panel's text gives way first)
/// and the first third of the shrink into the orb.
float panelAlphaOf(const HudScene& scene) noexcept {
    return bootAlphaOf(scene) * (1.0F - anim::clamp01(scene.expand * 2.0F)) *
           orbMorphAt(scene.orb).panelAlpha;
}

/// The y of the divider under the caption.
float dividerY(const HudScene& scene) noexcept {
    return scene.panelText.y + (textSize(scene, kCaptionSize) * kDividerDrop);
}

}  // namespace

// --- scrims and frame ------------------------------------------------------------------

void HudRenderer::drawScrim(ImDrawList& draw, const HudScene& scene) const {
    const float ringAlpha = bootAlphaOf(scene) * scene.opacity;
    if (ringAlpha <= 0.0F) {
        return;
    }
    const Rgba plate{.r = theme_.plate.r, .g = theme_.plate.g, .b = theme_.plate.b};
    // The ring keeps its dark disc while it flies into the expanded header.
    radialGlow(draw, centreOf(scene), scene.geometry.radius * kRingScrimRadius,
               withAlpha(plate, kRingScrimAlpha * ringAlpha));
    const float alpha = panelAlphaOf(scene) * scene.opacity;
    if (alpha <= 0.0F) {
        return;
    }
    // Nested rounded rectangles of low alpha add up to a dark core with a
    // feathered edge: readable over any wallpaper, but no hard panel border.
    const RectF& column = scene.panelText;
    const float step = kFeatherStep * scene.scale;
    const ImU32 layer = toU32(withAlpha(plate, kFeatherAlpha * alpha));
    for (int i = 0; i < kFeatherLayers; ++i) {
        const float grow = step * static_cast<float>(kFeatherLayers - i);
        draw.AddRectFilled({column.x - grow, column.y - grow},
                           {column.right() + grow, column.bottom() + grow}, layer,
                           (kScrimRounding * scene.scale) + grow);
    }
}

void HudRenderer::drawConnector(ImDrawList& draw, const HudScene& scene) const {
    const float alpha = panelAlphaOf(scene) * scene.opacity * intensity(scene);
    if (alpha <= 0.0F) {
        return;
    }
    const Rgba light = accent(scene);
    const RectF& column = scene.panelText;
    const float y = dividerY(scene);
    // From the ring's right edge along the divider: the text reads as part
    // of the instrument, not as a label next to it.
    const ImVec2 start{scene.geometry.centreX + (scene.geometry.radius * 1.08F),
                       scene.geometry.centreY};
    const ImVec2 elbow{column.x - (kPrefixGap * scene.scale), y};
    glowLine(draw, start, elbow,
             Stroke{.colour = withAlpha(light, 0.45F * alpha),
                    .thickness = scene.scale,
                    .glow = kSoftGlow});
    glowDot(draw, start, 2.0F * scene.scale, withAlpha(light, alpha));
    const float inset = kBracketInset * scene.scale;
    const RectF frame{.x = column.x - inset,
                      .y = column.y - inset,
                      .width = column.width + (2.0F * inset),
                      .height = column.height + (2.0F * inset)};
    cornerBrackets(draw, frame, kBracketLength * scene.scale * scene.textScale,
                   withAlpha(light, 0.8F * alpha), 1.4F * scene.scale);
}

// --- panel text ------------------------------------------------------------------------

void HudRenderer::drawPanelText(ImDrawList& draw, const HudScene& scene,
                                const HudFonts& fonts) const {
    const float alpha = panelAlphaOf(scene) * scene.opacity;
    if (alpha <= 0.0F) {
        return;
    }
    drawCaption(draw, scene, *fonts.display, alpha);
    drawDivider(draw, scene, dividerY(scene), alpha);
    if (scene.confirming) {
        return;  // the quit confirmation is drawn in these rows instead
    }
    const float transcript = anim::clamp01(scene.frame.transcript);
    drawIdleReadout(draw, scene, fonts, alpha * (1.0F - transcript));
    drawTranscript(draw, scene, fonts, alpha * transcript);
}

void HudRenderer::drawIdleReadout(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts,
                                  float alpha) const {
    if (alpha <= kNegligible) {
        return;
    }
    const RectF& column = scene.panelText;
    const float clockSize = textSize(scene, kClockSize);
    const float infoSize = textSize(scene, kInfoSize);
    const float tracking = kInfoTracking * scene.scale * scene.textScale;
    const float top = dividerY(scene) + (infoSize * kRowGap);
    const float clockWidth = spacedText(draw, *fonts.display, clockSize, {column.x, top},
                                        scene.clock.time, tracking, withAlpha(theme_.text, alpha));
    spacedText(draw, *fonts.display, infoSize,
               {column.x + clockWidth + (infoSize * 1.2F), top + ((clockSize - infoSize) * 0.6F)},
               scene.clock.date, tracking, withAlpha(accent(scene), alpha));

    std::string model{readoutValue(scene.snapshot, kModelLabel)};
    const std::string_view mode = readoutValue(scene.snapshot, kModeLabel);
    if (!model.empty() && !mode.empty()) {
        model.append(kSeparator);
    }
    model.append(mode);
    const std::array<std::pair<std::string_view, std::string>, 3> lines{{
        {kModelLabel, model},
        {kMicLabel, std::string{readoutValue(scene.snapshot, kMicLabel)}},
        {kUptimeLabel, formatUptime(scene.frame.time)},
    }};
    float y = top + (clockSize * 1.35F);
    for (const auto& [label, value] : lines) {
        if (!value.empty()) {
            drawInfoLine(draw, scene, fonts, {column.x, y}, label, value, alpha);
            y += infoSize * kInfoLine;
        }
    }
}

void HudRenderer::drawInfoLine(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts,
                               ImVec2 at, std::string_view label, std::string_view value,
                               float alpha) const {
    const RectF& column = scene.panelText;
    const float size = textSize(scene, kInfoSize);
    const float tracking = kInfoTracking * scene.scale * scene.textScale;
    spacedText(draw, *fonts.display, size, at, label, tracking, withAlpha(theme_.mutedText, alpha));
    const float x = column.x + (kInfoLabelWidth * scene.scale * scene.textScale);
    const float charWidth = fonts.body->CalcTextSizeA(size, FLT_MAX, 0.0F, "n").x;
    text(draw, *fonts.body, size, {x, at.y - (size * 0.1F)},
         truncateUtf8Head(value, charactersPerLine(column.right() - x, charWidth)),
         withAlpha(theme_.text, alpha));
}

void HudRenderer::drawTelemetry(ImDrawList& draw, const HudScene& scene) const {
    const float alpha = panelAlphaOf(scene) * scene.opacity * intensity(scene);
    if (alpha <= kNegligible || scene.inputOpen) {
        return;  // the text field sits where the strip is
    }
    // A strip of level bars along the bottom of the column: the live audio
    // while listening or speaking, a slow shimmer while idle.
    const RectF& column = scene.panelText;
    const std::vector<float> bars = waveformBars(scene.snapshot.levels, kTelemetryBars, 0.0F);
    const float height = textSize(scene, kTelemetryHeight);
    const float slot = column.width / static_cast<float>(kTelemetryBars);
    const float base = column.bottom();
    const Rgba light = accent(scene);
    std::size_t i = 0;
    for (const float level : bars) {
        const float phase = (scene.frame.time * 0.9F) + (static_cast<float>(i) * 0.35F);
        const float idle = 0.12F + (0.10F * (0.5F + (0.5F * std::sin(phase))));
        const float value = std::max(idle, anim::perceptualLevel(level));
        const float x = column.x + (slot * static_cast<float>(i++));
        draw.AddRectFilled({x, base - (height * value)}, {x + (slot * 0.6F), base},
                           toU32(withAlpha(light, (0.35F + (0.5F * value)) * alpha)));
    }
}

void HudRenderer::drawTranscript(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts,
                                 float alpha) const {
    if (alpha <= kNegligible) {
        return;
    }
    const RectF& column = scene.panelText;
    const float slide = (1.0F - anim::clamp01(scene.frame.transcript)) * kSlide * scene.scale;
    const float prefixSize = textSize(scene, kPrefixSize);
    const float prefixTracking = kPrefixTracking * scene.scale * scene.textScale;
    const float bodySize = textSize(scene, kBodySize);
    const float left = column.x + slide;
    const float textX = left +
                        spacedTextWidth(*fonts.display, prefixSize, kJarvisPrefix, prefixTracking) +
                        (kPrefixGap * scene.scale);
    const float charWidth = fonts.body->CalcTextSizeA(bodySize, FLT_MAX, 0.0F, "n").x;
    const std::size_t fit = charactersPerLine(column.right() - textX, charWidth);
    const float prefixDrop = (bodySize - prefixSize) * 0.5F;
    const float top = dividerY(scene) + (bodySize * kRowGap);
    const std::array<std::pair<float, bool>, 2> rows{{{0.0F, false}, {1.0F, true}}};
    for (const auto& [row, isReply] : rows) {
        const std::string& line = isReply ? scene.snapshot.response : scene.snapshot.transcript;
        if (line.empty()) {
            continue;  // an empty row is not drawn at all, not even its label
        }
        const float y = top + (row * bodySize * 1.9F);
        const Rgba prefixColour = isReply ? accent(scene) : theme_.mutedText;
        spacedText(draw, *fonts.display, prefixSize, {left, y + prefixDrop},
                   isReply ? kJarvisPrefix : kYouPrefix, prefixTracking,
                   withAlpha(prefixColour, alpha));
        text(draw, *fonts.body, bodySize, {textX, y}, truncateUtf8Tail(line, fit),
             withAlpha(theme_.text, alpha));
    }
}

void HudRenderer::drawCaption(ImDrawList& draw, const HudScene& scene, ImFont& font,
                              float alpha) const {
    const RectF& column = scene.panelText;
    const float size = textSize(scene, kCaptionSize);
    const float tracking = kCaptionTracking * scene.scale * scene.textScale;
    // Labels cross-fade with the state, like everything else.
    for (const AssistantState state : kStates) {
        const float weight = weightOf(scene, state);
        if (weight >= kNegligible) {
            spacedText(draw, font, size, {column.x, column.y}, hudLabel(state), tracking,
                       withAlpha(mixColour(stateColour(theme_, state), theme_.text, 0.35F),
                                 weight * alpha));
        }
    }
    const float wake = scene.frame.ripple.alpha;
    const float brandSize = textSize(scene, kBrandSize);
    const float brandTracking = kBrandTracking * scene.scale * scene.textScale;
    const float baseline = column.y + ((size - brandSize) * 0.5F);
    // The brand gives way to the hover controls drawn on the same line.
    const float brand = (1.0F - wake) * (1.0F - anim::clamp01(scene.hover));
    const std::array<std::pair<std::string_view, Rgba>, 2> marks{{
        {kBrand, withAlpha(accent(scene), 0.8F * brand * alpha)},
        {kWakeCaption, withAlpha(theme_.wakeFlash, wake * alpha)},
    }};
    for (const auto& [mark, colour] : marks) {
        const float width = spacedTextWidth(font, brandSize, mark, brandTracking);
        spacedText(draw, font, brandSize, {column.right() - width, baseline}, mark, brandTracking,
                   colour);
    }
}

void HudRenderer::drawDivider(ImDrawList& draw, const HudScene& scene, float y, float alpha) const {
    const RectF& column = scene.panelText;
    const Rgba light = accent(scene);
    draw.AddLine({column.x, y}, {column.right(), y}, toU32(withAlpha(light, 0.35F * alpha)),
                 scene.scale);
    // A bright runner travels along the divider, slowly when idle.
    const float busy = 1.0F - weightOf(scene, AssistantState::Idle);
    const float speed = 0.12F + (0.43F * busy);
    const float length = column.width * 0.22F;
    const float head = column.x + (fraction(scene.frame.time * speed) * (column.width + length));
    const float from = std::max(head - length, column.x);
    const float to = std::min(head, column.right());
    if (to > from) {
        glowLine(draw, {from, y}, {to, y},
                 Stroke{.colour = withAlpha(light, (0.35F + (0.5F * busy)) * alpha),
                        .thickness = 1.4F * scene.scale,
                        .glow = kSoftGlow});
    }
}

}  // namespace jarvis::overlay::hud

// Design notes:
// - No panel rectangle: a radial scrim under the ring and a feathered scrim
//   under the text darken whatever is behind just enough for the text to
//   read, and the light (brackets, connector, ring) defines the shape. Over a
//   bright window the HUD still reads as a hologram, not as a grey box.
// - The rows below the caption are either the idle readout (clock, date,
//   model and mode from the boot readout) or the transcript, cross-faded by
//   AnimationFrame::transcript; an empty transcript row is not drawn at all.
// - All panel typography scales with the size preset (PanelMetrics), so
//   "compact" keeps the ADR-0011 strip and "large" is a showpiece.
