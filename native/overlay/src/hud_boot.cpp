#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"

#include <imgui.h>

#include "hud.hpp"
#include "hud_draw.hpp"
#include "hud_util.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace jarvis::overlay::hud {

namespace {

using anim::kTau;

constexpr float kFlashGlowRadius = 2.4F;
constexpr float kShockThickness = 3.0F;
constexpr float kShockEcho = 0.82F;
constexpr float kSweepRadius = 1.12F;
constexpr float kSweepSpan = 1.1F;
constexpr float kSweepFadeIn = 0.4F;

constexpr std::string_view kHexDigits = "0123456789ABCDEF";
constexpr std::size_t kGlyphRuns = 3;
constexpr std::size_t kGlyphsPerRun = 10;
constexpr float kGlyphRadius = 1.36F;
constexpr float kGlyphSpacing = 0.075F;
constexpr float kGlyphSize = 0.07F;
/// Glyphs re-roll this many times per second.
constexpr float kGlyphRate = 12.0F;
constexpr float kGlyphFadeIn = 0.4F;
constexpr float kGlyphDelay = 0.3F;

constexpr float kTitleSize = 0.072F;
constexpr float kTitleTracking = 0.42F;
constexpr float kTitleShake = 0.35F;
constexpr float kSubtitleSize = 0.03F;
constexpr float kSubtitleTracking = 0.75F;
constexpr float kSubtitleDrop = 1.55F;
constexpr std::string_view kSubtitle = "ALL SYSTEMS NOMINAL";

constexpr float kReadoutSize = 0.026F;
constexpr float kReadoutLine = 1.45F;
constexpr float kReadoutDrop = 2.4F;
constexpr float kReadoutGap = 1.2F;
constexpr std::size_t kMaxReadout = 8;
constexpr std::string_view kReadoutOk = "OK";

/// The readout lines to show: Python's first, then the renderer's own.
std::size_t collectReadout(const HudScene& scene,
                           std::array<const ReadoutLine*, kMaxReadout>& lines) noexcept {
    std::size_t count = 0;
    const auto add = [&lines, &count](const ReadoutLine& line) {
        if (count < lines.size()) {
            lines.at(count++) = &line;
        }
    };
    if (scene.snapshot.bootReadout != nullptr) {
        for (const ReadoutLine& line : *scene.snapshot.bootReadout) {
            add(line);
        }
    }
    for (const ReadoutLine& line : scene.systemReadout) {
        add(line);
    }
    return count;
}

float textWidth(ImFont& font, float size, std::string_view utf8) {
    return font.CalcTextSizeA(size, FLT_MAX, 0.0F, utf8.data(), utf8.data() + utf8.size()).x;
}

}  // namespace

void HudRenderer::drawIgnition(ImDrawList& draw, const HudScene& scene,
                               const anim::BootFrame& boot) const {
    const ImVec2 centre = centreOf(scene);
    const float radius = scene.geometry.radius;
    const float fade = boot.bootOnly * scene.opacity;
    const Rgba light = accent(scene);
    // The point of light and its flash: a white-hot bloom that fills the canvas for a beat.
    radialGlow(draw, centre, radius * kFlashGlowRadius * std::sqrt(boot.flash),
               withAlpha(mixColour(light, theme_.wakeFlash, 0.6F), 0.55F * boot.flash * fade));
    if (boot.shockAlpha > 0.0F) {
        const Stroke shock{.colour = withAlpha(theme_.wakeFlash, boot.shockAlpha * fade),
                           .thickness = kShockThickness * scene.scale,
                           .glow = glow()};
        glowCircle(draw, centre, radius * boot.shockRadius, shock);
        glowCircle(draw, centre, radius * boot.shockRadius * kShockEcho,
                   Stroke{.colour = withAlpha(light, 0.5F * boot.shockAlpha * fade),
                          .thickness = scene.scale,
                          .glow = kSoftGlow});
    }
    const float sweepIn =
        anim::progress(boot.time, animator_.timeline().beats().assemblyStart, kSweepFadeIn);
    sweep(draw, centre, radius * kSweepRadius, boot.sweepAngle, kSweepSpan,
          withAlpha(light, 0.18F * sweepIn * fade));
}

void HudRenderer::drawBootStream(ImDrawList& draw, const HudScene& scene,
                                 const anim::BootFrame& boot) {
    const anim::ParticleField& field = animator_.particles();
    if (boot.particleTime < 0.0F || boot.particleTime > field.settledAfter()) {
        return;
    }
    field.evaluate(boot.particleTime, particleScratch_);
    const ImVec2 centre = centreOf(scene);
    const float radius = scene.geometry.radius;
    const Rgba light = accent(scene);
    const float fade = boot.bootOnly * scene.opacity;
    for (const anim::ParticlePoint& point : particleScratch_) {
        const ImVec2 at{centre.x + (point.x * radius), centre.y + (point.y * radius)};
        glowDot(draw, at, 1.3F * scene.scale * point.size, withAlpha(light, point.alpha * fade));
    }
}

void HudRenderer::drawScanLine(ImDrawList& draw, const HudScene& scene,
                               const anim::BootFrame& boot) const {
    const float alpha = boot.scanAlpha * boot.bootOnly * scene.opacity;
    if (alpha <= 0.0F) {
        return;
    }
    const RectF& canvas = scene.canvas;
    const float y = canvas.y + (boot.scan * canvas.height);
    const float band = canvas.height * 0.12F;
    const float middle = canvas.x + (canvas.width * 0.5F);
    const Rgba light = accent(scene);
    const ImU32 clear = toU32(withAlpha(light, 0.0F));
    const ImU32 lit = toU32(withAlpha(light, 0.16F * alpha));
    draw.AddRectFilledMultiColor({canvas.x, y - band}, {canvas.right(), y}, clear, clear, lit, lit);
    // The line itself fades out towards both ends of the canvas.
    const float half = 1.0F * scene.scale;
    const ImU32 core = toU32(withAlpha(light, 0.9F * alpha));
    draw.AddRectFilledMultiColor({canvas.x, y - half}, {middle, y + half}, clear, core, core,
                                 clear);
    draw.AddRectFilledMultiColor({middle, y - half}, {canvas.right(), y + half}, core, clear, clear,
                                 core);
}

void HudRenderer::drawGlyphRuns(ImDrawList& draw, const HudScene& scene,
                                const anim::BootFrame& boot, const HudFonts& fonts) const {
    const float start = animator_.timeline().beats().assemblyStart + kGlyphDelay;
    const float alpha =
        anim::progress(boot.time, start, kGlyphFadeIn) * boot.bootOnly * scene.opacity;
    if (alpha <= 0.0F) {
        return;
    }
    const ImVec2 centre = centreOf(scene);
    const float radius = scene.geometry.radius;
    const float size = radius * kGlyphSize;
    const auto roll = static_cast<std::uint64_t>(boot.time * kGlyphRate);
    const Rgba light = accent(scene);
    for (std::size_t run = 0; run < kGlyphRuns; ++run) {
        const float base = (0.5F * scene.frame.streamPhase) +
                           (kTau * static_cast<float>(run) / static_cast<float>(kGlyphRuns));
        for (std::size_t k = 0; k < kGlyphsPerRun; ++k) {
            const auto digit =
                static_cast<std::size_t>(anim::hashUnit((run * kGlyphsPerRun) + k, roll) *
                                         static_cast<float>(kHexDigits.size()));
            const std::string_view glyph =
                kHexDigits.substr(std::min(digit, kHexDigits.size() - 1), 1);
            const ImVec2 at = polar(centre, radius * kGlyphRadius,
                                    base + (kGlyphSpacing * static_cast<float>(k)));
            const float tail = 1.0F - (static_cast<float>(k) / static_cast<float>(kGlyphsPerRun));
            text(draw, *fonts.display, size, {at.x - (size * 0.3F), at.y - (size * 0.5F)}, glyph,
                 withAlpha(light, 0.75F * tail * alpha));
        }
    }
}

void HudRenderer::drawBootTitle(ImDrawList& draw, const HudScene& scene,
                                const anim::BootFrame& boot, const HudFonts& fonts) const {
    const float alpha = boot.bootOnly * scene.opacity;
    const RectF& block = scene.geometry.text;
    const float side = std::min(scene.canvas.width, scene.canvas.height);
    const float size = side * kTitleSize;
    const float tracking = size * kTitleTracking;
    const std::string& title = animator_.timeline().title();
    const float width = spacedTextWidth(*fonts.display, size, title, tracking);
    const float centreX = block.x + (block.width * 0.5F);
    // The glitch shakes the title sideways; the composite pass splits its colours.
    const float shake =
        (anim::hashUnit(boot.titleChars, static_cast<std::uint64_t>(boot.time * 60.0F)) - 0.5F) *
        boot.glitch * size * kTitleShake;
    const ImVec2 origin{centreX - (width * 0.5F) + shake, block.y};
    const std::string_view typed = std::string_view{title}.substr(0, boot.titleChars);
    const Rgba light = accent(scene);
    const float typedWidth = spacedText(draw, *fonts.display, size, origin, typed, tracking,
                                        withAlpha(mixColour(light, theme_.text, 0.7F), alpha));
    if (boot.cursorVisible) {
        const float x = origin.x + typedWidth + (typed.empty() ? 0.0F : tracking * 0.5F);
        draw.AddRectFilled({x, origin.y + (size * 0.12F)}, {x + (size * 0.42F), origin.y + size},
                           toU32(withAlpha(light, 0.85F * alpha)));
    }
    const float subSize = side * kSubtitleSize;
    const float subTracking = subSize * kSubtitleTracking;
    const float subWidth = spacedTextWidth(*fonts.display, subSize, kSubtitle, subTracking);
    const float subY = origin.y + (size * kSubtitleDrop);
    const float lineHalf = width * 0.55F * boot.subtitle;
    glowLine(draw, {centreX - lineHalf, subY - (subSize * 0.5F)},
             {centreX + lineHalf, subY - (subSize * 0.5F)},
             Stroke{.colour = withAlpha(light, 0.6F * boot.subtitle * alpha),
                    .thickness = scene.scale,
                    .glow = kSoftGlow});
    spacedText(draw, *fonts.display, subSize, {centreX - (subWidth * 0.5F), subY}, kSubtitle,
               subTracking, withAlpha(light, boot.subtitle * alpha));
}

void HudRenderer::drawReadout(ImDrawList& draw, const HudScene& scene, const anim::BootFrame& boot,
                              const HudFonts& fonts) const {
    std::array<const ReadoutLine*, kMaxReadout> lines{};
    const std::size_t count = collectReadout(scene, lines);
    const float alpha = boot.bootOnly * scene.opacity;
    if (count == 0 || alpha <= 0.0F) {
        return;
    }
    const float side = std::min(scene.canvas.width, scene.canvas.height);
    const float size = side * kReadoutSize;
    const float gap = size * kReadoutGap;
    float labelWidth = 0.0F;
    float valueWidth = 0.0F;
    for (const ReadoutLine* line : std::span{lines}.first(count)) {
        labelWidth = std::max(labelWidth, textWidth(*fonts.display, size, line->label));
        valueWidth = std::max(valueWidth, textWidth(*fonts.body, size, line->value));
    }
    const float okWidth = textWidth(*fonts.display, size, kReadoutOk);
    const float block = labelWidth + gap + valueWidth + gap + okWidth;
    const RectF& column = scene.geometry.text;
    const float left = column.x + ((column.width - block) * 0.5F);
    const float top =
        column.y + (side * kTitleSize * kSubtitleDrop) + (side * kSubtitleSize * kReadoutDrop);
    const Rgba light = accent(scene);
    std::size_t row = 0;
    for (const ReadoutLine* line : std::span{lines}.first(count)) {
        const std::size_t labelChars = utf8Length(line->label);
        const std::size_t total = labelChars + utf8Length(line->value);
        const std::size_t typed = animator_.timeline().readoutChars(row, total, boot.time);
        const float y = top + (static_cast<float>(row++) * size * kReadoutLine);
        text(draw, *fonts.display, size, {left, y}, utf8Prefix(line->label, typed),
             withAlpha(theme_.mutedText, alpha));
        const std::size_t valueChars = typed > labelChars ? typed - labelChars : 0;
        text(draw, *fonts.body, size, {left + labelWidth + gap, y},
             utf8Prefix(line->value, valueChars), withAlpha(theme_.text, alpha));
        if (typed >= total && typed > 0) {
            text(draw, *fonts.display, size, {left + block - okWidth, y}, kReadoutOk,
                 withAlpha(light, alpha));
        }
    }
}

}  // namespace jarvis::overlay::hud

// Design notes:
// - The boot's act structure lives in animation.cpp (BootTimeline); this
//   file only maps each act's numbers to strokes, so retiming an act never
//   touches drawing code.
// - The title is drawn inside the light layer so it blooms and takes the
//   composite pass's chromatic glitch; the readout is drawn after the light
//   layer so the real values Python passed in (model, mode, microphone) stay
//   perfectly legible while the rest of the HUD glitches around them.
// - Readout lines type on by code point (utf8Prefix), so a non-ASCII
//   microphone name never shows half a character.
