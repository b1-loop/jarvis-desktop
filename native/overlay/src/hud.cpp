#include "hud.hpp"

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"

#include <imgui.h>

#include "hud_draw.hpp"
#include <d3d11.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jarvis::overlay::hud {

namespace {

using anim::clamp01;
using anim::kTau;

constexpr float kPi = std::numbers::pi_v<float>;
constexpr float kTop = -kPi / 2.0F;  // angles start at twelve o'clock
constexpr std::array<AssistantState, 4> kStates{AssistantState::Idle, AssistantState::Listening,
                                                AssistantState::Thinking, AssistantState::Speaking};
constexpr float kNegligible = 0.005F;

/// One concentric ring of the reactor. Radii are fractions of the outer
/// radius, speeds radians per second; a negative speed turns anticlockwise.
struct RingLayer {
    float radius;
    float thickness;
    int segments;  // 0 draws a full circle
    float fill;
    float speed;
    float alpha;
};

constexpr std::array<RingLayer, 5> kReactor{{
    {.radius = 1.00F,
     .thickness = 1.0F,
     .segments = 0,
     .fill = 1.0F,
     .speed = 0.0F,
     .alpha = 0.40F},
    {.radius = 0.90F,
     .thickness = 2.2F,
     .segments = 3,
     .fill = 0.24F,
     .speed = 0.30F,
     .alpha = 0.85F},
    {.radius = 0.78F,
     .thickness = 1.2F,
     .segments = 36,
     .fill = 0.50F,
     .speed = -0.12F,
     .alpha = 0.50F},
    {.radius = 0.64F,
     .thickness = 2.6F,
     .segments = 5,
     .fill = 0.58F,
     .speed = 0.50F,
     .alpha = 0.75F},
    {.radius = 0.46F,
     .thickness = 1.0F,
     .segments = 0,
     .fill = 1.0F,
     .speed = 0.0F,
     .alpha = 0.35F},
}};
/// Extra radians each layer turns during boot before settling (BootFrame::spin).
constexpr float kSpinBoost = 5.0F;
/// Inner rings appear first as the point of light expands outward.
constexpr float kRevealRate = 1.5F;
constexpr float kRevealStagger = 0.5F;

/// Stroke widths are designed for the panel ring (about 52 px at 96 DPI);
/// the larger boot ring thickens them a little, not proportionally.
constexpr float kReferenceRadius = 52.0F;
constexpr float kMaxStrokeGrowth = 1.8F;

// Idle is deliberately dim and slow: it is on screen all day.
constexpr float kIdleIntensity = 0.50F;
constexpr float kIdleBreathIntensity = 0.12F;

constexpr float kCoreRadius = 0.09F;
constexpr float kFlashHalo = 0.45F;

constexpr std::size_t kSpokes = 72;
constexpr std::size_t kWavePoints = 120;
constexpr float kWaveFloor = 0.05F;
constexpr float kWaveInner = 0.50F;
constexpr float kSpokeReach = 0.40F;
constexpr float kSpeakReach = 0.30F;

constexpr std::size_t kHexCount = 22;
constexpr std::uint64_t kHexSeed = 0x4845584147524944ULL;
constexpr float kFlickerRate = 22.0F;
constexpr float kGridSettle = 0.45F;

// Panel typography, in 96-DPI pixels.
constexpr float kCaptionSize = 14.0F;
constexpr float kCaptionTracking = 2.2F;
constexpr float kBrandSize = 10.0F;
constexpr float kBrandTracking = 1.6F;
constexpr float kPrefixSize = 10.5F;
constexpr float kPrefixTracking = 1.4F;
constexpr float kBodySize = 14.5F;
constexpr float kPrefixGap = 8.0F;
constexpr float kPlateRounding = 12.0F;
constexpr float kBracketInset = 4.0F;
constexpr float kBracketLength = 10.0F;

constexpr std::string_view kBrand = "J.A.R.V.I.S";
constexpr std::string_view kWakeCaption = "WAKE WORD";
constexpr std::string_view kYouPrefix = "YOU";
constexpr std::string_view kJarvisPrefix = "JARVIS";
constexpr std::string_view kSubtitle = "SYSTEMS ONLINE";

ImVec2 centreOf(const HudScene& scene) noexcept {
    return ImVec2{scene.geometry.centreX, scene.geometry.centreY};
}

/// Stroke width multiplier for the current ring size.
float strokeScale(const HudScene& scene) noexcept {
    const float reference = kReferenceRadius * scene.scale;
    const float growth = reference > 0.0F ? scene.geometry.radius / reference : 1.0F;
    return scene.scale * std::clamp(growth, 1.0F, kMaxStrokeGrowth);
}

float weightOf(const HudScene& scene, AssistantState state) noexcept {
    return scene.frame.weights.at(anim::stateIndex(state));
}

float fraction(float value) noexcept {
    return value - std::floor(value);
}

}  // namespace

HudRenderer::HudRenderer(ID3D11Device& device, Theme theme, const anim::OverlayAnimator& animator)
    : theme_(theme), animator_(animator), blend_(device) {
    anim::SplitMix64 random{kHexSeed};
    hexes_.reserve(kHexCount);
    for (std::size_t i = 0; i < kHexCount; ++i) {
        HexFragment hex;
        hex.angle = random.uniform(0.0F, kTau);
        hex.distance = random.uniform(1.45F, 2.25F);
        hex.size = random.uniform(0.06F, 0.11F);
        hex.rotation = random.nextUnit() < 0.5F ? 0.0F : kTau / 12.0F;
        hexes_.push_back(hex);
    }
    spokeScratch_.resize(std::max(kSpokes, kWavePoints));
    pointScratch_.reserve(kWavePoints);
}

void HudRenderer::draw(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts) {
    drawPlate(draw, scene);
    blend_.additive(draw);
    if (scene.frame.boot.has_value()) {
        drawBootBackdrop(draw, scene, *scene.frame.boot);
    }
    drawReactor(draw, scene);
    drawCore(draw, scene);
    drawListening(draw, scene);
    drawThinking(draw, scene);
    drawSpeaking(draw, scene);
    drawRipple(draw, scene);
    BlendModes::normal(draw);
    if (scene.frame.boot.has_value()) {
        drawBootTitle(draw, scene, *scene.frame.boot, fonts);
    }
    drawPanelText(draw, scene, fonts);
}

// --- shared state ----------------------------------------------------------------

Rgba HudRenderer::accent(const HudScene& scene) const noexcept {
    Rgba blended{.r = 0.0F, .g = 0.0F, .b = 0.0F, .a = 1.0F};
    for (const AssistantState state : kStates) {
        const float weight = weightOf(scene, state);
        const Rgba colour = stateColour(theme_, state);
        blended.r += colour.r * weight;
        blended.g += colour.g * weight;
        blended.b += colour.b * weight;
    }
    if (scene.frame.boot.has_value()) {
        return mixColour(theme_.accent, blended, scene.frame.boot->handoff);
    }
    return blended;
}

float HudRenderer::intensity(const HudScene& scene) noexcept {
    const float idle = weightOf(scene, AssistantState::Idle);
    const float calm = kIdleIntensity + (kIdleBreathIntensity * scene.frame.breathing);
    const float state = (idle * calm) + (1.0F - idle);
    if (scene.frame.boot.has_value()) {
        return anim::lerp(1.0F, state, scene.frame.boot->handoff);
    }
    return state;
}

float HudRenderer::panelAlpha(const HudScene& scene) noexcept {
    return scene.frame.boot.has_value() ? scene.frame.boot->handoff : 1.0F;
}

// --- plate -------------------------------------------------------------------------

void HudRenderer::drawPlate(ImDrawList& draw, const HudScene& scene) const {
    const float alpha = panelAlpha(scene) * scene.opacity;
    if (alpha <= 0.0F) {
        return;
    }
    const RectF& panel = scene.panel;
    const float rounding = kPlateRounding * scene.scale;
    const ImVec2 topLeft{panel.x, panel.y};
    const ImVec2 bottomRight{panel.right(), panel.bottom()};
    draw.AddRectFilled(topLeft, bottomRight, toU32(withAlpha(theme_.plate, alpha)), rounding);
    const Rgba light = accent(scene);
    draw.AddRect(topLeft, bottomRight, toU32(withAlpha(light, 0.16F * alpha)), rounding,
                 scene.scale);
    const float inset = kBracketInset * scene.scale;
    const RectF inner{.x = panel.x + inset,
                      .y = panel.y + inset,
                      .width = panel.width - (2.0F * inset),
                      .height = panel.height - (2.0F * inset)};
    cornerBrackets(draw, inner, kBracketLength * scene.scale,
                   withAlpha(light, 0.55F * intensity(scene) * alpha), 1.2F * scene.scale);
}

// --- boot-only elements ---------------------------------------------------------------

void HudRenderer::drawBootBackdrop(ImDrawList& draw, const HudScene& scene,
                                   const anim::BootFrame& boot) {
    const ImVec2 centre = centreOf(scene);
    const float radius = scene.geometry.radius;
    const float fade = boot.bootOnly * boot.ignite * scene.opacity;
    const Rgba light = accent(scene);
    sweep(draw, centre, radius * 1.18F, boot.sweepAngle, 1.1F, withAlpha(light, 0.22F * fade));
    const TickSpec outer{.inner = radius * 1.16F,
                         .outer = radius * 1.21F,
                         .majorOuter = radius * 1.27F,
                         .count = 72,
                         .majorEvery = 6,
                         .rotation = -0.05F * scene.frame.time,
                         .thickness = scene.scale};
    ticks(draw, centre, outer, withAlpha(light, 0.55F * fade));
    segmentedRing(
        draw, centre, radius * 1.34F, 4, 0.2F, 0.2F * scene.frame.time,
        Stroke{
            .colour = withAlpha(light, 0.30F * fade), .thickness = scene.scale, .glow = kSoftGlow});
    drawHexGrid(draw, scene, boot);
    drawParticles(draw, scene, boot);
    drawScanLine(draw, scene, boot);
}

void HudRenderer::drawHexGrid(ImDrawList& draw, const HudScene& scene,
                              const anim::BootFrame& boot) const {
    const float envelope = boot.grid * boot.bootOnly * scene.opacity;
    if (envelope <= 0.0F) {
        return;
    }
    const ImVec2 centre = centreOf(scene);
    const float radius = scene.geometry.radius;
    const bool settled = boot.time > animator_.timeline().beats().gridStart + kGridSettle;
    const auto step = static_cast<std::uint64_t>(std::max(boot.time, 0.0F) * kFlickerRate);
    const Rgba light = accent(scene);
    std::uint64_t index = 0;
    for (const HexFragment& hex : hexes_) {
        const float flicker = anim::hashUnit(index++, step) > 0.4F ? 0.8F : 0.1F;
        const float on = settled ? 0.5F : flicker;
        hexagon(draw, polar(centre, hex.distance * radius, hex.angle), hex.size * radius,
                hex.rotation, withAlpha(light, on * envelope), scene.scale);
    }
}

void HudRenderer::drawParticles(ImDrawList& draw, const HudScene& scene,
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

void HudRenderer::drawBootTitle(ImDrawList& draw, const HudScene& scene,
                                const anim::BootFrame& boot, const HudFonts& fonts) const {
    const float alpha = boot.bootOnly * scene.opacity;
    const RectF& block = scene.geometry.text;
    const float side = std::min(scene.canvas.width, scene.canvas.height);
    const float size = side * 0.072F;
    const float tracking = size * 0.42F;
    const std::string& title = animator_.timeline().title();
    const float width = spacedTextWidth(*fonts.display, size, title, tracking);
    const float centreX = block.x + (block.width * 0.5F);
    const ImVec2 origin{centreX - (width * 0.5F), block.y};
    const std::string_view typed = std::string_view{title}.substr(0, boot.titleChars);
    const Rgba light = accent(scene);

    const float offset = 1.5F * scene.scale;
    for (const ImVec2 shift :
         {ImVec2{offset, 0}, ImVec2{-offset, 0}, ImVec2{0, offset}, ImVec2{0, -offset}}) {
        spacedText(draw, *fonts.display, size, {origin.x + shift.x, origin.y + shift.y}, typed,
                   tracking, withAlpha(light, 0.22F * alpha));
    }
    const float typedWidth = spacedText(draw, *fonts.display, size, origin, typed, tracking,
                                        withAlpha(theme_.text, alpha));
    if (boot.cursorVisible) {
        const float x = origin.x + typedWidth + (typed.empty() ? 0.0F : tracking * 0.5F);
        draw.AddRectFilled({x, origin.y + (size * 0.12F)}, {x + (size * 0.42F), origin.y + size},
                           toU32(withAlpha(light, 0.85F * alpha)));
    }

    const float subSize = side * 0.03F;
    const float subTracking = subSize * 0.75F;
    const float subWidth = spacedTextWidth(*fonts.display, subSize, kSubtitle, subTracking);
    const float subY = origin.y + (size * 1.55F);
    const float lineHalf = width * 0.55F * boot.subtitle;
    glowLine(draw, {centreX - lineHalf, subY - (subSize * 0.5F)},
             {centreX + lineHalf, subY - (subSize * 0.5F)},
             Stroke{.colour = withAlpha(light, 0.5F * boot.subtitle * alpha),
                    .thickness = scene.scale,
                    .glow = kSoftGlow});
    spacedText(draw, *fonts.display, subSize, {centreX - (subWidth * 0.5F), subY}, kSubtitle,
               subTracking, withAlpha(theme_.mutedText, boot.subtitle * alpha));
}

// --- the reactor (boot and panel) -------------------------------------------------------------

void HudRenderer::drawReactor(ImDrawList& draw, const HudScene& scene) const {
    const ImVec2 centre = centreOf(scene);
    const float radius = scene.geometry.radius;
    const auto& boot = scene.frame.boot;
    const float ignite = boot.has_value() ? boot->ignite : 1.0F;
    const float spin = boot.has_value() ? boot->spin : 1.0F;
    const float strength = intensity(scene) * scene.opacity;
    const float stroke = strokeScale(scene);
    const Rgba light = accent(scene);
    for (const RingLayer& layer : kReactor) {
        const float reveal =
            anim::easeOutCubic(clamp01((ignite * kRevealRate) - (layer.radius * kRevealStagger)));
        const float direction = layer.speed < 0.0F ? -1.0F : 1.0F;
        const float rotation =
            kTop + (layer.speed * scene.frame.time) + (direction * kSpinBoost * spin);
        segmentedRing(draw, centre, radius * layer.radius * reveal, layer.segments, layer.fill,
                      rotation,
                      Stroke{.colour = withAlpha(light, layer.alpha * strength * reveal),
                             .thickness = layer.thickness * stroke});
    }
    const TickSpec bezel{.inner = radius * 1.04F,
                         .outer = radius * 1.08F,
                         .majorOuter = radius * 1.13F,
                         .count = 48,
                         .majorEvery = 4,
                         .rotation = kTop,
                         .thickness = scene.scale};
    ticks(draw, centre, bezel, withAlpha(light, 0.30F * strength * ignite));
}

void HudRenderer::drawCore(ImDrawList& draw, const HudScene& scene) const {
    const anim::AnimationFrame& frame = scene.frame;
    const float thinkingPulse = 0.5F + (0.3F * std::sin(frame.time * 5.0F));
    const std::array<float, 4> cores{
        0.25F + (0.45F * frame.breathing),
        0.55F + (0.45F * frame.level),
        thinkingPulse,
        0.55F + (0.45F * frame.level),
    };
    float core = 0.0F;
    for (std::size_t i = 0; i < cores.size(); ++i) {
        core += cores.at(i) * frame.weights.at(i);
    }
    const ImVec2 centre = centreOf(scene);
    const float radius = scene.geometry.radius;
    const Rgba light = accent(scene);
    float flash = 0.0F;
    if (frame.boot.has_value()) {
        flash = frame.boot->flash;
        core = std::max(core * frame.boot->handoff, flash);
        glowDot(draw, centre, radius * kFlashHalo * flash, withAlpha(light, 0.2F * flash));
    }
    glowDot(draw, centre, radius * kCoreRadius * (1.0F + flash),
            withAlpha(mixColour(light, theme_.wakeFlash, flash), core * scene.opacity));
}

// --- state layers ---------------------------------------------------------------------------

void HudRenderer::drawListening(ImDrawList& draw, const HudScene& scene) {
    const float weight = weightOf(scene, AssistantState::Listening);
    if (weight < kNegligible) {
        return;
    }
    const std::span<float> spokes{spokeScratch_.data(), kSpokes};
    anim::radialWaveform(scene.snapshot.levels, scene.frame.level, scene.frame.time, kWaveFloor,
                         spokes);
    const ImVec2 centre = centreOf(scene);
    const float radius = scene.geometry.radius;
    const float strength = weight * intensity(scene) * scene.opacity;
    const Rgba light = accent(scene);
    const float step = kTau / static_cast<float>(kSpokes);
    std::size_t index = 0;
    for (const float amplitude : spokes) {
        const float angle = kTop + (step * static_cast<float>(index++));
        const float inner = radius * kWaveInner;
        glowLine(draw, polar(centre, inner, angle),
                 polar(centre, inner + (amplitude * kSpokeReach * radius), angle),
                 Stroke{.colour = withAlpha(light, strength * (0.45F + (0.55F * amplitude))),
                        .thickness = 1.4F * scene.scale,
                        .glow = kSoftGlow});
    }
}

void HudRenderer::drawThinking(ImDrawList& draw, const HudScene& scene) const {
    const float weight = weightOf(scene, AssistantState::Thinking);
    if (weight < kNegligible) {
        return;
    }
    const ImVec2 centre = centreOf(scene);
    const float radius = scene.geometry.radius;
    const float time = scene.frame.time;
    const float strength = weight * intensity(scene) * scene.opacity;
    const Rgba light = accent(scene);
    const Stroke arc{.colour = withAlpha(light, strength), .thickness = 2.0F * scene.scale};
    const float first = time * 2.6F;
    const float second = kPi - (time * 1.8F);
    glowArc(draw, centre, radius * 0.56F, first, first + 1.2F, arc);
    glowArc(draw, centre, radius * 0.56F, second - 0.7F, second, arc);
    constexpr int kDots = 3;
    for (int i = 0; i < kDots; ++i) {
        const float angle = (time * 1.5F) + (kTau * static_cast<float>(i) / kDots);
        glowArc(draw, centre, radius * 0.88F, angle - 0.55F, angle,
                Stroke{.colour = withAlpha(light, 0.35F * strength),
                       .thickness = 1.2F * scene.scale,
                       .glow = kSoftGlow});
        glowDot(draw, polar(centre, radius * 0.88F, angle), 2.2F * scene.scale,
                withAlpha(light, strength));
    }
}

void HudRenderer::drawSpeaking(ImDrawList& draw, const HudScene& scene) {
    const float weight = weightOf(scene, AssistantState::Speaking);
    if (weight < kNegligible) {
        return;
    }
    const ImVec2 centre = centreOf(scene);
    const float radius = scene.geometry.radius;
    const float strength = weight * intensity(scene) * scene.opacity;
    const Rgba light = accent(scene);
    const std::span<float> wave{spokeScratch_.data(), kWavePoints};
    const float step = kTau / static_cast<float>(kWavePoints);
    // Two traces, the second slower and fainter, so the waveform has depth.
    for (const auto& [phase, reach, alpha] :
         {std::array<float, 3>{0.0F, kSpeakReach, 1.0F},
          std::array<float, 3>{2.1F, kSpeakReach * 0.6F, 0.4F}}) {
        anim::radialWaveform(scene.snapshot.levels, scene.frame.level,
                             (scene.frame.time * 1.3F) + phase, kWaveFloor, wave);
        pointScratch_.clear();
        for (const float amplitude : wave) {
            const float r = radius * (kWaveInner + (amplitude * reach));
            const auto index = static_cast<float>(pointScratch_.size());
            pointScratch_.push_back(polar(centre, r, kTop + (step * index)));
        }
        glowPolyline(
            draw, pointScratch_, true,
            Stroke{.colour = withAlpha(light, alpha * strength), .thickness = 1.5F * scene.scale});
    }
}

void HudRenderer::drawRipple(ImDrawList& draw, const HudScene& scene) const {
    const anim::Ripple& ripple = scene.frame.ripple;
    if (ripple.alpha <= 0.0F) {
        return;
    }
    const ImVec2 centre = centreOf(scene);
    const float radius = scene.geometry.radius;
    const float alpha = ripple.alpha * scene.opacity;
    glowCircle(
        draw, centre, radius * ripple.radius,
        Stroke{.colour = withAlpha(theme_.wakeFlash, alpha), .thickness = 2.0F * scene.scale});
    const float echo = 1.0F + ((ripple.radius - 1.0F) * 0.6F);
    glowCircle(draw, centre, radius * echo,
               Stroke{.colour = withAlpha(theme_.wakeFlash, 0.5F * alpha),
                      .thickness = 1.2F * scene.scale,
                      .glow = kSoftGlow});
}

// --- panel text ------------------------------------------------------------------------------

void HudRenderer::drawPanelText(ImDrawList& draw, const HudScene& scene,
                                const HudFonts& fonts) const {
    const float alpha = panelAlpha(scene) * scene.opacity;
    if (alpha <= 0.0F) {
        return;
    }
    const RectF& column = scene.panelText;
    drawCaption(draw, scene, *fonts.display, alpha);
    drawDivider(draw, scene, column.y + (column.height * 0.27F), alpha);

    const float prefixSize = kPrefixSize * scene.scale;
    const float prefixTracking = kPrefixTracking * scene.scale;
    const float bodySize = kBodySize * scene.scale;
    const float textX = column.x +
                        spacedTextWidth(*fonts.display, prefixSize, kJarvisPrefix, prefixTracking) +
                        (kPrefixGap * scene.scale);
    const float charWidth = fonts.body->CalcTextSizeA(bodySize, FLT_MAX, 0.0F, "n").x;
    const std::size_t fit = charactersPerLine(column.right() - textX, charWidth);
    const float prefixDrop = (bodySize - prefixSize) * 0.5F;

    const std::array<std::pair<float, bool>, 2> rows{{{0.38F, false}, {0.68F, true}}};
    for (const auto& [row, isReply] : rows) {
        const float y = column.y + (column.height * row);
        const Rgba prefixColour = isReply ? accent(scene) : theme_.mutedText;
        spacedText(draw, *fonts.display, prefixSize, {column.x, y + prefixDrop},
                   isReply ? kJarvisPrefix : kYouPrefix, prefixTracking,
                   withAlpha(prefixColour, alpha));
        const std::string& line = isReply ? scene.snapshot.response : scene.snapshot.transcript;
        text(draw, *fonts.body, bodySize, {textX, y}, truncateUtf8Tail(line, fit),
             withAlpha(theme_.text, alpha));
    }
}

void HudRenderer::drawCaption(ImDrawList& draw, const HudScene& scene, ImFont& font,
                              float alpha) const {
    const RectF& column = scene.panelText;
    const float size = kCaptionSize * scene.scale;
    const float tracking = kCaptionTracking * scene.scale;
    // Labels cross-fade with the state, like everything else.
    for (const AssistantState state : kStates) {
        const float weight = weightOf(scene, state);
        if (weight >= kNegligible) {
            spacedText(draw, font, size, {column.x, column.y}, hudLabel(state), tracking,
                       withAlpha(stateColour(theme_, state), weight * alpha));
        }
    }
    const float wake = scene.frame.ripple.alpha;
    const float brandSize = kBrandSize * scene.scale;
    const float brandTracking = kBrandTracking * scene.scale;
    const float baseline = column.y + ((size - brandSize) * 0.5F);
    const std::array<std::pair<std::string_view, Rgba>, 2> marks{{
        {kBrand, withAlpha(theme_.mutedText, 0.45F * (1.0F - wake) * alpha)},
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
    draw.AddLine({column.x, y}, {column.right(), y}, toU32(withAlpha(light, 0.22F * alpha)),
                 scene.scale);
    // A bright runner travels along the divider whenever Jarvis is busy.
    const float busy = 1.0F - weightOf(scene, AssistantState::Idle);
    if (busy < kNegligible) {
        return;
    }
    const float length = column.width * 0.22F;
    const float head = column.x + (fraction(scene.frame.time * 0.55F) * (column.width + length));
    const float from = std::max(head - length, column.x);
    const float to = std::min(head, column.right());
    if (to > from) {
        glowLine(draw, {from, y}, {to, y},
                 Stroke{.colour = withAlpha(light, 0.85F * busy * alpha),
                        .thickness = 1.2F * scene.scale,
                        .glow = kSoftGlow});
    }
}

}  // namespace jarvis::overlay::hud

// Design notes:
// - All four state layers are drawn every frame, each scaled by its
//   cross-fade weight, so a transition is literally a blend of two pictures
//   and there is no per-state branching to get out of sync.
// - The reactor is the same drawing during boot and in the panel; boot only
//   changes its geometry (radius and centre, interpolated by the window),
//   its reveal and an extra spin that winds down to exactly the idle
//   rotation, which is what makes the handoff seamless.
// - Draw order: plate and text use normal blending, everything that is
//   "light" is drawn in between inside one additive section.
