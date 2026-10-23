#include "hud.hpp"

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/animator.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/orb.hpp"
#include "jarvis/overlay/scene.hpp"

#include <imgui.h>

#include "hud_draw.hpp"
#include "hud_util.hpp"
#include "postfx.hpp"
#include <d3d11.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jarvis::overlay::hud {

namespace {

using anim::kTau;

constexpr std::array<AssistantState, 4> kStates{AssistantState::Idle, AssistantState::Listening,
                                                AssistantState::Thinking, AssistantState::Speaking};

// Idle is deliberately dim and slow: it is on screen all day.
constexpr float kIdleIntensity = 0.85F;
constexpr float kIdleBreathIntensity = 0.12F;
/// Loud audio brightens every stroke a little on top of the bloom.
constexpr float kEnergyIntensity = 0.25F;

constexpr float kCoreRadius = 0.09F;
constexpr float kFlashHalo = 0.45F;

constexpr std::size_t kSpokes = 72;
constexpr std::size_t kWavePoints = 120;
constexpr float kWaveFloor = 0.05F;
constexpr float kWaveInner = 0.50F;
constexpr float kSpokeReach = 0.40F;
constexpr float kSpeakReach = 0.30F;
/// Syllable onsets punch the speaking waveform out further.
constexpr float kOnsetReach = 0.35F;

}  // namespace

HudRenderer::HudRenderer(ID3D11Device& device, Theme theme, const anim::OverlayAnimator& animator,
                         PostProcessor* post)
    : theme_(theme), animator_(animator), blend_(device, post), globe_(anim::buildGlobe({})) {
    spokeScratch_.resize(std::max(kSpokes, kWavePoints));
    pointScratch_.reserve(kWavePoints);
}

void HudRenderer::draw(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts) {
    if (scene.orb >= 1.0F) {
        drawOrbFrame(draw, scene);  // minimized: the orb alone, the cheap idle path
        return;
    }
    // Mid-morph the HUD's scene fades as its ring shrinks into the orb,
    // and the orb fades in on top of it (orb.hpp has the curves).
    const OrbMorph morph = orbMorphAt(scene.orb);
    HudScene hud = scene;
    hud.opacity *= morph.sceneAlpha;
    const float orbAlpha = morph.orbAlpha * scene.opacity;
    drawScrim(draw, hud);
    drawExpandedScrim(draw, hud);
    drawOrbScrim(draw, scene, orbAlpha);
    blend_.beginLight(draw);
    drawSceneLight(draw, hud, fonts);
    drawOrbLight(draw, scene, orbAlpha);
    blend_.endLight(draw);
    // Text after the light layer: never bloomed, never aberrated, always crisp.
    if (hud.frame.boot.has_value()) {
        drawReadout(draw, hud, *hud.frame.boot, fonts);
    }
    drawPanelText(draw, hud, fonts);
    drawExpandedCaption(draw, hud, fonts);
}

void HudRenderer::drawSceneLight(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts) {
    const auto& boot = scene.frame.boot;
    drawBackdropGlow(draw, scene);
    if (boot.has_value()) {
        drawIgnition(draw, scene, *boot);
        drawBootStream(draw, scene, *boot);
        drawScanLine(draw, scene, *boot);
        drawGlyphRuns(draw, scene, *boot, fonts);
    }
    drawField(draw, scene);
    drawGlobe(draw, scene);
    drawRings(draw, scene, fonts);
    drawConnector(draw, scene);
    drawTelemetry(draw, scene);
    drawSatellites(draw, scene);
    drawCore(draw, scene);
    drawListening(draw, scene);
    drawThinking(draw, scene);
    drawSpeaking(draw, scene);
    drawRipple(draw, scene);
    drawExpandedLight(draw, scene);
    if (boot.has_value()) {
        drawBootTitle(draw, scene, *boot, fonts);
    }
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
    const float energy = kEnergyIntensity * scene.frame.energy.level;
    const float state = std::min((idle * calm) + (1.0F - idle) + energy, 1.0F);
    if (scene.frame.boot.has_value()) {
        return anim::lerp(1.0F, state, scene.frame.boot->handoff);
    }
    return state;
}

float HudRenderer::panelAlpha(const HudScene& scene) noexcept {
    return scene.frame.boot.has_value() ? scene.frame.boot->handoff : 1.0F;
}

std::span<const GlowPass> HudRenderer::glow() const noexcept {
    if (blend_.bloom()) {
        return kBloomGlow;
    }
    return kGlow;
}

// --- core and state layers --------------------------------------------------------------------

void HudRenderer::drawCore(ImDrawList& draw, const HudScene& scene) const {
    const anim::AnimationFrame& frame = scene.frame;
    const float thinkingPulse = 0.5F + (0.3F * std::sin(frame.time * 5.0F));
    const float speech = 0.55F + (0.45F * std::max(frame.energy.level, frame.energy.peak * 0.8F));
    const std::array<float, 4> cores{0.25F + (0.45F * frame.breathing), speech, thinkingPulse,
                                     speech};
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
    const float pulse = 1.0F + flash + (0.6F * frame.energy.onset);
    radialGlow(draw, centre, radius * 0.32F * pulse,
               withAlpha(light, 0.35F * core * scene.opacity));
    glowDot(draw, centre, radius * kCoreRadius * pulse,
            withAlpha(mixColour(light, theme_.wakeFlash, flash), core * scene.opacity));
}

void HudRenderer::drawListening(ImDrawList& draw, const HudScene& scene) {
    const float weight = weightOf(scene, AssistantState::Listening);
    if (weight < kNegligible) {
        return;
    }
    const std::span<float> spokes{spokeScratch_.data(), kSpokes};
    anim::radialWaveform(scene.frame.levels, scene.frame.level, scene.frame.time, kWaveFloor,
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
    const Stroke arc{
        .colour = withAlpha(light, strength), .thickness = 2.0F * scene.scale, .glow = glow()};
    const float first = time * 2.6F;
    const float second = kPi - (time * 1.8F);
    glowArc(draw, centre, radius * 0.50F, first, first + 1.2F, arc);
    glowArc(draw, centre, radius * 0.50F, second - 0.7F, second, arc);
    constexpr int kDots = 3;
    for (int i = 0; i < kDots; ++i) {
        const float angle = (time * 1.5F) + (kTau * static_cast<float>(i) / kDots);
        glowArc(draw, centre, radius * 0.95F, angle - 0.55F, angle,
                Stroke{.colour = withAlpha(light, 0.35F * strength),
                       .thickness = 1.2F * scene.scale,
                       .glow = kSoftGlow});
        glowDot(draw, polar(centre, radius * 0.95F, angle), 2.2F * scene.scale,
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
    const float punch = 1.0F + (kOnsetReach * scene.frame.energy.onset);
    // Two traces, the second slower and fainter, so the waveform has depth.
    for (const auto& [phase, reach, alpha] :
         {std::array<float, 3>{0.0F, kSpeakReach, 1.0F},
          std::array<float, 3>{2.1F, kSpeakReach * 0.6F, 0.4F}}) {
        anim::radialWaveform(scene.frame.levels, scene.frame.level,
                             (scene.frame.time * 1.3F) + phase, kWaveFloor, wave);
        pointScratch_.clear();
        for (const float amplitude : wave) {
            const float r = radius * (kWaveInner + (amplitude * reach * punch));
            const auto index = static_cast<float>(pointScratch_.size());
            pointScratch_.push_back(polar(centre, r, kTop + (step * index)));
        }
        glowPolyline(draw, pointScratch_, true,
                     Stroke{.colour = withAlpha(light, alpha * strength),
                            .thickness = 1.5F * scene.scale,
                            .glow = glow()});
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
    glowCircle(draw, centre, radius * ripple.radius,
               Stroke{.colour = withAlpha(theme_.wakeFlash, alpha),
                      .thickness = 2.0F * scene.scale,
                      .glow = glow()});
    const float echo = 1.0F + ((ripple.radius - 1.0F) * 0.6F);
    glowCircle(draw, centre, radius * echo,
               Stroke{.colour = withAlpha(theme_.wakeFlash, 0.5F * alpha),
                      .thickness = 1.2F * scene.scale,
                      .glow = kSoftGlow});
}

}  // namespace jarvis::overlay::hud

// Design notes:
// - All four state layers are drawn every frame, each scaled by its
//   cross-fade weight, so a transition is literally a blend of two pictures
//   and there is no per-state branching to get out of sync.
// - Draw order is three sections: the dark scrims with normal blending,
//   everything that is light inside BlendModes::beginLight/endLight (bloomed
//   by the GPU pipeline), then the panel text and the boot readout, which
//   therefore stay crisp whatever the bloom and the glitch do.
// - With bloom the strokes need fewer glow passes (kBloomGlow): the blur
//   supplies the wide halo, which also saves vertices.
