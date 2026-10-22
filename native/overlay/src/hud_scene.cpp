#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/animator.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/particles.hpp"
#include "jarvis/overlay/scene.hpp"

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
#include <string_view>

namespace jarvis::overlay::hud {

namespace {

using anim::clamp01;
using anim::kTau;

// Globe: a share of the ring radius, pulsing with the audio.
constexpr float kGlobeShare = 0.40F;
constexpr float kGlobePulse = 0.10F;
constexpr float kGlobeOnsetPulse = 0.08F;
/// The globe's axis leans like a planet's.
constexpr float kAxialTilt = 0.38F;
constexpr float kLatitudeAlpha = 0.42F;
constexpr float kMeridianAlpha = 0.32F;
constexpr float kOrbitAlpha = 0.55F;
/// Lines on the far side keep this much of their brightness.
constexpr float kBackLineFloor = 0.15F;
constexpr float kPenRadius = 1.8F;
constexpr int kDepthBuckets = 6;
constexpr float kGlobeStroke = 1.1F;

// Satellites: a short fading trail behind each node.
constexpr int kTrailSamples = 12;
constexpr float kTrailStep = 0.07F;
constexpr float kSatelliteSize = 1.7F;
constexpr float kSatelliteDepthSize = 0.9F;

/// Depth of each ring layer for parallax, outermost (nearest) first.
constexpr std::array<float, anim::kRingCount> kRingDepth{0.35F, 0.2F, 0.05F, -0.1F, -0.25F};
/// Extra radians each layer turns during boot before settling (BootFrame::spin).
constexpr float kSpinBoost = 5.0F;
/// Ring lock jitter: how far a ring shakes while the boot glitch is high.
constexpr float kLockShake = 0.04F;
constexpr float kTickInner = 1.03F;
constexpr float kTickOuter = 1.07F;
constexpr float kTickMajor = 1.12F;
constexpr float kLabelRadius = 1.20F;
/// Labels only appear on rings at least this large (96-DPI px): the boot ring.
constexpr float kLabelMinRadius = 80.0F;
constexpr float kLabelSize = 0.055F;
/// In the panel, labels further right than this (in ring radii) are left out.
constexpr float kLabelTextSide = 0.75F;
constexpr float kDashSpan = 0.09F;
constexpr float kDashFlow = 1.6F;

// Particle field.
constexpr float kDotSize = 1.25F;
constexpr float kDotDepthSize = 0.35F;
constexpr float kFieldAlpha = 0.55F;
constexpr float kFieldEnergyAlpha = 0.45F;
constexpr float kFieldWhiten = 0.35F;
constexpr float kMinDotAlpha = 0.01F;
/// Particles fade out between these distances right of the ring centre.
constexpr float kTextFadeStart = 1.05F;
constexpr float kTextFadeEnd = 1.45F;
constexpr int kQuadIndices = 6;
constexpr int kQuadVertices = 4;

constexpr float kBackdropRadius = 1.35F;
constexpr float kBackdropAlpha = 0.07F;

anim::Viewport globeViewport(const HudScene& scene) noexcept {
    const anim::AudioEnergy& energy = scene.frame.energy;
    const float pulse = 1.0F + (kGlobePulse * energy.level) + (kGlobeOnsetPulse * energy.onset);
    return anim::viewportFor(scene.frame.camera, scene.geometry.centreX, scene.geometry.centreY,
                             scene.geometry.radius * kGlobeShare * pulse);
}

/// 1 for the nearest point of the globe, 0 for the farthest.
float frontness(const anim::Camera& camera, float depth) noexcept {
    return clamp01((camera.distance + 1.0F - depth) * 0.5F);
}

/// Depth quantised into kDepthBuckets shades, so runs of segments can share a colour.
int depthBucket(const anim::Camera& camera, float depth) noexcept {
    const auto bucket =
        static_cast<int>(frontness(camera, depth) * static_cast<float>(kDepthBuckets));
    return std::min(bucket, kDepthBuckets - 1);
}

/// Brightness of a depth bucket: the far side keeps kBackLineFloor of it.
float depthShade(int bucket) noexcept {
    const float front = (static_cast<float>(bucket) + 0.5F) / static_cast<float>(kDepthBuckets);
    return kBackLineFloor + ((1.0F - kBackLineFloor) * front * front);
}

ImVec2 toImVec(const anim::Projected& point) noexcept {
    return ImVec2{point.x, point.y};
}

Rgba globeColour(anim::GlobeLineKind kind, Rgba light) noexcept {
    switch (kind) {
        case anim::GlobeLineKind::Meridian:
            return withAlpha(light, kMeridianAlpha);
        case anim::GlobeLineKind::Orbit:
            return withAlpha(light, kOrbitAlpha);
        case anim::GlobeLineKind::Latitude:
            break;
    }
    return withAlpha(light, kLatitudeAlpha);
}

/// Three zero-padded digits; HUD degree labels are 000-359.
std::string_view threeDigits(int value, std::array<char, 4>& buffer) noexcept {
    constexpr int kBase = 10;
    constexpr int kHundred = 100;
    const int clamped = std::clamp(value, 0, (kHundred * kBase) - 1);
    buffer.at(0) = static_cast<char>('0' + (clamped / kHundred));
    buffer.at(1) = static_cast<char>('0' + ((clamped / kBase) % kBase));
    buffer.at(2) = static_cast<char>('0' + (clamped % kBase));
    buffer.at(3) = '\0';
    return std::string_view{buffer.data(), 3};
}

}  // namespace

void HudRenderer::drawBackdropGlow(ImDrawList& draw, const HudScene& scene) const {
    const float flash = scene.frame.boot.has_value() ? scene.frame.boot->flash : 0.0F;
    const float alpha = (kBackdropAlpha * intensity(scene)) + (0.15F * flash);
    radialGlow(draw, centreOf(scene), scene.geometry.radius * kBackdropRadius,
               withAlpha(accent(scene), alpha * scene.opacity));
}

// --- globe -------------------------------------------------------------------------

void HudRenderer::drawGlobe(ImDrawList& draw, const HudScene& scene) {
    const auto& boot = scene.frame.boot;
    const float reveal = boot.has_value() ? boot->globe : 1.0F;
    if (reveal <= 0.0F) {
        return;
    }
    const float strength = intensity(scene) * scene.opacity;
    const Rgba light = accent(scene);
    const anim::Projector projector{scene.frame.camera, globeViewport(scene),
                                    scene.frame.globeAngle, kAxialTilt};
    std::size_t index = 0;
    for (const anim::GlobeLine& line : globe_) {
        const float drawn = anim::lineReveal(reveal, index++, globe_.size());
        if (drawn > 0.0F) {
            const Rgba colour = globeColour(line.kind, light);
            drawGlobeLine(draw, scene, projector, line, drawn, withAlpha(colour, strength));
        }
    }
}

void HudRenderer::drawGlobeLine(ImDrawList& draw, const HudScene& scene,
                                const anim::Projector& projector, const anim::GlobeLine& line,
                                float reveal, Rgba colour) {
    projectedScratch_.clear();
    for (const anim::Vec3& point : line.points) {
        projectedScratch_.push_back(projector(point));
    }
    const std::size_t count = projectedScratch_.size();
    const std::size_t segments = line.closed ? count : count - 1;
    const float drawn = reveal * static_cast<float>(segments);
    // Consecutive segments at a similar depth share one polyline: a handful
    // of draw calls per line instead of one per segment.
    pointScratch_.clear();
    int run = -1;
    for (std::size_t s = 0; s < segments && static_cast<float>(s) < drawn; ++s) {
        const anim::Projected& a = projectedScratch_.at(s);
        const anim::Projected& b = projectedScratch_.at((s + 1) % count);
        if (!a.visible || !b.visible) {
            flushRun(draw, scene, colour, run);
            continue;
        }
        const int bucket = depthBucket(projector.camera(), (a.depth + b.depth) * 0.5F);
        if (bucket != run) {
            flushRun(draw, scene, colour, run);
            run = bucket;
            pointScratch_.push_back(toImVec(a));
        }
        const float partial = std::min(drawn - static_cast<float>(s), 1.0F);
        const ImVec2 to{anim::lerp(a.x, b.x, partial), anim::lerp(a.y, b.y, partial)};
        pointScratch_.push_back(to);
        if (partial < 1.0F) {
            glowDot(draw, to, kPenRadius * scene.scale, withAlpha(colour, 1.0F));  // the "pen"
        }
    }
    flushRun(draw, scene, colour, run);
}

void HudRenderer::flushRun(ImDrawList& draw, const HudScene& scene, Rgba colour, int run) {
    if (pointScratch_.size() >= 2 && run >= 0) {
        draw.AddPolyline(pointScratch_.data(), static_cast<int>(pointScratch_.size()),
                         toU32(withAlpha(colour, depthShade(run))), scene.scale * kGlobeStroke,
                         ImDrawFlags_None);
    }
    pointScratch_.clear();
}

// --- satellites ----------------------------------------------------------------------

void HudRenderer::drawSatellites(ImDrawList& draw, const HudScene& scene) const {
    const auto& boot = scene.frame.boot;
    const float gate = boot.has_value() ? anim::smoothstep(0.55F, 1.0F, boot->globe) : 1.0F;
    const float strength = gate * intensity(scene) * scene.opacity;
    if (strength <= 0.0F) {
        return;
    }
    const anim::Camera& camera = scene.frame.camera;
    const anim::Projector projector{camera, globeViewport(scene)};
    const Rgba light = mixColour(accent(scene), theme_.wakeFlash, 0.35F);
    std::size_t index = 0;
    for (const anim::Orbit& orbit : anim::satelliteOrbits().first(anim::kSatelliteCount)) {
        const float angle = scene.frame.satelliteAngles.at(index++);
        const float direction = orbit.speed < 0.0F ? -1.0F : 1.0F;
        anim::Projected previous = projector(anim::orbitPosition(orbit, angle));
        const anim::Projected head = previous;
        for (int k = 1; k <= kTrailSamples; ++k) {
            const float back = angle - (direction * kTrailStep * static_cast<float>(k));
            const anim::Projected point = projector(anim::orbitPosition(orbit, back));
            const float fade = 1.0F - (static_cast<float>(k) / static_cast<float>(kTrailSamples));
            if (point.visible && previous.visible) {
                draw.AddLine(toImVec(previous), toImVec(point),
                             toU32(withAlpha(light, 0.6F * fade * fade * strength)), scene.scale);
            }
            previous = point;
        }
        if (head.visible) {
            const float front = frontness(camera, head.depth);
            glowDot(draw, toImVec(head),
                    (kSatelliteSize + (kSatelliteDepthSize * front)) * scene.scale,
                    withAlpha(light, (0.4F + (0.6F * front)) * strength));
        }
    }
}

// --- rings -----------------------------------------------------------------------------

void HudRenderer::drawRings(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts) const {
    const auto& boot = scene.frame.boot;
    const float spin = boot.has_value() ? boot->spin : 1.0F;
    const float glitch = boot.has_value() ? boot->glitch : 0.0F;
    const float radius = scene.geometry.radius;
    const float strength = intensity(scene) * scene.opacity;
    const float stroke = strokeScale(scene);
    const Rgba light = accent(scene);
    const auto step = static_cast<std::uint64_t>(scene.frame.time * 30.0F);
    std::size_t index = 0;
    for (const anim::RingLayer& layer : anim::ringLayers()) {
        const std::size_t i = index++;
        const anim::RingPose pose =
            anim::ringPose(boot.has_value() ? boot->ringArrival.at(i) : 1.0F);
        if (pose.alpha <= 0.0F) {
            continue;
        }
        const anim::Parallax shift = anim::parallax(scene.frame.camera, kRingDepth.at(i));
        const float shake = (anim::hashUnit(i, step) - 0.5F) * glitch * kLockShake * radius;
        const ImVec2 centre{scene.geometry.centreX + (shift.x * radius) + shake,
                            scene.geometry.centreY + (shift.y * radius)};
        const float rx = radius * layer.radius * pose.scale;
        const ImVec2 radii{rx, rx * std::cos(pose.tilt)};
        const float direction = layer.speed < 0.0F ? -1.0F : 1.0F;
        const float rotation =
            kTop + scene.frame.ringAngles.at(i) + (direction * kSpinBoost * spin);
        const Rgba colour = withAlpha(light, layer.alpha * strength * pose.alpha);
        segmentedEllipse(
            draw, centre, radii.x, radii.y, layer.segments, layer.fill, rotation,
            Stroke{.colour = colour, .thickness = layer.thickness * stroke, .glow = glow()});
        drawRingDetail(draw, scene, layer, centre, radii, rotation, colour);
        drawRingLabels(draw, scene, layer, centre, radii, rotation, colour, fonts);
    }
}

void HudRenderer::drawRingDetail(ImDrawList& draw, const HudScene& scene,
                                 const anim::RingLayer& layer, ImVec2 centre, ImVec2 radii,
                                 float rotation, Rgba colour) const {
    if (layer.ticks > 0) {
        const ImU32 packed = toU32(withAlpha(colour, 0.8F));
        const float step = kTau / static_cast<float>(layer.ticks);
        const int major = std::max(layer.majorEvery, 1);
        for (int t = 0; t < layer.ticks; ++t) {
            const float angle = rotation + (step * static_cast<float>(t));
            const float outer = t % major == 0 ? kTickMajor : kTickOuter;
            draw.AddLine(ellipsePoint(centre, radii.x * kTickInner, radii.y * kTickInner, angle),
                         ellipsePoint(centre, radii.x * outer, radii.y * outer, angle), packed,
                         scene.scale);
        }
    }
    if (layer.dashes > 0) {
        // Data streams: bright dashes flowing along the ring, faster than it turns.
        const float direction = layer.speed < 0.0F ? -1.0F : 1.0F;
        const float slot = kTau / static_cast<float>(layer.dashes);
        const Stroke dash{.colour = withAlpha(mixColour(colour, theme_.wakeFlash, 0.4F), 1.6F),
                          .thickness = 1.6F * scene.scale,
                          .glow = kSoftGlow};
        for (int d = 0; d < layer.dashes; ++d) {
            const float jitter = 0.35F * anim::hashUnit(static_cast<std::uint64_t>(d), 0);
            const float start = rotation + (direction * kDashFlow * scene.frame.streamPhase) +
                                (slot * (static_cast<float>(d) + jitter));
            glowEllipseArc(draw, centre, radii.x, radii.y, start, start + kDashSpan, dash);
        }
    }
}

void HudRenderer::drawRingLabels(ImDrawList& draw, const HudScene& scene,
                                 const anim::RingLayer& layer, ImVec2 centre, ImVec2 radii,
                                 float rotation, Rgba colour, const HudFonts& fonts) {
    if (!layer.labels || layer.ticks <= 0 || radii.x < kLabelMinRadius * scene.scale) {
        return;
    }
    const auto& boot = scene.frame.boot;
    const float countUp = boot.has_value() ? boot->labels : 1.0F;
    const int major = std::max(layer.majorEvery, 1);
    const auto majors = static_cast<std::size_t>(layer.ticks / major);
    const float size = scene.geometry.radius * kLabelSize;
    const float step = kTau / static_cast<float>(majors);
    std::array<char, 4> buffer{};
    for (std::size_t m = 0; m < majors; ++m) {
        const std::string_view label = threeDigits(anim::tickLabel(m, majors, countUp), buffer);
        const float angle = rotation + (step * static_cast<float>(m));
        const ImVec2 at =
            ellipsePoint(centre, radii.x * kLabelRadius, radii.y * kLabelRadius, angle);
        if (!boot.has_value() && at.x > centre.x + (radii.x * kLabelTextSide)) {
            continue;  // the right-hand labels would sit on the panel's text
        }
        const ImVec2 extent = fonts.display->CalcTextSizeA(size, FLT_MAX, 0.0F, label.data(),
                                                           label.data() + label.size());
        text(draw, *fonts.display, size, {at.x - (extent.x * 0.5F), at.y - (extent.y * 0.5F)},
             label, withAlpha(colour, 1.4F));
    }
}

// --- particle field -----------------------------------------------------------------------

void HudRenderer::drawField(ImDrawList& draw, const HudScene& scene) const {
    const anim::ParticleSystem& field = animator_.field();
    const std::span<const anim::SimParticle> particles = field.particles();
    const auto count = static_cast<std::size_t>(static_cast<float>(particles.size()) *
                                                clamp01(scene.frame.particleShare));
    if (count == 0) {
        return;
    }
    const anim::AudioEnergy& energy = scene.frame.energy;
    const float strength =
        intensity(scene) * scene.opacity * (kFieldAlpha + (kFieldEnergyAlpha * energy.level));
    const Rgba light = mixColour(accent(scene), theme_.wakeFlash, kFieldWhiten * energy.peak);
    const ImVec2 centre = centreOf(scene);
    const float radius = scene.geometry.radius;
    // In the panel the field must not wash over the text column to the right.
    const float panel = scene.frame.boot.has_value() ? scene.frame.boot->handoff : 1.0F;
    const float fadeFrom = centre.x + (kTextFadeStart * radius);
    const float fadeTo = centre.x + (kTextFadeEnd * radius);
    const int reserved = static_cast<int>(count);
    draw.PrimReserve(reserved * kQuadIndices, reserved * kQuadVertices);
    int used = 0;
    for (const anim::SimParticle& p : particles.first(count)) {
        const anim::Parallax shift = anim::parallax(scene.frame.camera, p.depth);
        const ImVec2 at{centre.x + ((p.x + shift.x) * radius),
                        centre.y + ((p.y + shift.y) * radius)};
        const float clear = 1.0F - (panel * anim::smoothstep(fadeFrom, fadeTo, at.x));
        const float alpha = field.alpha(p) * strength * clear;
        if (alpha < kMinDotAlpha) {
            continue;
        }
        const float half =
            0.5F * kDotSize * p.size * (1.0F + (kDotDepthSize * p.depth)) * scene.scale;
        draw.PrimRect({at.x - half, at.y - half}, {at.x + half, at.y + half},
                      toU32(withAlpha(light, alpha)));
        ++used;
    }
    draw.PrimUnreserve((reserved - used) * kQuadIndices, (reserved - used) * kQuadVertices);
}

}  // namespace jarvis::overlay::hud

// Design notes:
// - The globe is projected point by point every frame (a few hundred
//   points) instead of caching screen geometry: the camera drifts, the globe
//   turns and pulses, so nothing would survive to the next frame anyway.
// - Depth is shown three ways that cost nothing on the GPU: back lines of
//   the globe dim, near satellites are larger and brighter, and the ring
//   layers and particles shift with the camera by their depth (parallax).
// - Particles are single quads written straight into the draw list
//   (PrimReserve/PrimRect): four vertices each instead of the dozens of a
//   filled circle, and the bloom pass turns the squares into soft points.
