#include "jarvis/overlay/orb.hpp"

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/animator.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/placement.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace jarvis::overlay {

namespace {

// Morph phases, as shares of the whole morph.
constexpr float kPanelGone = 0.35F;
constexpr float kSceneFadeStart = 0.55F;
constexpr float kSceneFadeEnd = 0.95F;
constexpr float kOrbFadeStart = 0.45F;
constexpr float kOrbFadeEnd = 0.9F;

/// Radians per second of the thinking orbit (before the intensity's motion).
constexpr float kOrbitSpeed = 2.8F;
constexpr float kThinkingFlicker = 4.0F;

/// One state's contribution to the look.
struct StateLook {
    float core;
    float pulse;
    float rim;
};

int along(float anchor, int side) noexcept {
    return static_cast<int>(std::lround(anchor - (static_cast<float>(side) / 2.0F)));
}

}  // namespace

DockEdge nearestEdge(Placement panel, PixelRect area) noexcept {
    const std::array<int, 4> distances{
        panel.x - area.left,                    // Left
        area.right - (panel.x + panel.width),   // Right
        panel.y - area.top,                     // Top
        area.bottom - (panel.y + panel.height)  // Bottom
    };
    constexpr std::array<DockEdge, 4> kEdges{DockEdge::Left, DockEdge::Right, DockEdge::Top,
                                             DockEdge::Bottom};
    std::size_t best = 0;
    for (std::size_t i = 1; i < distances.size(); ++i) {
        if (distances.at(i) < distances.at(best)) {
            best = i;
        }
    }
    return kEdges.at(best);
}

int orbSide(float scale) noexcept {
    return scaled(kOrbDiameter, scale);
}

Placement squareAt(float centreX, float centreY, int side) noexcept {
    return Placement{
        .x = along(centreX, side), .y = along(centreY, side), .width = side, .height = side};
}

Placement dockOrb(Placement panel, float anchorX, float anchorY,
                  const MonitorArea& monitor) noexcept {
    const PixelRect& work = monitor.work;
    const int side = orbSide(monitor.scale);
    const int margin = scaled(kOrbMargin, monitor.scale);
    const int reach = scaled(kOrbCornerReach, monitor.scale);
    const int toLeft = panel.x - work.left;
    const int toRight = work.right - (panel.x + panel.width);
    const int toTop = panel.y - work.top;
    const int toBottom = work.bottom - (panel.y + panel.height);
    const int leftX = work.left + margin;
    const int rightX = work.right - margin - side;
    const int topY = work.top + margin;
    const int bottomY = work.bottom - margin - side;
    Placement orb{
        .x = along(anchorX, side), .y = along(anchorY, side), .width = side, .height = side};
    if (std::min(toLeft, toRight) <= reach && std::min(toTop, toBottom) <= reach) {
        orb.x = toLeft <= toRight ? leftX : rightX;
        orb.y = toTop <= toBottom ? topY : bottomY;
        return clampInto(orb, work);
    }
    switch (nearestEdge(panel, work)) {
        case DockEdge::Left:
            orb.x = leftX;
            break;
        case DockEdge::Right:
            orb.x = rightX;
            break;
        case DockEdge::Top:
            orb.y = topY;
            break;
        case DockEdge::Bottom:
            orb.y = bottomY;
            break;
    }
    return clampInto(orb, work);
}

OrbMorph orbMorphAt(float t) noexcept {
    const float linear = anim::clamp01(t);
    return OrbMorph{
        .travel = anim::easeInOutCubic(linear),
        .panelAlpha = 1.0F - anim::smoothstep(0.0F, kPanelGone, linear),
        .sceneAlpha = 1.0F - anim::smoothstep(kSceneFadeStart, kSceneFadeEnd, linear),
        .orbAlpha = anim::smoothstep(kOrbFadeStart, kOrbFadeEnd, linear),
    };
}

OrbLook orbLook(const anim::AnimationFrame& frame) noexcept {
    const float level = anim::clamp01(frame.energy.level);
    const float peak = anim::clamp01(frame.energy.peak);
    const float onset = anim::clamp01(frame.energy.onset);
    const float breath = anim::clamp01(frame.breathing);
    const std::array<StateLook, 4> looks{{
        // Idle: a slow, dim breath; it is on screen all day.
        {.core = 0.35F + (0.3F * breath),
         .pulse = 0.9F + (0.08F * breath),
         .rim = 0.35F + (0.15F * breath)},
        // Listening: bright, swelling with the user's voice.
        {.core = 0.75F + (0.25F * level), .pulse = 0.95F + (0.35F * level), .rim = 0.9F},
        // Thinking: steady core, the orbit carries the motion.
        {.core = 0.55F + (0.1F * std::sin(frame.time * kThinkingFlicker)),
         .pulse = 0.95F,
         .rim = 0.6F},
        // Speaking: follows Jarvis's voice, punched by syllable onsets.
        {.core = 0.6F + (0.4F * std::max(level, peak * 0.8F)),
         .pulse = 0.95F + (0.3F * level) + (0.15F * onset),
         .rim = 0.8F},
    }};
    OrbLook look{.core = 0.0F, .pulse = 0.0F, .rim = 0.0F};
    for (std::size_t i = 0; i < looks.size(); ++i) {
        const float weight = frame.weights.at(i);
        look.core += looks.at(i).core * weight;
        look.pulse += looks.at(i).pulse * weight;
        look.rim += looks.at(i).rim * weight;
    }
    look.core = anim::clamp01(look.core);
    look.rim = anim::clamp01(look.rim);
    look.orbit = frame.weights.at(anim::stateIndex(AssistantState::Thinking));
    look.orbitAngle = std::fmod(frame.time * kOrbitSpeed * frame.motion, anim::kTau);
    look.rippleRadius = frame.ripple.radius;
    look.rippleAlpha = frame.ripple.alpha;
    return look;
}

}  // namespace jarvis::overlay

// Design notes:
// - The orb is centred on the HUD's ring along the docking edge, not on the
//   panel: the morph flies the ring into the orb, and a ring that ends up
//   where it started (only smaller) reads as "minimized" rather than as a
//   second widget appearing somewhere else.
// - A corner wins over a single edge: a HUD in the bottom-right corner
//   leaves its orb in that corner, where a user who snapped it there wants
//   it, instead of half-way along the bottom edge.
// - The morph is one curve of linear progress, evaluated forwards to
//   minimize and backwards to restore. The panel's text is gone before the
//   ring has moved a third of the way, and the orb only fades in once the
//   shrinking ring is about its size, so no frame shows a flat rectangle.
// - The look is weighted by the same state cross-fade weights as the full
//   HUD, so a state change on the orb blends over the same 200 ms.
