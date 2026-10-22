#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/orb.hpp"

#include <imgui.h>

#include "hud.hpp"
#include "hud_draw.hpp"
#include "hud_util.hpp"

#include <algorithm>

namespace jarvis::overlay::hud {

namespace {

// Sizes as shares of the orb's radius.
constexpr float kHaloRadius = 1.55F;
constexpr float kBodyRadius = 0.8F;
constexpr float kHotRadius = 0.22F;
constexpr float kRimRadius = 0.86F;
constexpr float kScrimRadius = 1.05F;
/// Length of the thinking orbit's trailing arc, radians.
constexpr float kOrbitTrail = 1.4F;

// Brightness of each layer at full look.
constexpr float kHaloAlpha = 0.22F;
constexpr float kBodyAlpha = 0.55F;
constexpr float kScrimAlpha = 0.6F;
/// The rim brightens this much while the cursor is on the orb.
constexpr float kHoverRim = 0.4F;

}  // namespace

void HudRenderer::drawOrbFrame(ImDrawList& draw, const HudScene& scene) const {
    drawOrbScrim(draw, scene, scene.opacity);
    blend_.beginLight(draw);
    drawOrbLight(draw, scene, scene.opacity);
    blend_.endLight(draw);
}

void HudRenderer::drawOrbScrim(ImDrawList& draw, const HudScene& scene, float alpha) const {
    if (alpha <= kNegligible) {
        return;
    }
    // A dark disc under the light so the orb reads on a white page too.
    const Rgba plate{.r = theme_.plate.r, .g = theme_.plate.g, .b = theme_.plate.b};
    radialGlow(draw, centreOf(scene), scene.geometry.radius * kScrimRadius,
               withAlpha(plate, kScrimAlpha * alpha));
}

void HudRenderer::drawOrbLight(ImDrawList& draw, const HudScene& scene, float alpha) const {
    if (alpha <= kNegligible) {
        return;
    }
    const OrbLook look = orbLook(scene.frame);
    const ImVec2 centre = centreOf(scene);
    const float radius = scene.geometry.radius;
    const Rgba light = accent(scene);
    const float core = look.core * alpha;
    radialGlow(draw, centre, radius * kHaloRadius, withAlpha(light, kHaloAlpha * core));
    radialGlow(draw, centre, radius * kBodyRadius * look.pulse,
               withAlpha(light, kBodyAlpha * core));
    glowDot(draw, centre, radius * kHotRadius * look.pulse,
            withAlpha(mixColour(light, theme_.wakeFlash, 0.5F), core));
    const float rim = std::min(look.rim + (kHoverRim * scene.hover), 1.0F) * alpha;
    const Stroke rimStroke{
        .colour = withAlpha(light, rim), .thickness = 1.4F * scene.scale, .glow = glow()};
    glowCircle(draw, centre, radius * kRimRadius, rimStroke);
    if (look.orbit > kNegligible) {
        const float head = kTop + look.orbitAngle;
        glowArc(draw, centre, radius * kRimRadius, head - kOrbitTrail, head,
                Stroke{.colour = withAlpha(light, 0.6F * look.orbit * alpha),
                       .thickness = 2.0F * scene.scale,
                       .glow = kSoftGlow});
        glowDot(draw, polar(centre, radius * kRimRadius, head), 2.0F * scene.scale,
                withAlpha(theme_.wakeFlash, look.orbit * alpha));
    }
    if (look.rippleAlpha > 0.0F) {
        glowCircle(draw, centre, radius * look.rippleRadius,
                   Stroke{.colour = withAlpha(theme_.wakeFlash, look.rippleAlpha * alpha),
                          .thickness = 1.5F * scene.scale,
                          .glow = kSoftGlow});
    }
}

}  // namespace jarvis::overlay::hud

// Design notes:
// - The orb is a handful of glows (halo, body, hot core, rim) rather than a
//   miniature of the HUD: at 48 logical pixels a globe or rings would be
//   noise, and the whole frame is a few dozen vertices, which keeps the
//   idle orb's cost near the bare cost of presenting a frame.
// - It is drawn in the same light section as the HUD, so it is bloomed by
//   the same pipeline and takes the same state colour, and mid-morph both
//   pictures add up like light instead of one covering the other.
// - The rim brightens under the cursor: the orb has no tooltip (its window
//   is too small for one), so that is the cue that it is clickable.
