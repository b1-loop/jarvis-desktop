#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/hud_layout.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"

#include <imgui.h>

#include "hud.hpp"
#include "hud_draw.hpp"
#include "hud_util.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <string_view>

namespace jarvis::overlay::hud {

namespace {

constexpr std::array<AssistantState, 4> kStates{AssistantState::Idle, AssistantState::Listening,
                                                AssistantState::Thinking, AssistantState::Speaking};

// The expanded scrim: a dark core inside feathered layers, never a box.
constexpr int kFeatherLayers = 9;
constexpr float kFeatherStep = 5.0F;
constexpr float kFeatherAlpha = 0.14F;
constexpr float kCoreAlpha = 0.62F;
constexpr float kRounding = 16.0F;
constexpr float kBracketLength = 26.0F;
constexpr float kCaptionSize = 20.0F;
constexpr float kCaptionTracking = 3.0F;
constexpr float kClockSize = 13.0F;
constexpr float kClockTracking = 1.6F;
constexpr float kBrandSize = 10.0F;
constexpr std::string_view kBrand = "J.A.R.V.I.S";

/// How far the panel's content has faded in: only over the last 40 % of
/// the morph, once the frame has nearly reached its size.
float contentAlpha(const HudScene& scene) noexcept {
    return anim::clamp01((scene.expand - 0.6F) / 0.4F) * scene.opacity;
}

}  // namespace

void HudRenderer::drawExpandedScrim(ImDrawList& draw, const HudScene& scene) const {
    const float alpha = anim::clamp01(scene.expand * 1.5F) * scene.opacity;
    if (alpha <= kNegligible) {
        return;
    }
    const RectF& frame = scene.panelFrame;
    const Rgba plate{.r = theme_.plate.r, .g = theme_.plate.g, .b = theme_.plate.b};
    const float step = kFeatherStep * scene.scale;
    const ImU32 layer = toU32(withAlpha(plate, kFeatherAlpha * alpha));
    for (int i = 0; i < kFeatherLayers; ++i) {
        // Grows inwards: the outermost layer is the frame, the edge stays soft.
        const float inset = step * static_cast<float>(i);
        draw.AddRectFilled({frame.x + inset, frame.y + inset},
                           {frame.right() - inset, frame.bottom() - inset}, layer,
                           std::max((kRounding * scene.scale) - inset, 0.0F));
    }
    const float core = step * static_cast<float>(kFeatherLayers);
    draw.AddRectFilled({frame.x + core, frame.y + core},
                       {frame.right() - core, frame.bottom() - core},
                       toU32(withAlpha(plate, kCoreAlpha * alpha)));
}

void HudRenderer::drawExpandedLight(ImDrawList& draw, const HudScene& scene) const {
    if (scene.expand <= kNegligible) {
        return;
    }
    const RectF& frame = scene.panelFrame;
    const Rgba light = accent(scene);
    const float alpha = anim::clamp01(scene.expand * 2.0F) * scene.opacity;
    cornerBrackets(draw, frame, kBracketLength * scene.scale, withAlpha(light, 0.85F * alpha),
                   1.5F * scene.scale);
    const ExpandedLayout layout = expandedLayout(frame, scene.scale);
    const float divider = layout.tabConversation.y - (4.0F * scene.scale);
    const float content = contentAlpha(scene);
    draw.AddLine({layout.body.x, divider}, {layout.body.right(), divider},
                 toU32(withAlpha(light, 0.35F * content)), scene.scale);
    if (scene.expand < 1.0F) {
        // The unfold: a bright scan line sweeps down the frame as it opens.
        const float y = frame.y + (frame.height * scene.expand);
        const float sweep = std::sin(kPi * scene.expand) * scene.opacity;
        glowLine(draw, {frame.x, y}, {frame.right(), y},
                 Stroke{.colour = withAlpha(theme_.wakeFlash, 0.7F * sweep),
                        .thickness = 1.5F * scene.scale,
                        .glow = glow()});
    }
}

void HudRenderer::drawExpandedCaption(ImDrawList& draw, const HudScene& scene,
                                      const HudFonts& fonts) const {
    const float alpha = contentAlpha(scene);
    if (alpha <= kNegligible) {
        return;
    }
    const ExpandedLayout layout = expandedLayout(scene.panelFrame, scene.scale);
    const RectF& caption = layout.caption;
    const float size = kCaptionSize * scene.scale;
    const float top = caption.y + (caption.height * 0.22F);
    for (const AssistantState state : kStates) {
        const float weight = weightOf(scene, state);
        if (weight >= kNegligible) {
            spacedText(draw, *fonts.display, size, {caption.x, top}, hudLabel(state),
                       kCaptionTracking * scene.scale,
                       withAlpha(mixColour(stateColour(theme_, state), theme_.text, 0.35F),
                                 weight * alpha));
        }
    }
    const float clock = kClockSize * scene.scale;
    const std::string line = scene.clock.time + "   " + scene.clock.date;
    spacedText(draw, *fonts.display, clock, {caption.x, top + (size * 1.5F)}, line,
               kClockTracking * scene.scale, withAlpha(theme_.mutedText, alpha));
    const float brand = kBrandSize * scene.scale;
    spacedText(draw, *fonts.display, brand, {caption.x, top + (size * 1.5F) + (clock * 1.7F)},
               kBrand, kClockTracking * scene.scale, withAlpha(accent(scene), 0.7F * alpha));
}

}  // namespace jarvis::overlay::hud

// Design notes:
// - The morph is drawing, not a window animation: on the transparent
//   surface the window spans both rectangles for its duration, so the
//   frame grows, the ring flies into the header and the scan line sweeps
//   without a single swap-chain resize (the ADR-0023 hand-off technique).
// - Order of the reveal: the corner panel's text fades out over the first
//   half, the frame and brackets grow with the eased progress, and the
//   panel's own content only fades in over the last 40 %, so text is
//   never drawn over a frame that is still moving.
// - The scrim's layers grow inwards from the frame, so the dark core is
//   readable under the history while the edge stays feathered; there is
//   still no border rectangle, only light (brackets, divider) for shape.
