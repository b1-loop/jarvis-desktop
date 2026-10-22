#include "jarvis/overlay/hud_layout.hpp"
#include "jarvis/overlay/layout.hpp"

#include <imgui.h>

#include "hud_draw.hpp"
#include "hud_interact.hpp"

#include <array>
#include <numbers>

namespace jarvis::overlay::hud {

namespace {

constexpr float kPi = std::numbers::pi_v<float>;
constexpr int kGearTeeth = 8;

/// A point of @p box given as shares of its width and height.
ImVec2 at(RectF box, float u, float v) noexcept {
    return ImVec2{box.x + (box.width * u), box.y + (box.height * v)};
}

void keyboard(ImDrawList& draw, RectF box, ImU32 colour, float thickness) {
    draw.AddRect(at(box, 0.0F, 0.18F), at(box, 1.0F, 0.82F), colour, box.width * 0.1F, thickness);
    constexpr std::array<float, 4> kColumns{0.2F, 0.4F, 0.6F, 0.8F};
    for (const float u : kColumns) {
        draw.AddCircleFilled(at(box, u, 0.38F), thickness * 0.7F, colour);
        draw.AddCircleFilled(at(box, u, 0.52F), thickness * 0.7F, colour);
    }
    draw.AddLine(at(box, 0.3F, 0.68F), at(box, 0.7F, 0.68F), colour, thickness);
}

void microphone(ImDrawList& draw, RectF box, ImU32 colour, float thickness, bool muted) {
    draw.AddRect(at(box, 0.34F, 0.0F), at(box, 0.66F, 0.58F), colour, box.width * 0.16F, thickness);
    draw.PathArcTo(at(box, 0.5F, 0.42F), box.width * 0.32F, 0.0F, kPi);
    draw.PathStroke(colour, thickness);
    draw.AddLine(at(box, 0.5F, 0.74F), at(box, 0.5F, 0.92F), colour, thickness);
    draw.AddLine(at(box, 0.3F, 0.92F), at(box, 0.7F, 0.92F), colour, thickness);
    if (muted) {
        draw.AddLine(at(box, 0.05F, 0.05F), at(box, 0.95F, 0.95F), colour, thickness * 1.3F);
    }
}

void gear(ImDrawList& draw, RectF box, ImU32 colour, float thickness) {
    const ImVec2 centre = at(box, 0.5F, 0.5F);
    const float radius = box.width * 0.5F;
    draw.AddCircle(centre, radius * 0.62F, colour, 0, thickness);
    draw.AddCircle(centre, radius * 0.24F, colour, 0, thickness);
    for (int i = 0; i < kGearTeeth; ++i) {
        const float angle = 2.0F * kPi * static_cast<float>(i) / kGearTeeth;
        draw.AddLine(polar(centre, radius * 0.66F, angle), polar(centre, radius, angle), colour,
                     thickness * 1.6F);
    }
}

/// Two corner arrows pointing out (expand) or in (collapse).
void expandArrows(ImDrawList& draw, RectF box, ImU32 colour, float thickness, bool inward) {
    const float tip = inward ? 0.58F : 1.0F;
    const float base = inward ? 1.0F : 0.58F;
    const float head = 0.3F;
    for (const float sign : {1.0F, -1.0F}) {
        const auto point = [&](float u, float v) {
            return sign > 0.0F ? at(box, u, 1.0F - v) : at(box, 1.0F - u, v);
        };
        draw.AddLine(point(base, base), point(tip, tip), colour, thickness);
        const float back = inward ? tip + head : tip - head;
        draw.AddLine(point(tip, tip), point(back, tip), colour, thickness);
        draw.AddLine(point(tip, tip), point(tip, back), colour, thickness);
    }
}

void power(ImDrawList& draw, RectF box, ImU32 colour, float thickness) {
    constexpr float kGap = 0.7F;
    draw.PathArcTo(at(box, 0.5F, 0.55F), box.width * 0.42F, (-kPi * 0.5F) + kGap,
                   (kPi * 1.5F) - kGap);
    draw.PathStroke(colour, thickness);
    draw.AddLine(at(box, 0.5F, 0.0F), at(box, 0.5F, 0.45F), colour, thickness);
}

/// A bar along the bottom with a small orb above it: "shrink to the orb".
void minimize(ImDrawList& draw, RectF box, ImU32 colour, float thickness) {
    draw.AddLine(at(box, 0.12F, 0.86F), at(box, 0.88F, 0.86F), colour, thickness);
    draw.AddCircle(at(box, 0.5F, 0.42F), box.width * 0.2F, colour, 0, thickness);
}

void cross(ImDrawList& draw, RectF box, ImU32 colour, float thickness) {
    draw.AddLine(at(box, 0.1F, 0.1F), at(box, 0.9F, 0.9F), colour, thickness);
    draw.AddLine(at(box, 0.9F, 0.1F), at(box, 0.1F, 0.9F), colour, thickness);
}

}  // namespace

void drawControlIcon(ImDrawList& draw, HudControl control, RectF rect, Rgba colour, float thickness,
                     bool muted, bool expanded) {
    const ImU32 packed = toU32(colour);
    switch (control) {
        case HudControl::Keyboard:
            keyboard(draw, rect, packed, thickness);
            return;
        case HudControl::Mute:
            microphone(draw, rect, packed, thickness, muted);
            return;
        case HudControl::Settings:
            gear(draw, rect, packed, thickness);
            return;
        case HudControl::Expand:
            expandArrows(draw, rect, packed, thickness, expanded);
            return;
        case HudControl::Quit:
            power(draw, rect, packed, thickness);
            return;
        case HudControl::Minimize:
            minimize(draw, rect, packed, thickness);
            return;
        case HudControl::Close:
            break;
    }
    cross(draw, rect, packed, thickness);
}

}  // namespace jarvis::overlay::hud

// Design notes:
// - Glyphs are a handful of vector strokes in the HUD's light rather than
//   an icon font: no extra asset to ship, they scale with the DPI, and
//   they take the state colour like every other stroke of the instrument.
