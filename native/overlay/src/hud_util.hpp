/// @file hud_util.hpp
/// @brief Small helpers shared by the three HUD translation units (private).

#pragma once

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/model.hpp"

#include <imgui.h>

#include "hud.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace jarvis::overlay::hud {

inline constexpr float kPi = std::numbers::pi_v<float>;
/// Angles start at twelve o'clock.
inline constexpr float kTop = -kPi / 2.0F;
/// Layers whose weight is below this are not drawn at all.
inline constexpr float kNegligible = 0.005F;
/// Stroke widths are designed for the panel ring (about 52 px at 96 DPI);
/// the larger boot ring thickens them a little, not proportionally.
inline constexpr float kReferenceRadius = 52.0F;
inline constexpr float kMaxStrokeGrowth = 2.2F;

inline ImVec2 centreOf(const HudScene& scene) noexcept {
    return ImVec2{scene.geometry.centreX, scene.geometry.centreY};
}

/// Stroke width multiplier for the current ring size.
inline float strokeScale(const HudScene& scene) noexcept {
    const float reference = kReferenceRadius * scene.scale;
    const float growth = reference > 0.0F ? scene.geometry.radius / reference : 1.0F;
    return scene.scale * std::clamp(growth, 1.0F, kMaxStrokeGrowth);
}

inline float weightOf(const HudScene& scene, AssistantState state) noexcept {
    return scene.frame.weights.at(anim::stateIndex(state));
}

inline float fraction(float value) noexcept {
    return value - std::floor(value);
}

}  // namespace jarvis::overlay::hud
