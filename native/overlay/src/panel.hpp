/// @file panel.hpp
/// @brief Dear ImGui draw calls for one frame of the overlay (private).
///
/// Only window.cpp includes this. The decisions about what to show live in
/// layout.hpp; this layer positions and paints them.

#pragma once

#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"

namespace jarvis::overlay {

/// Pixel sizes of the panel's parts at a given DPI scale.
struct PanelMetrics {
    float scale{1.0F};
    float padding{14.0F};
    float dotRadius{6.0F};
    float barWidth{3.0F};
    float barGap{2.0F};
    float waveformHeight{30.0F};
    float rowGap{6.0F};
    float ringThickness{2.0F};

    /// Metrics for a 96 DPI design multiplied by @p scale.
    [[nodiscard]] static PanelMetrics forScale(float scale) noexcept;
};

/// Paint @p snapshot over the whole ImGui display area. Must run between
/// ImGui::NewFrame() and ImGui::Render() on the UI thread.
void drawPanel(const OverlaySnapshot& snapshot, const Theme& theme, const PanelMetrics& metrics);

}  // namespace jarvis::overlay
