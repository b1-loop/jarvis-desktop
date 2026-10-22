/// @file hud.hpp
/// @brief Draws one frame of the HUD: boot sequence and corner panel (private).
///
/// Only window.cpp includes this. Every time-dependent number arrives in an
/// anim::AnimationFrame and every position in a HudGeometry; this layer
/// decides colours, stroke widths and draw order, and nothing else.

#pragma once

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"

#include <imgui.h>

#include "hud_draw.hpp"
#include <d3d11.h>

#include <vector>

namespace jarvis::overlay::hud {

/// The two fonts the HUD uses; both owned by ImGui's font atlas.
struct HudFonts {
    /// Transcript and reply text.
    ImFont* body{nullptr};
    /// Captions and the boot title (Bahnschrift when Windows has it).
    ImFont* display{nullptr};
};

/// Everything one frame needs, in the window's client coordinates.
struct HudScene {
    const OverlaySnapshot& snapshot;
    const anim::AnimationFrame& frame;
    /// Ring centre, radius and text area for this frame (mid-handoff while booting).
    HudGeometry geometry;
    /// Where the corner panel is (its final position, also during boot).
    RectF panel;
    /// The panel's text column (its final position, also during boot).
    RectF panelText;
    /// Boot canvas for this frame; only meaningful while frame.boot is set.
    RectF canvas;
    /// Monitor DPI scale (1 at 96 DPI).
    float scale{1.0F};
    /// Global opacity in [0, 1] (the configured opacity on transparent surfaces).
    float opacity{1.0F};
};

/// A fragment of the hex grid that flickers in during boot.
struct HexFragment {
    float angle{0.0F};
    /// Distance from the centre in ring radii.
    float distance{0.0F};
    /// Circumradius in ring radii.
    float size{0.0F};
    float rotation{0.0F};
};

/// Stateless apart from fixed resources (blend state, the hex layout and a
/// scratch buffer reused every frame so drawing allocates nothing).
class HudRenderer {
public:
    /// @throws OverlayError if the additive blend state cannot be created.
    HudRenderer(ID3D11Device& device, Theme theme, const anim::OverlayAnimator& animator);

    /// Append the frame to @p draw.
    void draw(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts);

    [[nodiscard]] const Theme& theme() const noexcept { return theme_; }

private:
    void drawPlate(ImDrawList& draw, const HudScene& scene) const;
    void drawBootBackdrop(ImDrawList& draw, const HudScene& scene, const anim::BootFrame& boot);
    void drawHexGrid(ImDrawList& draw, const HudScene& scene, const anim::BootFrame& boot) const;
    void drawParticles(ImDrawList& draw, const HudScene& scene, const anim::BootFrame& boot);
    void drawScanLine(ImDrawList& draw, const HudScene& scene, const anim::BootFrame& boot) const;
    void drawReactor(ImDrawList& draw, const HudScene& scene) const;
    void drawCore(ImDrawList& draw, const HudScene& scene) const;
    void drawListening(ImDrawList& draw, const HudScene& scene);
    void drawThinking(ImDrawList& draw, const HudScene& scene) const;
    void drawSpeaking(ImDrawList& draw, const HudScene& scene);
    void drawRipple(ImDrawList& draw, const HudScene& scene) const;
    void drawBootTitle(ImDrawList& draw, const HudScene& scene, const anim::BootFrame& boot,
                       const HudFonts& fonts) const;
    void drawPanelText(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts) const;
    void drawCaption(ImDrawList& draw, const HudScene& scene, ImFont& font, float alpha) const;
    void drawDivider(ImDrawList& draw, const HudScene& scene, float y, float alpha) const;

    [[nodiscard]] Rgba accent(const HudScene& scene) const noexcept;
    [[nodiscard]] static float intensity(const HudScene& scene) noexcept;
    [[nodiscard]] static float panelAlpha(const HudScene& scene) noexcept;

    Theme theme_;
    const anim::OverlayAnimator& animator_;
    BlendModes blend_;
    std::vector<HexFragment> hexes_;
    std::vector<anim::ParticlePoint> particleScratch_;
    std::vector<float> spokeScratch_;
    std::vector<ImVec2> pointScratch_;
};

}  // namespace jarvis::overlay::hud
