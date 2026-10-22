/// @file hud.hpp
/// @brief Draws one frame of the HUD: boot sequence and corner panel (private).
///
/// Only window.cpp includes this. Every time-dependent number arrives in an
/// anim::AnimationFrame and every position in a HudGeometry; this layer
/// decides colours, stroke widths and draw order, and nothing else. The
/// member functions are spread over three files: hud.cpp (frame order,
/// state layers, plate and panel text), hud_scene.cpp (globe, rings,
/// satellites, particle field) and hud_boot.cpp (boot-only elements).

#pragma once

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/animator.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/scene.hpp"

#include <imgui.h>

#include "hud_draw.hpp"
#include "postfx.hpp"
#include <d3d11.h>

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::overlay::hud {

/// The two fonts the HUD uses; both owned by ImGui's font atlas.
struct HudFonts {
    /// Transcript and reply text.
    ImFont* body{nullptr};
    /// Captions and the boot title (Bahnschrift when Windows has it).
    ImFont* display{nullptr};
};

/// The wall clock as the idle readout shows it.
struct HudClock {
    std::string time;
    std::string date;
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
    /// Multiplier of the panel's typography (PanelMetrics::textScale).
    float textScale{1.0F};
    /// Readout lines the renderer itself knows (device, surface), shown
    /// after the ones from the snapshot.
    std::span<const ReadoutLine> systemReadout;
    /// Local time for the idle readout.
    HudClock clock;
};

/// Stateless apart from fixed resources (blend callbacks, the globe
/// geometry and scratch buffers reused every frame so drawing allocates
/// nothing after the first frames).
class HudRenderer {
public:
    /// @p post is the post-processing pipeline, or null to draw without bloom.
    /// @throws OverlayError if the additive blend state cannot be created.
    HudRenderer(ID3D11Device& device, Theme theme, const anim::OverlayAnimator& animator,
                PostProcessor* post);

    /// Append the frame to @p draw.
    void draw(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts);

    [[nodiscard]] const Theme& theme() const noexcept { return theme_; }
    /// True when light is bloomed by the GPU pipeline.
    [[nodiscard]] bool bloom() const noexcept { return blend_.bloom(); }

private:
    // hud.cpp: scrims, core, state layers, ripple, panel text.
    void drawScrim(ImDrawList& draw, const HudScene& scene) const;
    void drawConnector(ImDrawList& draw, const HudScene& scene) const;
    void drawIdleReadout(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts,
                         float alpha) const;
    void drawTranscript(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts,
                        float alpha) const;
    void drawInfoLine(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts, ImVec2 at,
                      std::string_view label, std::string_view value, float alpha) const;
    void drawTelemetry(ImDrawList& draw, const HudScene& scene);
    void drawCore(ImDrawList& draw, const HudScene& scene) const;
    void drawListening(ImDrawList& draw, const HudScene& scene);
    void drawThinking(ImDrawList& draw, const HudScene& scene) const;
    void drawSpeaking(ImDrawList& draw, const HudScene& scene);
    void drawRipple(ImDrawList& draw, const HudScene& scene) const;
    void drawPanelText(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts) const;
    void drawCaption(ImDrawList& draw, const HudScene& scene, ImFont& font, float alpha) const;
    void drawDivider(ImDrawList& draw, const HudScene& scene, float y, float alpha) const;

    // hud_scene.cpp: the holographic scene shared by boot and panel.
    void drawBackdropGlow(ImDrawList& draw, const HudScene& scene) const;
    void drawGlobe(ImDrawList& draw, const HudScene& scene);
    void drawGlobeLine(ImDrawList& draw, const HudScene& scene, const anim::Projector& projector,
                       const anim::GlobeLine& line, float reveal, Rgba colour);
    /// Stroke the points collected in pointScratch_ at depth bucket @p run, then clear them.
    void flushRun(ImDrawList& draw, const HudScene& scene, Rgba colour, int run);
    void drawSatellites(ImDrawList& draw, const HudScene& scene) const;
    void drawRings(ImDrawList& draw, const HudScene& scene, const HudFonts& fonts) const;
    void drawRingDetail(ImDrawList& draw, const HudScene& scene, const anim::RingLayer& layer,
                        ImVec2 centre, ImVec2 radii, float rotation, Rgba colour) const;
    static void drawRingLabels(ImDrawList& draw, const HudScene& scene,
                               const anim::RingLayer& layer, ImVec2 centre, ImVec2 radii,
                               float rotation, Rgba colour, const HudFonts& fonts);
    void drawField(ImDrawList& draw, const HudScene& scene) const;

    // hud_boot.cpp: elements that only exist while booting.
    void drawIgnition(ImDrawList& draw, const HudScene& scene, const anim::BootFrame& boot) const;
    void drawBootStream(ImDrawList& draw, const HudScene& scene, const anim::BootFrame& boot);
    void drawScanLine(ImDrawList& draw, const HudScene& scene, const anim::BootFrame& boot) const;
    void drawGlyphRuns(ImDrawList& draw, const HudScene& scene, const anim::BootFrame& boot,
                       const HudFonts& fonts) const;
    void drawBootTitle(ImDrawList& draw, const HudScene& scene, const anim::BootFrame& boot,
                       const HudFonts& fonts) const;
    void drawReadout(ImDrawList& draw, const HudScene& scene, const anim::BootFrame& boot,
                     const HudFonts& fonts) const;

    [[nodiscard]] Rgba accent(const HudScene& scene) const noexcept;
    [[nodiscard]] static float intensity(const HudScene& scene) noexcept;
    [[nodiscard]] static float panelAlpha(const HudScene& scene) noexcept;
    /// Glow passes for strokes: fewer when the bloom pass adds the halo.
    [[nodiscard]] std::span<const GlowPass> glow() const noexcept;

    Theme theme_;
    const anim::OverlayAnimator& animator_;
    BlendModes blend_;
    std::vector<anim::GlobeLine> globe_;
    std::vector<anim::ParticlePoint> particleScratch_;
    std::vector<float> spokeScratch_;
    std::vector<ImVec2> pointScratch_;
    std::vector<anim::Projected> projectedScratch_;
};

}  // namespace jarvis::overlay::hud
