/// @file hud_draw.hpp
/// @brief Glowing HUD primitives on top of ImDrawList (private).
///
/// Everything that is light is drawn between BlendModes::beginLight() and
/// endLight(): with the GPU post-processing pipeline (postfx.hpp) that
/// section goes into a half-float light layer that is bloomed and
/// composited; without it (the pipeline failed to start) it is drawn
/// additively straight onto the swap chain and the strokes carry their own
/// glow in layered passes, the ADR-0022 look. Inside the section
/// overlapping light adds up towards white the way light does.

#pragma once

#include "jarvis/overlay/layout.hpp"

#include <imgui.h>

#include "postfx.hpp"
#include <d3d11.h>
#include <wrl/client.h>

#include <array>
#include <span>
#include <string_view>

namespace jarvis::overlay::hud {

/// One pass of a glowing stroke: its width and alpha relative to the core.
struct GlowPass {
    float width{1.0F};
    float alpha{1.0F};
};

/// Four passes: a wide haze, a halo, an inner glow and the core line.
inline constexpr std::array<GlowPass, 4> kGlow{{
    {.width = 6.0F, .alpha = 0.05F},
    {.width = 3.2F, .alpha = 0.10F},
    {.width = 1.8F, .alpha = 0.25F},
    {.width = 1.0F, .alpha = 0.95F},
}};

/// Two passes for small or numerous strokes (spokes, tails).
inline constexpr std::array<GlowPass, 2> kSoftGlow{{
    {.width = 3.0F, .alpha = 0.14F},
    {.width = 1.0F, .alpha = 0.90F},
}};

/// Two passes when the bloom pass supplies the wide glow: a soft halo and the core.
inline constexpr std::array<GlowPass, 2> kBloomGlow{{
    {.width = 2.6F, .alpha = 0.10F},
    {.width = 1.0F, .alpha = 0.95F},
}};

/// Just the core line, for dense geometry (globe wireframe, particles' trails).
inline constexpr std::array<GlowPass, 1> kCoreOnly{{
    {.width = 1.0F, .alpha = 1.0F},
}};

/// A stroke: colour (alpha already scaled), core thickness in pixels, passes.
struct Stroke {
    Rgba colour;
    float thickness{1.0F};
    std::span<const GlowPass> glow{kGlow};
};

[[nodiscard]] ImU32 toU32(Rgba colour) noexcept;
[[nodiscard]] ImVec2 polar(ImVec2 centre, float radius, float angle) noexcept;
[[nodiscard]] Rgba mixColour(Rgba from, Rgba to, float t) noexcept;

void glowArc(ImDrawList& draw, ImVec2 centre, float radius, float fromAngle, float toAngle,
             const Stroke& stroke);
void glowCircle(ImDrawList& draw, ImVec2 centre, float radius, const Stroke& stroke);
void glowPolyline(ImDrawList& draw, std::span<const ImVec2> points, bool closed,
                  const Stroke& stroke);
void glowLine(ImDrawList& draw, ImVec2 from, ImVec2 to, const Stroke& stroke);
/// A filled point of light with a soft halo three times its radius.
void glowDot(ImDrawList& draw, ImVec2 centre, float radius, Rgba colour);

/// Point at @p angle on the ellipse with radii (@p rx, @p ry) around @p centre.
[[nodiscard]] ImVec2 ellipsePoint(ImVec2 centre, float rx, float ry, float angle) noexcept;

/// An arc of an axis-aligned ellipse; a circle when @p rx == @p ry.
void glowEllipseArc(ImDrawList& draw, ImVec2 centre, float rx, float ry, float fromAngle,
                    float toAngle, const Stroke& stroke);

/// @p segments equal arcs around an ellipse, each filling @p fill of its
/// slot; a full ellipse when @p segments is 0.
void segmentedEllipse(ImDrawList& draw, ImVec2 centre, float rx, float ry, int segments, float fill,
                      float rotation, const Stroke& stroke);

/// A disc of light fading from @p colour at the centre to nothing at @p radius.
void radialGlow(ImDrawList& draw, ImVec2 centre, float radius, Rgba colour);

/// @p segments equal arcs around a circle, each filling @p fill of its slot.
void segmentedRing(ImDrawList& draw, ImVec2 centre, float radius, int segments, float fill,
                   float rotation, const Stroke& stroke);

/// Radial tick marks between @p inner and @p outer; every @p majorEvery-th
/// tick reaches @p majorOuter instead.
struct TickSpec {
    float inner{0.0F};
    float outer{0.0F};
    float majorOuter{0.0F};
    int count{0};
    int majorEvery{1};
    float rotation{0.0F};
    float thickness{1.0F};
};
void ticks(ImDrawList& draw, ImVec2 centre, const TickSpec& spec, Rgba colour);

/// A wedge of light whose leading edge is at @p headAngle and which fades
/// to nothing over @p span radians behind it (vertex-colour gradient).
void sweep(ImDrawList& draw, ImVec2 centre, float radius, float headAngle, float span, Rgba colour);

/// Hexagon outline of circumradius @p size.
void hexagon(ImDrawList& draw, ImVec2 centre, float size, float rotation, Rgba colour,
             float thickness);

/// L-shaped brackets on the four corners of @p rect.
void cornerBrackets(ImDrawList& draw, RectF rect, float length, Rgba colour, float thickness);

/// Width of @p text drawn with spacedText(); ASCII captions only.
[[nodiscard]] float spacedTextWidth(ImFont& font, float size, std::string_view text,
                                    float tracking);

/// Draw @p text one glyph at a time with @p tracking pixels between glyphs
/// (HUD captions are letter-spaced); ASCII captions only. Returns the width.
float spacedText(ImDrawList& draw, ImFont& font, float size, ImVec2 position, std::string_view text,
                 float tracking, Rgba colour);

/// Plain UTF-8 text at @p size.
void text(ImDrawList& draw, ImFont& font, float size, ImVec2 position, std::string_view utf8,
          Rgba colour);

/// Switches the draw commands that follow between the light layer and
/// normal ("over") blending through draw-list callbacks into the DX11 backend.
class BlendModes {
public:
    /// @p post is the post-processing pipeline, or null to draw light
    /// additively onto the swap chain. Non-owning; it must outlive every
    /// frame rendered with these callbacks.
    /// @throws OverlayError if the additive blend state cannot be created.
    BlendModes(ID3D11Device& device, PostProcessor* post);

    /// Commands added after this are light: into the bloomed light layer,
    /// or additively onto the swap chain without post-processing.
    void beginLight(ImDrawList& draw) const;
    /// End the light section: bloom and composite it (if post-processing
    /// runs), then restore the backend's default blending.
    void endLight(ImDrawList& draw) const;
    /// True when the light section is bloomed by the GPU pipeline.
    [[nodiscard]] bool bloom() const noexcept { return post_ != nullptr; }

private:
    Microsoft::WRL::ComPtr<ID3D11BlendState> additive_;
    PostProcessor* post_;  // non-owning, see the constructor
};

}  // namespace jarvis::overlay::hud
