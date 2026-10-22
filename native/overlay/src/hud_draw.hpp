/// @file hud_draw.hpp
/// @brief Glowing HUD primitives on top of ImDrawList (private).
///
/// Glow is layered strokes: the same path drawn wide and faint, then
/// narrower and brighter, then a crisp core. Inside an additive section
/// (BlendModes) overlapping light adds up towards white the way light does,
/// which is most of what makes the rings read as emitted light rather than
/// paint. No shader, no blur pass, a few hundred vertices per ring.

#pragma once

#include "jarvis/overlay/layout.hpp"

#include <imgui.h>

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

/// Switches the draw commands that follow between additive and normal
/// ("over") blending through draw-list callbacks into the DX11 backend.
class BlendModes {
public:
    /// @throws OverlayError if the blend state cannot be created.
    explicit BlendModes(ID3D11Device& device);

    /// Commands added after this are blended additively (light adds up).
    void additive(ImDrawList& draw) const;
    /// Commands added after this use the backend's default blending again.
    static void normal(ImDrawList& draw);

private:
    Microsoft::WRL::ComPtr<ID3D11BlendState> additive_;
};

}  // namespace jarvis::overlay::hud
