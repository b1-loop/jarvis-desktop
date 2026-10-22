#include "hud_draw.hpp"

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/errors.hpp"
#include "jarvis/overlay/layout.hpp"

#include <imgui.h>
#include <imgui_impl_dx11.h>

#include <d3d11.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <span>
#include <string>
#include <string_view>

namespace jarvis::overlay::hud {

namespace {

/// Roughly one segment per this many pixels of arc keeps curves smooth
/// without spending vertices on tiny arcs.
constexpr float kPixelsPerSegment = 3.0F;
constexpr int kMinArcSegments = 6;
constexpr int kSweepSlices = 32;
constexpr int kHexCorners = 6;
constexpr float kDotHalo = 3.0F;
constexpr float kDotHaloAlpha = 0.18F;

int arcSegments(float radius, float span) noexcept {
    const float length = std::abs(span) * radius;
    return std::max(kMinArcSegments, static_cast<int>(length / kPixelsPerSegment));
}

Rgba scaledAlpha(Rgba colour, float factor) noexcept {
    return withAlpha(colour, factor);
}

void setAdditive(const ImDrawList* /*list*/, const ImDrawCmd* command) {
    const auto* const state =
        static_cast<const ImGui_ImplDX11_RenderState*>(ImGui::GetPlatformIO().Renderer_RenderState);
    if (state == nullptr) {
        return;
    }
    auto* const blend = static_cast<ID3D11BlendState*>(command->UserCallbackData);
    constexpr std::array<float, 4> kNoFactor{0.0F, 0.0F, 0.0F, 0.0F};
    constexpr UINT kAllSamples = 0xFFFFFFFFU;
    state->DeviceContext->OMSetBlendState(blend, kNoFactor.data(), kAllSamples);
}

}  // namespace

ImU32 toU32(Rgba colour) noexcept {
    return ImGui::ColorConvertFloat4ToU32(ImVec4{colour.r, colour.g, colour.b, colour.a});
}

ImVec2 polar(ImVec2 centre, float radius, float angle) noexcept {
    return ImVec2{centre.x + (radius * std::cos(angle)), centre.y + (radius * std::sin(angle))};
}

Rgba mixColour(Rgba from, Rgba to, float t) noexcept {
    return Rgba{.r = anim::lerp(from.r, to.r, t),
                .g = anim::lerp(from.g, to.g, t),
                .b = anim::lerp(from.b, to.b, t),
                .a = anim::lerp(from.a, to.a, t)};
}

void glowArc(ImDrawList& draw, ImVec2 centre, float radius, float fromAngle, float toAngle,
             const Stroke& stroke) {
    if (radius <= 0.0F || stroke.colour.a <= 0.0F) {
        return;
    }
    const int segments = arcSegments(radius, toAngle - fromAngle);
    for (const GlowPass& pass : stroke.glow) {
        draw.PathArcTo(centre, radius, fromAngle, toAngle, segments);
        draw.PathStroke(toU32(scaledAlpha(stroke.colour, pass.alpha)),
                        stroke.thickness * pass.width);
    }
}

void glowCircle(ImDrawList& draw, ImVec2 centre, float radius, const Stroke& stroke) {
    if (radius <= 0.0F || stroke.colour.a <= 0.0F) {
        return;
    }
    const int segments = arcSegments(radius, anim::kTau);
    for (const GlowPass& pass : stroke.glow) {
        draw.AddCircle(centre, radius, toU32(scaledAlpha(stroke.colour, pass.alpha)), segments,
                       stroke.thickness * pass.width);
    }
}

void glowPolyline(ImDrawList& draw, std::span<const ImVec2> points, bool closed,
                  const Stroke& stroke) {
    if (points.size() < 2 || stroke.colour.a <= 0.0F) {
        return;
    }
    const ImDrawFlags flags = closed ? ImDrawFlags_Closed : ImDrawFlags_None;
    for (const GlowPass& pass : stroke.glow) {
        draw.AddPolyline(points.data(), static_cast<int>(points.size()),
                         toU32(scaledAlpha(stroke.colour, pass.alpha)),
                         stroke.thickness * pass.width, flags);
    }
}

void glowLine(ImDrawList& draw, ImVec2 from, ImVec2 to, const Stroke& stroke) {
    if (stroke.colour.a <= 0.0F) {
        return;
    }
    for (const GlowPass& pass : stroke.glow) {
        draw.AddLine(from, to, toU32(scaledAlpha(stroke.colour, pass.alpha)),
                     stroke.thickness * pass.width);
    }
}

void glowDot(ImDrawList& draw, ImVec2 centre, float radius, Rgba colour) {
    if (radius <= 0.0F || colour.a <= 0.0F) {
        return;
    }
    draw.AddCircleFilled(centre, radius * kDotHalo, toU32(scaledAlpha(colour, kDotHaloAlpha)));
    draw.AddCircleFilled(centre, radius, toU32(colour));
}

void segmentedRing(ImDrawList& draw, ImVec2 centre, float radius, int segments, float fill,
                   float rotation, const Stroke& stroke) {
    if (segments <= 0) {
        glowCircle(draw, centre, radius, stroke);
        return;
    }
    const float slot = anim::kTau / static_cast<float>(segments);
    const float span = slot * anim::clamp01(fill);
    for (int i = 0; i < segments; ++i) {
        const float start = rotation + (slot * static_cast<float>(i));
        glowArc(draw, centre, radius, start, start + span, stroke);
    }
}

void ticks(ImDrawList& draw, ImVec2 centre, const TickSpec& spec, Rgba colour) {
    if (spec.count <= 0 || colour.a <= 0.0F) {
        return;
    }
    const ImU32 packed = toU32(colour);
    const float step = anim::kTau / static_cast<float>(spec.count);
    const int major = std::max(spec.majorEvery, 1);
    for (int i = 0; i < spec.count; ++i) {
        const float angle = spec.rotation + (step * static_cast<float>(i));
        const float outer = i % major == 0 ? spec.majorOuter : spec.outer;
        draw.AddLine(polar(centre, spec.inner, angle), polar(centre, outer, angle), packed,
                     spec.thickness);
    }
}

void sweep(ImDrawList& draw, ImVec2 centre, float radius, float headAngle, float span,
           Rgba colour) {
    if (radius <= 0.0F || colour.a <= 0.0F) {
        return;
    }
    const ImVec2 uv = ImGui::GetFontTexUvWhitePixel();
    const ImU32 centreColour = toU32(scaledAlpha(colour, 0.0F));
    constexpr int kVerticesPerSlice = 3;
    draw.PrimReserve(kSweepSlices * kVerticesPerSlice, kSweepSlices * kVerticesPerSlice);
    for (int i = 0; i < kSweepSlices; ++i) {
        const float t0 = static_cast<float>(i) / static_cast<float>(kSweepSlices);
        const float t1 = static_cast<float>(i + 1) / static_cast<float>(kSweepSlices);
        const float a0 = headAngle - (span * (1.0F - t0));
        const float a1 = headAngle - (span * (1.0F - t1));
        // Alpha grows towards the leading edge; squared so the tail is soft.
        draw.PrimVtx(centre, uv, centreColour);
        draw.PrimVtx(polar(centre, radius, a0), uv, toU32(scaledAlpha(colour, t0 * t0)));
        draw.PrimVtx(polar(centre, radius, a1), uv, toU32(scaledAlpha(colour, t1 * t1)));
    }
}

void hexagon(ImDrawList& draw, ImVec2 centre, float size, float rotation, Rgba colour,
             float thickness) {
    if (size <= 0.0F || colour.a <= 0.0F) {
        return;
    }
    std::array<ImVec2, kHexCorners> corners{};
    const float step = anim::kTau / static_cast<float>(kHexCorners);
    for (std::size_t i = 0; i < corners.size(); ++i) {
        corners.at(i) = polar(centre, size, rotation + (step * static_cast<float>(i)));
    }
    draw.AddPolyline(corners.data(), kHexCorners, toU32(colour), thickness, ImDrawFlags_Closed);
}

void cornerBrackets(ImDrawList& draw, RectF rect, float length, Rgba colour, float thickness) {
    if (colour.a <= 0.0F) {
        return;
    }
    const ImU32 packed = toU32(colour);
    const float left = rect.x;
    const float top = rect.y;
    const float right = rect.right();
    const float bottom = rect.bottom();
    const std::array<std::array<ImVec2, 3>, 4> brackets{{
        {{{left, top + length}, {left, top}, {left + length, top}}},
        {{{right - length, top}, {right, top}, {right, top + length}}},
        {{{right, bottom - length}, {right, bottom}, {right - length, bottom}}},
        {{{left + length, bottom}, {left, bottom}, {left, bottom - length}}},
    }};
    for (const auto& bracket : brackets) {
        draw.AddPolyline(bracket.data(), static_cast<int>(bracket.size()), packed, thickness);
    }
}

float spacedTextWidth(ImFont& font, float size, std::string_view text, float tracking) {
    float width = 0.0F;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char* glyph = text.data() + i;
        width += font.CalcTextSizeA(size, FLT_MAX, 0.0F, glyph, glyph + 1).x;
        if (i + 1 < text.size()) {
            width += tracking;
        }
    }
    return width;
}

float spacedText(ImDrawList& draw, ImFont& font, float size, ImVec2 position, std::string_view text,
                 float tracking, Rgba colour) {
    const ImU32 packed = toU32(colour);
    float x = position.x;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char* glyph = text.data() + i;
        if (colour.a > 0.0F) {
            draw.AddText(&font, size, ImVec2{x, position.y}, packed, glyph, glyph + 1);
        }
        x += font.CalcTextSizeA(size, FLT_MAX, 0.0F, glyph, glyph + 1).x + tracking;
    }
    return text.empty() ? 0.0F : x - position.x - tracking;
}

void text(ImDrawList& draw, ImFont& font, float size, ImVec2 position, std::string_view utf8,
          Rgba colour) {
    if (utf8.empty() || colour.a <= 0.0F) {
        return;
    }
    draw.AddText(&font, size, position, toU32(colour), utf8.data(), utf8.data() + utf8.size());
}

BlendModes::BlendModes(ID3D11Device& device) {
    // Additive light on a premultiplied target: colour += src * srcAlpha and
    // coverage += srcAlpha, so colour never exceeds alpha (stays valid
    // premultiplied) and overlapping glows brighten instead of occluding.
    // Zero-filling is how the D3D11 documentation initialises this struct;
    // render targets 1-7 stay disabled and are never read.
    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
    D3D11_BLEND_DESC desc{};
    D3D11_RENDER_TARGET_BLEND_DESC& target = desc.RenderTarget[0];
    target.BlendEnable = TRUE;
    target.SrcBlend = D3D11_BLEND_SRC_ALPHA;
    target.DestBlend = D3D11_BLEND_ONE;
    target.BlendOp = D3D11_BLEND_OP_ADD;
    target.SrcBlendAlpha = D3D11_BLEND_ONE;
    target.DestBlendAlpha = D3D11_BLEND_ONE;
    target.BlendOpAlpha = D3D11_BLEND_OP_ADD;
    target.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(device.CreateBlendState(&desc, &additive_))) {
        throw OverlayError("could not create the additive blend state");
    }
}

void BlendModes::additive(ImDrawList& draw) const {
    draw.AddCallback(&setAdditive, additive_.Get());
}

void BlendModes::normal(ImDrawList& draw) {
    draw.AddCallback(ImGui::GetPlatformIO().DrawCallback_ResetRenderState, nullptr);
}

}  // namespace jarvis::overlay::hud

// Design notes:
// - Every primitive returns early on zero alpha: cross-faded layers whose
//   weight is 0 then cost nothing, which is what lets the panel draw all
//   four state layers unconditionally and stay cheap at idle.
// - The sweep is a vertex-colour triangle fan rather than many alpha-stepped
//   triangles, so the gradient is smooth without extra geometry.
// - spacedText() walks bytes because HUD captions are fixed ASCII strings;
//   user text (transcripts) goes through text(), which ImGui decodes as UTF-8.
