/// @file shaders.hpp
/// @brief HLSL sources of the post-processing passes, embedded at build time.
///
/// The shaders live as .hlsl files in native/overlay/shaders/. CMake embeds
/// them into a generated header at configure time and the renderer compiles
/// them with D3DCompile once, when the window is created; the compiled
/// blobs are kept for the window's lifetime. This header only exposes the
/// text, so it needs no Direct3D and the tests can compile every pass on
/// their own.

#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace jarvis::overlay {

/// The passes of the post-processing pipeline.
enum class ShaderId : std::uint8_t { Fullscreen, Prefilter, Downsample, Upsample, Composite };

/// One shader: its name (for errors), target profile, entry point and the
/// complete source (the shared prelude already prepended).
struct ShaderSource {
    ShaderId id{ShaderId::Fullscreen};
    std::string_view name;
    std::string_view profile;
    std::string_view entry;
    std::string_view code;
};

/// Every shader, in ShaderId order.
[[nodiscard]] std::span<const ShaderSource> shaderSources() noexcept;

/// The shader for @p id.
[[nodiscard]] const ShaderSource& shaderSource(ShaderId id) noexcept;

}  // namespace jarvis::overlay
