#include "jarvis/overlay/shaders.hpp"

#include "jarvis/overlay/embedded_shaders.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace jarvis::overlay {

namespace {

constexpr std::string_view kVertexProfile = "vs_4_0";
constexpr std::string_view kPixelProfile = "ps_4_0";
constexpr std::string_view kEntry = "main";

constexpr std::array<ShaderSource, 5> kShaders{{
    {.id = ShaderId::Fullscreen,
     .name = "fullscreen_vs",
     .profile = kVertexProfile,
     .entry = kEntry,
     .code = embedded::kFullscreenVs},
    {.id = ShaderId::Prefilter,
     .name = "bloom_prefilter_ps",
     .profile = kPixelProfile,
     .entry = kEntry,
     .code = embedded::kBloomPrefilterPs},
    {.id = ShaderId::Downsample,
     .name = "bloom_down_ps",
     .profile = kPixelProfile,
     .entry = kEntry,
     .code = embedded::kBloomDownPs},
    {.id = ShaderId::Upsample,
     .name = "bloom_up_ps",
     .profile = kPixelProfile,
     .entry = kEntry,
     .code = embedded::kBloomUpPs},
    {.id = ShaderId::Composite,
     .name = "composite_ps",
     .profile = kPixelProfile,
     .entry = kEntry,
     .code = embedded::kCompositePs},
}};

}  // namespace

std::span<const ShaderSource> shaderSources() noexcept {
    return kShaders;
}

const ShaderSource& shaderSource(ShaderId id) noexcept {
    return kShaders.at(static_cast<std::size_t>(id) % kShaders.size());
}

}  // namespace jarvis::overlay

// Design notes:
// - Shader model 4.0 rather than 5.0: feature level 10_0 devices and WARP
//   (the software fallback) run it, and the passes need nothing newer.
// - Sources, not precompiled bytecode, are embedded: fxc is not on every
//   build machine's PATH and a configure-only clang-tidy run must find the
//   generated header without building, while D3DCompile ships with Windows
//   (Dear ImGui's DX11 backend already uses it). Compiling five small
//   shaders once per window costs a few milliseconds at startup, and the
//   Catch2 suite compiles every one, so a broken shader fails the tests.
