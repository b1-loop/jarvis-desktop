#include "postfx.hpp"

#include "jarvis/overlay/errors.hpp"
#include "jarvis/overlay/shaders.hpp"

#include <d3d11.h>
#include <d3dcompiler.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace jarvis::overlay {

namespace {

using Microsoft::WRL::ComPtr;

/// Packed float light (4 bytes a pixel): additive glows may sum past 1
/// before the bright pass, and the light layer needs no alpha (the composite
/// derives coverage from brightness). Half-float RGBA is the fallback for a
/// device that cannot blend into R11G11B10.
constexpr DXGI_FORMAT kPackedLight = DXGI_FORMAT_R11G11B10_FLOAT;
constexpr DXGI_FORMAT kWideLight = DXGI_FORMAT_R16G16B16A16_FLOAT;

DXGI_FORMAT lightFormat(ID3D11Device& device) noexcept {
    UINT support = 0;
    constexpr UINT kNeeded = static_cast<UINT>(D3D11_FORMAT_SUPPORT_RENDER_TARGET) |
                             static_cast<UINT>(D3D11_FORMAT_SUPPORT_BLENDABLE) |
                             static_cast<UINT>(D3D11_FORMAT_SUPPORT_SHADER_SAMPLE);
    if (SUCCEEDED(device.CheckFormatSupport(kPackedLight, &support)) &&
        (support & kNeeded) == kNeeded) {
        return kPackedLight;
    }
    return kWideLight;
}
constexpr int kMinBloomLevels = 2;
constexpr int kMaxBloomLevels = 4;
/// Everything brighter than this blooms; the knee makes the onset soft.
constexpr float kThreshold = 0.22F;
constexpr float kKnee = 0.30F;
/// Alpha of the composited light relative to its brightest channel; a
/// little below 1 so light adds onto the desktop rather than covering it.
constexpr float kCoverage = 0.9F;
constexpr UINT kAllSamples = 0xFFFFFFFFU;
constexpr std::array<float, 4> kNoBlendFactor{0.0F, 0.0F, 0.0F, 0.0F};
constexpr std::array<float, 4> kClearLight{0.0F, 0.0F, 0.0F, 0.0F};
constexpr double kMillisecondsPerSecond = 1000.0;
/// Texels added around the scissored region at every level for the blur's reach.
constexpr LONG kScissorMargin = 4;

/// Matches cbuffer PostParams in shaders/common.hlsli (three float4 rows).
struct PostConstants {
    std::array<float, 2> sourceTexel{};
    float threshold{kThreshold};
    float knee{kKnee};
    float bloomStrength{0.0F};
    float glitch{0.0F};
    float time{0.0F};
    float targetHeight{1.0F};
    std::array<float, 2> whip{};
    float coverage{kCoverage};
    float bloomNorm{1.0F};
};
static_assert(sizeof(PostConstants) == 48, "must match the HLSL constant buffer layout");

void check(HRESULT result, std::string_view what) {
    if (FAILED(result)) {
        throw OverlayError(std::string{what} + " failed (HRESULT " +
                           std::to_string(static_cast<long>(result)) + ")");
    }
}

ComPtr<ID3DBlob> compile(const ShaderSource& shader) {
    const std::string name{shader.name};
    const std::string entry{shader.entry};
    const std::string profile{shader.profile};
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> errors;
    const HRESULT result = D3DCompile(shader.code.data(), shader.code.size(), name.c_str(), nullptr,
                                      nullptr, entry.c_str(), profile.c_str(),
                                      D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_ENABLE_STRICTNESS,
                                      0, &bytecode, &errors);
    if (FAILED(result)) {
        std::string message = "shader " + name + " does not compile";
        if (errors != nullptr) {
            message += ": ";
            message.append(static_cast<const char*>(errors->GetBufferPointer()),
                           errors->GetBufferSize());
        }
        throw OverlayError(message);
    }
    return bytecode;
}

ComPtr<ID3D11PixelShader> pixelShader(ID3D11Device& device, ShaderId id) {
    const ComPtr<ID3DBlob> blob = compile(shaderSource(id));
    ComPtr<ID3D11PixelShader> shader;
    check(
        device.CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &shader),
        "CreatePixelShader");
    return shader;
}

/// A blend state for render target 0 with the given colour and alpha factors.
ComPtr<ID3D11BlendState> blendState(ID3D11Device& device, D3D11_BLEND source, D3D11_BLEND dest,
                                    D3D11_BLEND sourceAlpha, D3D11_BLEND destAlpha) {
    // Zero-filling is how the D3D11 documentation initialises this struct;
    // render targets 1-7 stay disabled and are never read.
    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization)
    D3D11_BLEND_DESC desc{};
    D3D11_RENDER_TARGET_BLEND_DESC& target = desc.RenderTarget[0];
    target.BlendEnable = TRUE;
    target.SrcBlend = source;
    target.DestBlend = dest;
    target.BlendOp = D3D11_BLEND_OP_ADD;
    target.SrcBlendAlpha = sourceAlpha;
    target.DestBlendAlpha = destAlpha;
    target.BlendOpAlpha = D3D11_BLEND_OP_ADD;
    target.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    ComPtr<ID3D11BlendState> state;
    check(device.CreateBlendState(&desc, &state), "CreateBlendState");
    return state;
}

ComPtr<ID3D11SamplerState> linearClamp(ID3D11Device& device) {
    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization) - documented zero init
    D3D11_SAMPLER_DESC desc{};
    desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    desc.MaxLOD = D3D11_FLOAT32_MAX;
    ComPtr<ID3D11SamplerState> sampler;
    check(device.CreateSamplerState(&desc, &sampler), "CreateSamplerState");
    return sampler;
}

/// Solid fill, scissored to the light region (ImGui's last clip rectangle
/// is replaced before every pass).
ComPtr<ID3D11RasterizerState> regionScissor(ID3D11Device& device) {
    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization) - documented zero init
    D3D11_RASTERIZER_DESC desc{};
    desc.FillMode = D3D11_FILL_SOLID;
    desc.CullMode = D3D11_CULL_NONE;
    desc.DepthClipEnable = TRUE;
    desc.ScissorEnable = TRUE;  // our own rectangle: the light region (scissorFor)
    ComPtr<ID3D11RasterizerState> state;
    check(device.CreateRasterizerState(&desc, &state), "CreateRasterizerState");
    return state;
}

ComPtr<ID3D11DepthStencilState> depthOff(ID3D11Device& device) {
    // NOLINTNEXTLINE(bugprone-invalid-enum-default-initialization) - documented zero init
    D3D11_DEPTH_STENCIL_DESC desc{};
    desc.DepthEnable = FALSE;
    desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    desc.DepthFunc = D3D11_COMPARISON_ALWAYS;
    ComPtr<ID3D11DepthStencilState> state;
    check(device.CreateDepthStencilState(&desc, &state), "CreateDepthStencilState");
    return state;
}

SIZE viewSize(ID3D11RenderTargetView& view) {
    ComPtr<ID3D11Resource> resource;
    view.GetResource(&resource);
    ComPtr<ID3D11Texture2D> texture;
    if (FAILED(resource.As(&texture))) {
        return SIZE{.cx = 0, .cy = 0};
    }
    D3D11_TEXTURE2D_DESC desc{};
    texture->GetDesc(&desc);
    return SIZE{.cx = static_cast<LONG>(desc.Width), .cy = static_cast<LONG>(desc.Height)};
}

void setViewport(ID3D11DeviceContext& context, UINT width, UINT height) {
    const D3D11_VIEWPORT viewport{.TopLeftX = 0.0F,
                                  .TopLeftY = 0.0F,
                                  .Width = static_cast<float>(width),
                                  .Height = static_cast<float>(height),
                                  .MinDepth = 0.0F,
                                  .MaxDepth = 1.0F};
    context.RSSetViewports(1, &viewport);
}

void unbindResources(ID3D11DeviceContext& context) {
    const std::array<ID3D11ShaderResourceView*, 2> none{nullptr, nullptr};
    context.PSSetShaderResources(0, static_cast<UINT>(none.size()), none.data());
}

}  // namespace

// --- PostProcessor -----------------------------------------------------------------

PostProcessor::PostProcessor(ID3D11Device& device)
    : device_(&device),
      format_(lightFormat(device)),
      prefilter_(pixelShader(device, ShaderId::Prefilter)),
      downsample_(pixelShader(device, ShaderId::Downsample)),
      upsample_(pixelShader(device, ShaderId::Upsample)),
      composite_(pixelShader(device, ShaderId::Composite)),
      sampler_(linearClamp(device)),
      // Light layer: colour += src * srcAlpha, coverage += srcAlpha (the
      // same additive light as the ADR-0022 glow, now into half floats).
      lightBlend_(blendState(device, D3D11_BLEND_SRC_ALPHA, D3D11_BLEND_ONE, D3D11_BLEND_ONE,
                             D3D11_BLEND_ONE)),
      addBlend_(
          blendState(device, D3D11_BLEND_ONE, D3D11_BLEND_ONE, D3D11_BLEND_ONE, D3D11_BLEND_ONE)),
      // Premultiplied "over" onto the swap chain, which is what DWM expects.
      overBlend_(blendState(device, D3D11_BLEND_ONE, D3D11_BLEND_INV_SRC_ALPHA, D3D11_BLEND_ONE,
                            D3D11_BLEND_INV_SRC_ALPHA)),
      rasterizer_(regionScissor(device)),
      noDepth_(depthOff(device)) {
    const ComPtr<ID3DBlob> vertex = compile(shaderSource(ShaderId::Fullscreen));
    check(device.CreateVertexShader(vertex->GetBufferPointer(), vertex->GetBufferSize(), nullptr,
                                    &fullscreen_),
          "CreateVertexShader");
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = sizeof(PostConstants);
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    check(device.CreateBuffer(&desc, nullptr, &constants_), "CreateBuffer (post constants)");
}

PostProcessor::Target PostProcessor::createTarget(UINT width, UINT height) const {
    Target target;
    target.width = std::max<UINT>(width, 1);
    target.height = std::max<UINT>(height, 1);
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = target.width;
    desc.Height = target.height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format_;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags =
        static_cast<UINT>(D3D11_BIND_RENDER_TARGET) | static_cast<UINT>(D3D11_BIND_SHADER_RESOURCE);
    check(device_->CreateTexture2D(&desc, nullptr, &target.texture), "CreateTexture2D (light)");
    check(device_->CreateRenderTargetView(target.texture.Get(), nullptr, &target.view),
          "CreateRenderTargetView (light)");
    check(device_->CreateShaderResourceView(target.texture.Get(), nullptr, &target.resource),
          "CreateShaderResourceView (light)");
    return target;
}

void PostProcessor::ensureSize(UINT width, UINT height) {
    if (light_.view != nullptr && light_.width == width && light_.height == height) {
        return;
    }
    light_ = createTarget(width, height);
    UINT w = width;
    UINT h = height;
    for (Target& mip : mips_) {
        w = std::max<UINT>(w / 2, 1);
        h = std::max<UINT>(h / 2, 1);
        mip = createTarget(w, h);
    }
}

void PostProcessor::beginLight(ID3D11DeviceContext& context) {
    if (!failure_.empty()) {
        return;
    }
    try {
        redirect(context);
    } catch (const OverlayError& error) {
        // Not swallowed: kept in failure(), which the window reports and
        // acts on (it switches the bloom off) right after this frame.
        failure_ = error.what();
        output_.Reset();
    }
}

void PostProcessor::redirect(ID3D11DeviceContext& context) {
    output_.Reset();
    context.OMGetRenderTargets(1, output_.GetAddressOf(), nullptr);
    if (output_ == nullptr) {
        return;
    }
    const SIZE size = viewSize(*output_.Get());
    outputWidth_ = static_cast<UINT>(size.cx);
    outputHeight_ = static_cast<UINT>(size.cy);
    ensureSize(outputWidth_, outputHeight_);
    context.OMSetRenderTargets(1, light_.view.GetAddressOf(), nullptr);
    context.ClearRenderTargetView(light_.view.Get(), kClearLight.data());
    context.OMSetBlendState(lightBlend_.Get(), kNoBlendFactor.data(), kAllSamples);
}

void PostProcessor::bindPass(ID3D11DeviceContext& context, const Target& output,
                             ID3D11PixelShader& shader, ID3D11BlendState* blend) {
    context.OMSetRenderTargets(1, output.view.GetAddressOf(), nullptr);
    context.OMSetBlendState(blend, kNoBlendFactor.data(), kAllSamples);
    setViewport(context, output.width, output.height);
    context.PSSetShader(&shader, nullptr, 0);
}

void PostProcessor::updateConstants(ID3D11DeviceContext& context, const Target& source,
                                    float bloomNorm) {
    PostConstants constants;
    constants.sourceTexel = {1.0F / static_cast<float>(source.width),
                             1.0F / static_cast<float>(source.height)};
    constants.bloomStrength = settings_.bloomStrength;
    constants.glitch = settings_.glitch;
    constants.time = settings_.time;
    constants.targetHeight = static_cast<float>(std::max<UINT>(outputHeight_, 1));
    constants.whip = {settings_.whipX / static_cast<float>(std::max<UINT>(outputWidth_, 1)),
                      settings_.whipY / constants.targetHeight};
    constants.bloomNorm = bloomNorm;
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (SUCCEEDED(context.Map(constants_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        *static_cast<PostConstants*>(mapped.pData) = constants;
        context.Unmap(constants_.Get(), 0);
    }
}

void PostProcessor::scissorFor(ID3D11DeviceContext& context, UINT width, UINT height) const {
    const RECT& r = settings_.region;
    D3D11_RECT scissor{.left = 0,
                       .top = 0,
                       .right = static_cast<LONG>(width),
                       .bottom = static_cast<LONG>(height)};
    if (r.right > r.left && r.bottom > r.top && outputWidth_ > 0 && outputHeight_ > 0) {
        // Map from output pixels to this target, rounding outwards, plus a
        // margin for the blur kernels' reach.
        const auto sx = static_cast<float>(width) / static_cast<float>(outputWidth_);
        const auto sy = static_cast<float>(height) / static_cast<float>(outputHeight_);
        scissor.left =
            std::max<LONG>(static_cast<LONG>(static_cast<float>(r.left) * sx) - kScissorMargin, 0);
        scissor.top =
            std::max<LONG>(static_cast<LONG>(static_cast<float>(r.top) * sy) - kScissorMargin, 0);
        scissor.right =
            std::min<LONG>(static_cast<LONG>(static_cast<float>(r.right) * sx) + kScissorMargin,
                           static_cast<LONG>(width));
        scissor.bottom =
            std::min<LONG>(static_cast<LONG>(static_cast<float>(r.bottom) * sy) + kScissorMargin,
                           static_cast<LONG>(height));
    }
    context.RSSetScissorRects(1, &scissor);
}

void PostProcessor::runPass(ID3D11DeviceContext& context, const Target& source,
                            const Target& output, ID3D11PixelShader& shader,
                            ID3D11BlendState* blend) {
    unbindResources(context);
    bindPass(context, output, shader, blend);
    scissorFor(context, output.width, output.height);
    updateConstants(context, source, 1.0F);
    context.PSSetShaderResources(0, 1, source.resource.GetAddressOf());
    context.Draw(3, 0);
}

void PostProcessor::endLight(ID3D11DeviceContext& context) {
    if (output_ == nullptr || !failure_.empty()) {
        return;
    }
    context.IASetInputLayout(nullptr);
    context.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context.VSSetShader(fullscreen_.Get(), nullptr, 0);
    context.PSSetConstantBuffers(0, 1, constants_.GetAddressOf());
    context.PSSetSamplers(0, 1, sampler_.GetAddressOf());
    context.RSSetState(rasterizer_.Get());
    context.OMSetDepthStencilState(noDepth_.Get(), 0);

    const int levels = std::clamp(settings_.bloomLevels, kMinBloomLevels, kMaxBloomLevels);
    const auto last = static_cast<std::size_t>(levels - 1);
    // The blur reads a few texels beyond the scissored region; clearing the
    // (small) mips keeps last frame's light from leaking in there.
    for (std::size_t i = 0; i <= last; ++i) {
        context.ClearRenderTargetView(mips_.at(i).view.Get(), kClearLight.data());
    }
    runPass(context, light_, mips_.at(0), *prefilter_.Get(), nullptr);
    for (std::size_t i = 1; i <= last; ++i) {
        runPass(context, mips_.at(i - 1), mips_.at(i), *downsample_.Get(), nullptr);
    }
    for (std::size_t i = last; i >= 1; --i) {
        runPass(context, mips_.at(i), mips_.at(i - 1), *upsample_.Get(), addBlend_.Get());
    }
    composite(context, levels);
    unbindResources(context);
    output_.Reset();
}

void PostProcessor::composite(ID3D11DeviceContext& context, int levels) {
    unbindResources(context);
    context.OMSetRenderTargets(1, output_.GetAddressOf(), nullptr);
    context.OMSetBlendState(overBlend_.Get(), kNoBlendFactor.data(), kAllSamples);
    setViewport(context, outputWidth_, outputHeight_);
    context.PSSetShader(composite_.Get(), nullptr, 0);
    scissorFor(context, outputWidth_, outputHeight_);
    updateConstants(context, light_, 1.0F / static_cast<float>(levels));
    const std::array<ID3D11ShaderResourceView*, 2> inputs{light_.resource.Get(),
                                                          mips_.at(0).resource.Get()};
    context.PSSetShaderResources(0, static_cast<UINT>(inputs.size()), inputs.data());
    context.Draw(3, 0);
}

// --- GpuTimer ------------------------------------------------------------------------

GpuTimer::GpuTimer(ID3D11Device& device) {
    const D3D11_QUERY_DESC disjoint{.Query = D3D11_QUERY_TIMESTAMP_DISJOINT, .MiscFlags = 0};
    const D3D11_QUERY_DESC stamp{.Query = D3D11_QUERY_TIMESTAMP, .MiscFlags = 0};
    for (Slot& slot : slots_) {
        check(device.CreateQuery(&disjoint, &slot.disjoint), "CreateQuery (disjoint)");
        check(device.CreateQuery(&stamp, &slot.start), "CreateQuery (timestamp)");
        check(device.CreateQuery(&stamp, &slot.stop), "CreateQuery (timestamp)");
    }
}

void GpuTimer::begin(ID3D11DeviceContext& context) {
    const Slot& slot = slots_.at(write_);
    if (slot.pending) {
        open_ = false;  // every slot is still in flight: skip timing this frame
        return;
    }
    context.Begin(slot.disjoint.Get());
    context.End(slot.start.Get());
    open_ = true;
}

void GpuTimer::end(ID3D11DeviceContext& context) {
    if (!open_) {
        return;
    }
    Slot& slot = slots_.at(write_);
    context.End(slot.stop.Get());
    context.End(slot.disjoint.Get());
    slot.pending = true;
    write_ = (write_ + 1) % kSlots;
    open_ = false;
}

std::optional<double> GpuTimer::collect(ID3D11DeviceContext& context) {
    Slot& slot = slots_.at(read_);
    if (!slot.pending) {
        return std::nullopt;
    }
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT clock{};
    std::uint64_t start = 0;
    std::uint64_t stop = 0;
    constexpr UINT kNoFlush = D3D11_ASYNC_GETDATA_DONOTFLUSH;
    if (context.GetData(slot.disjoint.Get(), &clock, sizeof(clock), kNoFlush) != S_OK ||
        context.GetData(slot.start.Get(), &start, sizeof(start), kNoFlush) != S_OK ||
        context.GetData(slot.stop.Get(), &stop, sizeof(stop), kNoFlush) != S_OK) {
        return std::nullopt;  // not finished yet; asked again next frame
    }
    slot.pending = false;
    read_ = (read_ + 1) % kSlots;
    if (clock.Disjoint != FALSE || clock.Frequency == 0 || stop < start) {
        return std::nullopt;
    }
    return static_cast<double>(stop - start) * kMillisecondsPerSecond /
           static_cast<double>(clock.Frequency);
}

}  // namespace jarvis::overlay

// Design notes:
// - The pipeline runs inside Dear ImGui's own RenderDrawData through two
//   draw-list callbacks, so the font atlas, the backend and its texture
//   updates stay untouched: everything drawn between the callbacks goes to
//   the light layer, everything after them (text) straight to the swap
//   chain. A ResetRenderState callback after endLight() restores ImGui's
//   pipeline; the render target is restored here.
// - Dual Kawase over a separable Gaussian: 5 and 8 taps per pixel per level
//   for a wide, smooth falloff, and the number of levels (2-4, set by the
//   intensity) is the only knob for the radius.
// - The light layer is sized like the swap chain, so the boot (work-area
//   window) and the panel (small window) both get exact pixels; the
//   targets are recreated only when that size changes, i.e. twice per boot.
// - Composited alpha is the brightest channel times kCoverage: dark parts
//   of the light layer stay transparent on the DirectComposition surface,
//   bright cores become opaque, and the result is valid premultiplied alpha.
