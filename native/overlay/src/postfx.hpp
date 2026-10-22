/// @file postfx.hpp
/// @brief GPU post-processing of the HUD's light layer, and GPU timing (private).
///
/// The HUD's glowing parts are drawn into an offscreen half-float target
/// instead of the swap chain. PostProcessor then extracts the bright parts,
/// blurs them down and up a dual-Kawase mip chain (2-4 levels) and
/// composites light plus bloom onto the swap chain as premultiplied alpha,
/// with the glitch (chromatic aberration, tearing, scanlines) and the
/// hand-off whip blur in the same pass. Text is drawn after it, so it stays
/// crisp. Only window.cpp and hud_draw.cpp include this.

#pragma once

#include <d3d11.h>
#include <windows.h>
#include <wrl/client.h>

#include <array>
#include <cstddef>
#include <optional>
#include <string>

namespace jarvis::overlay {

/// Per-frame inputs of the composite pass.
struct PostSettings {
    /// How much blurred light is added back; 0 disables the bloom's contribution.
    float bloomStrength{1.0F};
    /// Glitch amount in [0, 1].
    float glitch{0.0F};
    /// Seconds, for the tearing pattern.
    float time{0.0F};
    /// Whip-pan blur: the motion of the last moment, in target pixels.
    float whipX{0.0F};
    float whipY{0.0F};
    /// Mip levels of the bloom chain, clamped to [2, 4].
    int bloomLevels{3};
    /// Part of the target that holds light, in target pixels; every pass is
    /// scissored to it. Empty (right <= left) means the whole target. During
    /// the boot the window spans the work area but the light only the
    /// canvas, so this keeps the GPU cost proportional to what is drawn.
    RECT region{.left = 0, .top = 0, .right = 0, .bottom = 0};
};

/// The light layer's render targets, the bloom chain and the passes.
///
/// Created once per window on the UI thread. beginLight() and endLight()
/// are called from draw-list callbacks inside Dear ImGui's DX11 backend,
/// on the same immediate context, between ImGui's own draw commands.
class PostProcessor {
public:
    /// Compiles the shaders and creates the fixed pipeline state.
    /// @throws OverlayError if a shader does not compile or a state cannot be created.
    explicit PostProcessor(ID3D11Device& device);

    PostProcessor(const PostProcessor&) = delete;
    PostProcessor& operator=(const PostProcessor&) = delete;
    PostProcessor(PostProcessor&&) = delete;
    PostProcessor& operator=(PostProcessor&&) = delete;
    ~PostProcessor() = default;

    /// Settings for the next endLight().
    void setFrame(const PostSettings& settings) noexcept { settings_ = settings; }

    /// Redirect drawing to the cleared light layer, sized like the render
    /// target currently bound (the swap chain's back buffer), and switch to
    /// additive blending. Remembers that target for endLight().
    ///
    /// Runs inside a draw callback, where an exception would unwind through
    /// Dear ImGui's renderer, so a failure (the light targets cannot be
    /// created, e.g. out of video memory) is recorded in failure() instead
    /// and the frame continues on the swap chain; the window then turns the
    /// pipeline off and reports why.
    void beginLight(ID3D11DeviceContext& context);

    /// Bloom the light layer and composite it onto the remembered target.
    /// The caller restores ImGui's render state afterwards. Does nothing
    /// after a failure.
    void endLight(ID3D11DeviceContext& context);

    /// Why the pipeline stopped working; empty while it works.
    [[nodiscard]] const std::string& failure() const noexcept { return failure_; }

private:
    struct Target {
        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
        Microsoft::WRL::ComPtr<ID3D11RenderTargetView> view;
        Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> resource;
        UINT width{0};
        UINT height{0};
    };

    static constexpr std::size_t kMaxLevels = 4;

    void redirect(ID3D11DeviceContext& context);
    void ensureSize(UINT width, UINT height);
    [[nodiscard]] Target createTarget(UINT width, UINT height) const;
    static void bindPass(ID3D11DeviceContext& context, const Target& output,
                         ID3D11PixelShader& shader, ID3D11BlendState* blend);
    void updateConstants(ID3D11DeviceContext& context, const Target& source, float bloomNorm);
    /// Scissor to the light region as seen in a target of @p width x @p height.
    void scissorFor(ID3D11DeviceContext& context, UINT width, UINT height) const;
    void runPass(ID3D11DeviceContext& context, const Target& source, const Target& output,
                 ID3D11PixelShader& shader, ID3D11BlendState* blend);
    void composite(ID3D11DeviceContext& context, int levels);

    ID3D11Device* device_;  // non-owning: the window owns the device and outlives us
    DXGI_FORMAT format_;
    Microsoft::WRL::ComPtr<ID3D11VertexShader> fullscreen_;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> prefilter_;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> downsample_;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> upsample_;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> composite_;
    Microsoft::WRL::ComPtr<ID3D11Buffer> constants_;
    Microsoft::WRL::ComPtr<ID3D11SamplerState> sampler_;
    Microsoft::WRL::ComPtr<ID3D11BlendState> lightBlend_;
    Microsoft::WRL::ComPtr<ID3D11BlendState> addBlend_;
    Microsoft::WRL::ComPtr<ID3D11BlendState> overBlend_;
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> rasterizer_;
    Microsoft::WRL::ComPtr<ID3D11DepthStencilState> noDepth_;
    Target light_;
    std::array<Target, kMaxLevels> mips_;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> output_;
    UINT outputWidth_{0};
    UINT outputHeight_{0};
    PostSettings settings_;
    std::string failure_;
};

/// GPU time of a frame from D3D11 timestamp queries.
///
/// Results are read a few frames later without flushing (GetData with
/// DONOTFLUSH), so timing never stalls the CPU on the GPU; frames whose
/// results are not ready, or whose clock was disjoint, are simply skipped.
class GpuTimer {
public:
    /// @throws OverlayError if the queries cannot be created.
    explicit GpuTimer(ID3D11Device& device);

    /// Bracket the GPU work of one frame.
    void begin(ID3D11DeviceContext& context);
    void end(ID3D11DeviceContext& context);

    /// Milliseconds of the oldest finished frame, if one is ready.
    [[nodiscard]] std::optional<double> collect(ID3D11DeviceContext& context);

private:
    struct Slot {
        Microsoft::WRL::ComPtr<ID3D11Query> disjoint;
        Microsoft::WRL::ComPtr<ID3D11Query> start;
        Microsoft::WRL::ComPtr<ID3D11Query> stop;
        bool pending{false};
    };
    static constexpr std::size_t kSlots = 4;

    std::array<Slot, kSlots> slots_;
    std::size_t write_{0};
    std::size_t read_{0};
    /// True between a begin() that started a measurement and its end().
    bool open_{false};
};

}  // namespace jarvis::overlay
