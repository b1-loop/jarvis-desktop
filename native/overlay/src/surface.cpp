#include "surface.hpp"

#include "jarvis/overlay/errors.hpp"

#include <d3d11.h>
#include <dcomp.h>
#include <dxgi1_2.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace jarvis::overlay {

namespace {

using Microsoft::WRL::ComPtr;

constexpr UINT kBufferCount = 2;
constexpr int kMaxOpacity = 255;

void check(HRESULT result, std::string_view what) {
    if (FAILED(result)) {
        throw OverlayError(std::string{what} + " failed (HRESULT " +
                           std::to_string(static_cast<long>(result)) + ")");
    }
}

/// Physical client size of @p window, at least 1x1 (swap chains reject 0).
SIZE clientSize(HWND window) noexcept {
    RECT client{};
    GetClientRect(window, &client);
    return SIZE{.cx = std::max<LONG>(client.right - client.left, 1),
                .cy = std::max<LONG>(client.bottom - client.top, 1)};
}

ComPtr<IDXGIFactory2> factoryOf(ID3D11Device& device) {
    ComPtr<IDXGIDevice> dxgiDevice;
    check(device.QueryInterface(IID_PPV_ARGS(&dxgiDevice)), "IDXGIDevice query");
    ComPtr<IDXGIAdapter> adapter;
    check(dxgiDevice->GetAdapter(&adapter), "IDXGIDevice::GetAdapter");
    ComPtr<IDXGIFactory2> factory;
    check(adapter->GetParent(IID_PPV_ARGS(&factory)), "IDXGIFactory2 query");
    return factory;
}

ComPtr<ID3D11RenderTargetView> createView(ID3D11Device& device, IDXGISwapChain& swapChain) {
    ComPtr<ID3D11Texture2D> backBuffer;
    check(swapChain.GetBuffer(0, IID_PPV_ARGS(&backBuffer)), "swap chain GetBuffer");
    ComPtr<ID3D11RenderTargetView> view;
    check(device.CreateRenderTargetView(backBuffer.Get(), nullptr, &view),
          "CreateRenderTargetView");
    return view;
}

/// Shared by both surfaces: the swap chain, its view and resizing.
class SwapChainSurface : public RenderSurface {
public:
    SwapChainSurface(ID3D11Device& device, ComPtr<IDXGISwapChain> swapChain)
        : device_(&device),
          swapChain_(std::move(swapChain)),
          view_(createView(*device_, *swapChain_.Get())) {}

    void resize(UINT width, UINT height) override {
        if (width == 0 || height == 0) {
            return;
        }
        view_.Reset();
        check(swapChain_->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0),
              "IDXGISwapChain::ResizeBuffers");
        view_ = createView(*device_, *swapChain_.Get());
    }

    [[nodiscard]] ID3D11RenderTargetView* target() const noexcept override { return view_.Get(); }

    [[nodiscard]] HRESULT present() override { return swapChain_->Present(0, 0); }

private:
    ID3D11Device* device_;  // non-owning: the window owns the device and outlives us
    ComPtr<IDXGISwapChain> swapChain_;
    ComPtr<ID3D11RenderTargetView> view_;
};

/// Per-pixel alpha: DWM composes the premultiplied back buffer over the desktop.
class CompositionSurface final : public SwapChainSurface {
public:
    CompositionSurface(ID3D11Device& device, ComPtr<IDXGISwapChain1> swapChain,
                       ComPtr<IDCompositionDevice> composition, ComPtr<IDCompositionTarget> target,
                       ComPtr<IDCompositionVisual> visual)
        : SwapChainSurface(device, std::move(swapChain)),
          composition_(std::move(composition)),
          target_(std::move(target)),
          visual_(std::move(visual)) {}

    [[nodiscard]] bool transparent() const noexcept override { return true; }

private:
    // Released in reverse: visual, target, then the composition device.
    ComPtr<IDCompositionDevice> composition_;
    ComPtr<IDCompositionTarget> target_;
    ComPtr<IDCompositionVisual> visual_;
};

/// Whole-window alpha through SetLayeredWindowAttributes.
class LayeredSurface final : public SwapChainSurface {
public:
    using SwapChainSurface::SwapChainSurface;

    [[nodiscard]] bool transparent() const noexcept override { return false; }
};

}  // namespace

DWORD compositionWindowStyle() noexcept {
    // NOLINTNEXTLINE(bugprone-signed-bitwise) - Win32 defines style flags as signed longs
    return WS_EX_NOREDIRECTIONBITMAP | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE;
}

DWORD layeredWindowStyle() noexcept {
    // NOLINTNEXTLINE(bugprone-signed-bitwise) - Win32 defines style flags as signed longs
    return WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE;
}

std::unique_ptr<RenderSurface> createCompositionSurface(HWND window, ID3D11Device& device) {
    const SIZE size = clientSize(window);
    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width = static_cast<UINT>(size.cx);
    desc.Height = static_cast<UINT>(size.cy);
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;  // the format DirectComposition composes natively
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = kBufferCount;
    desc.Scaling = DXGI_SCALING_STRETCH;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    desc.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;

    ComPtr<IDXGISwapChain1> swapChain;
    check(factoryOf(device)->CreateSwapChainForComposition(&device, &desc, nullptr, &swapChain),
          "CreateSwapChainForComposition");

    ComPtr<IDXGIDevice> dxgiDevice;
    check(device.QueryInterface(IID_PPV_ARGS(&dxgiDevice)), "IDXGIDevice query");
    ComPtr<IDCompositionDevice> composition;
    check(DCompositionCreateDevice(dxgiDevice.Get(), IID_PPV_ARGS(&composition)),
          "DCompositionCreateDevice");
    ComPtr<IDCompositionTarget> target;
    check(composition->CreateTargetForHwnd(window, TRUE, &target), "CreateTargetForHwnd");
    ComPtr<IDCompositionVisual> visual;
    check(composition->CreateVisual(&visual), "IDCompositionDevice::CreateVisual");
    check(visual->SetContent(swapChain.Get()), "IDCompositionVisual::SetContent");
    check(target->SetRoot(visual.Get()), "IDCompositionTarget::SetRoot");
    check(composition->Commit(), "IDCompositionDevice::Commit");

    return std::make_unique<CompositionSurface>(
        device, std::move(swapChain), std::move(composition), std::move(target), std::move(visual));
}

std::unique_ptr<RenderSurface> createLayeredSurface(HWND window, ID3D11Device& device,
                                                    int opacity) {
    const SIZE size = clientSize(window);
    DXGI_SWAP_CHAIN_DESC desc{};
    desc.BufferCount = kBufferCount;
    desc.BufferDesc.Width = static_cast<UINT>(size.cx);
    desc.BufferDesc.Height = static_cast<UINT>(size.cy);
    desc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.OutputWindow = window;
    desc.SampleDesc.Count = 1;
    desc.Windowed = TRUE;
    // Flip-model swap chains cannot be used on layered windows, so the
    // fallback uses the blit model that DWM composes with the window alpha.
    desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    ComPtr<IDXGISwapChain> swapChain;
    check(factoryOf(device)->CreateSwapChain(&device, &desc, &swapChain), "CreateSwapChain");
    const auto alpha = static_cast<BYTE>(std::clamp(opacity, 0, kMaxOpacity));
    if (SetLayeredWindowAttributes(window, 0, alpha, LWA_ALPHA) == 0) {
        throw OverlayError("SetLayeredWindowAttributes failed: Win32 error " +
                           std::to_string(GetLastError()));
    }
    return std::make_unique<LayeredSurface>(device, std::move(swapChain));
}

std::array<float, 4> clearColourFor(const RenderSurface& surface,
                                    const std::array<float, 4>& opaque) noexcept {
    return surface.transparent() ? std::array<float, 4>{0.0F, 0.0F, 0.0F, 0.0F} : opaque;
}

}  // namespace jarvis::overlay

// Design notes:
// - Dear ImGui's DX11 blend state is "over" for colour and ONE/INV_SRC_ALPHA
//   for alpha. Drawing that onto a buffer cleared to (0, 0, 0, 0) yields
//   premultiplied colour, which is exactly what DXGI_ALPHA_MODE_PREMULTIPLIED
//   expects, so the backend needs no changes for per-pixel transparency.
// - B8G8R8A8 rather than R8G8B8A8: it is the format DirectComposition and
//   DWM use, so no conversion happens on the compositor side.
// - The composition objects are kept alive by the surface for as long as the
//   swap chain is on screen; dropping the visual would blank the window.
