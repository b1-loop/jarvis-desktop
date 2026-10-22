/// @file surface.hpp
/// @brief Where the overlay's pixels go: a DirectComposition visual with
/// per-pixel alpha, or a layered window with whole-window alpha (private).
///
/// Only window.cpp includes this. Both surfaces own their swap chain and
/// render target view; the window owns the HWND and the Direct3D device and
/// outlives the surface.

#pragma once

#include <d3d11.h>
#include <windows.h>

#include <array>
#include <memory>

namespace jarvis::overlay {

/// A swap chain bound to the overlay window, plus how to clear it.
class RenderSurface {
public:
    RenderSurface() = default;
    virtual ~RenderSurface() = default;
    RenderSurface(const RenderSurface&) = delete;
    RenderSurface& operator=(const RenderSurface&) = delete;
    RenderSurface(RenderSurface&&) = delete;
    RenderSurface& operator=(RenderSurface&&) = delete;

    /// Match the back buffers to a new client size (physical pixels).
    virtual void resize(UINT width, UINT height) = 0;

    /// The view to draw the current frame into.
    [[nodiscard]] virtual ID3D11RenderTargetView* target() const noexcept = 0;

    /// Hand the frame to DWM; never blocks for vsync (frames are paced by
    /// the window's timer). Returns the DXGI result for device-loss checks.
    [[nodiscard]] virtual HRESULT present() = 0;

    /// True when uncovered pixels show the desktop (per-pixel alpha).
    [[nodiscard]] virtual bool transparent() const noexcept = 0;
};

/// Extended window styles each surface needs at CreateWindowEx time; they
/// cannot be changed afterwards for WS_EX_NOREDIRECTIONBITMAP.
[[nodiscard]] DWORD compositionWindowStyle() noexcept;
[[nodiscard]] DWORD layeredWindowStyle() noexcept;

/// Flip-model swap chain with premultiplied alpha on a DirectComposition
/// visual. @p window must have been created with compositionWindowStyle().
/// @throws OverlayError if DirectComposition or the swap chain is unavailable.
[[nodiscard]] std::unique_ptr<RenderSurface> createCompositionSurface(HWND window,
                                                                      ID3D11Device& device);

/// Blit-model swap chain on a layered window drawn at @p opacity (0-255).
/// @throws OverlayError if the swap chain cannot be created.
[[nodiscard]] std::unique_ptr<RenderSurface> createLayeredSurface(HWND window, ID3D11Device& device,
                                                                  int opacity);

/// Clear colour (premultiplied RGBA) for @p surface: fully transparent on a
/// composition surface, @p opaque on the layered fallback.
[[nodiscard]] std::array<float, 4> clearColourFor(const RenderSurface& surface,
                                                  const std::array<float, 4>& opaque) noexcept;

}  // namespace jarvis::overlay
