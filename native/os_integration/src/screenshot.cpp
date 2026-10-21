#include "jarvis/os/screenshot.hpp"

#include "jarvis/os/errors.hpp"
#include "jarvis/os/window_manager.hpp"

#include "win32_util.hpp"
#include <dwmapi.h>
#include <wincodec.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace jarvis::os {

namespace {

using Microsoft::WRL::ComPtr;

/// A rectangle in virtual-screen coordinates.
struct Region {
    int left;
    int top;
    int width;
    int height;
};

// --- GDI RAII ----------------------------------------------------------------

struct DeleteDcDeleter {
    void operator()(HDC dc) const noexcept { DeleteDC(dc); }
};
using UniqueMemoryDc = std::unique_ptr<std::remove_pointer_t<HDC>, DeleteDcDeleter>;

struct DeleteObjectDeleter {
    void operator()(HBITMAP bitmap) const noexcept { DeleteObject(bitmap); }
};
using UniqueBitmap = std::unique_ptr<std::remove_pointer_t<HBITMAP>, DeleteObjectDeleter>;

/// GetDC(nullptr)/ReleaseDC pair for the whole screen.
class ScreenDc {
public:
    ScreenDc() : dc_(GetDC(nullptr)) {
        if (dc_ == nullptr) {
            throw OsError("GetDC failed: no interactive desktop");
        }
    }
    ~ScreenDc() { ReleaseDC(nullptr, dc_); }

    ScreenDc(const ScreenDc&) = delete;
    ScreenDc& operator=(const ScreenDc&) = delete;
    ScreenDc(ScreenDc&&) = delete;
    ScreenDc& operator=(ScreenDc&&) = delete;

    [[nodiscard]] HDC get() const noexcept { return dc_; }

private:
    HDC dc_;
};

/// Pixels of a region, 32-bit BGRX, rows top-down, no padding.
struct Capture {
    int width{0};
    int height{0};
    std::vector<std::uint8_t> pixels;
};

Region primaryScreen() {
    return Region{.left = 0,
                  .top = 0,
                  .width = GetSystemMetrics(SM_CXSCREEN),
                  .height = GetSystemMetrics(SM_CYSCREEN)};
}

Region virtualScreen() {
    return Region{.left = GetSystemMetrics(SM_XVIRTUALSCREEN),
                  .top = GetSystemMetrics(SM_YVIRTUALSCREEN),
                  .width = GetSystemMetrics(SM_CXVIRTUALSCREEN),
                  .height = GetSystemMetrics(SM_CYVIRTUALSCREEN)};
}

/// The window's visible frame. DWM's extended bounds exclude the invisible
/// resize border that GetWindowRect includes on Windows 10 and 11.
Region windowRegion(HWND hwnd) {
    RECT rect{};
    const HRESULT result =
        DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, &rect, sizeof(rect));
    if (FAILED(result) && GetWindowRect(hwnd, &rect) == FALSE) {
        win32::throwLastError("GetWindowRect");
    }
    return Region{.left = rect.left,
                  .top = rect.top,
                  .width = rect.right - rect.left,
                  .height = rect.bottom - rect.top};
}

BITMAPINFO topDownBgrx(const Region& region) {
    BITMAPINFO info{};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = region.width;
    info.bmiHeader.biHeight = -region.height;  // negative: rows top-down, as PNG wants
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    return info;
}

Capture captureRegion(const Region& region) {
    if (region.width <= 0 || region.height <= 0) {
        throw OsError("nothing to capture: the region is empty");
    }
    const win32::DpiAwarenessScope dpi;
    const ScreenDc screen;
    const UniqueMemoryDc memory{CreateCompatibleDC(screen.get())};
    if (memory == nullptr) {
        win32::throwLastError("CreateCompatibleDC");
    }
    const BITMAPINFO info = topDownBgrx(region);
    void* bits = nullptr;
    const UniqueBitmap bitmap{
        CreateDIBSection(memory.get(), &info, DIB_RGB_COLORS, &bits, nullptr, 0)};
    if (bitmap == nullptr || bits == nullptr) {
        win32::throwLastError("CreateDIBSection");
    }
    HGDIOBJ previous = SelectObject(memory.get(), bitmap.get());
    // CAPTUREBLT includes layered (transparent) windows, which plain SRCCOPY skips.
    const BOOL copied = BitBlt(memory.get(), 0, 0, region.width, region.height, screen.get(),
                               region.left, region.top, SRCCOPY | CAPTUREBLT);
    SelectObject(memory.get(), previous);
    if (copied == FALSE) {
        win32::throwLastError("BitBlt");
    }
    const std::size_t byteCount =
        static_cast<std::size_t>(region.width) * static_cast<std::size_t>(region.height) * 4;
    const auto* start = static_cast<const std::uint8_t*>(bits);
    return Capture{.width = region.width,
                   .height = region.height,
                   .pixels = std::vector<std::uint8_t>(start, start + byteCount)};
}

// --- WIC PNG encoding ---------------------------------------------------------

ComPtr<IStream> memoryStream() {
    ComPtr<IStream> stream;
    win32::checkHresult("CreateStreamOnHGlobal", CreateStreamOnHGlobal(nullptr, TRUE, &stream));
    return stream;
}

void encodePng(const Capture& capture, IStream& stream) {
    ComPtr<IWICImagingFactory> factory;
    win32::checkHresult("CoCreateInstance(WICImagingFactory)",
                        CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                         IID_PPV_ARGS(&factory)));
    ComPtr<IWICBitmapEncoder> encoder;
    win32::checkHresult("CreateEncoder(PNG)",
                        factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder));
    win32::checkHresult("IWICBitmapEncoder::Initialize",
                        encoder->Initialize(&stream, WICBitmapEncoderNoCache));
    ComPtr<IWICBitmapFrameEncode> frame;
    win32::checkHresult("CreateNewFrame", encoder->CreateNewFrame(&frame, nullptr));
    win32::checkHresult("IWICBitmapFrameEncode::Initialize", frame->Initialize(nullptr));
    win32::checkHresult("SetSize", frame->SetSize(static_cast<UINT>(capture.width),
                                                  static_cast<UINT>(capture.height)));
    // 32bppBGR: the DIB's fourth byte is padding, not alpha, so the PNG is opaque RGB.
    WICPixelFormatGUID format = GUID_WICPixelFormat32bppBGR;
    win32::checkHresult("SetPixelFormat", frame->SetPixelFormat(&format));
    const UINT stride = static_cast<UINT>(capture.width) * 4;
    win32::checkHresult(
        "WritePixels",
        frame->WritePixels(static_cast<UINT>(capture.height), stride,
                           static_cast<UINT>(capture.pixels.size()),
                           const_cast<std::uint8_t*>(capture.pixels.data())));  // NOLINT
    win32::checkHresult("IWICBitmapFrameEncode::Commit", frame->Commit());
    win32::checkHresult("IWICBitmapEncoder::Commit", encoder->Commit());
}

std::vector<std::uint8_t> streamContents(IStream& stream) {
    HGLOBAL block = nullptr;
    win32::checkHresult("GetHGlobalFromStream", GetHGlobalFromStream(&stream, &block));
    ULARGE_INTEGER end{};
    const LARGE_INTEGER zero{};
    win32::checkHresult("IStream::Seek", stream.Seek(zero, STREAM_SEEK_CUR, &end));
    const auto size = static_cast<std::size_t>(end.QuadPart);  // NOLINT(*-union-access)
    const auto* start = static_cast<const std::uint8_t*>(GlobalLock(block));
    if (start == nullptr) {
        win32::throwLastError("GlobalLock");
    }
    std::vector<std::uint8_t> png(start, start + size);
    GlobalUnlock(block);
    return png;
}

std::vector<std::uint8_t> toPng(const Capture& capture) {
    const win32::ComScope com{win32::Apartment::MultiThreaded};
    const ComPtr<IStream> stream = memoryStream();
    encodePng(capture, *stream.Get());
    return streamContents(*stream.Get());
}

}  // namespace

std::vector<std::uint8_t> GdiScreenshotter::captureScreen(bool allMonitors) const {
    const win32::DpiAwarenessScope dpi;  // metrics must be read DPI-aware too
    return toPng(captureRegion(allMonitors ? virtualScreen() : primaryScreen()));
}

std::vector<std::uint8_t> GdiScreenshotter::captureWindow(WindowHandle handle) const {
    HWND hwnd = win32::requireWindow(handle);
    const win32::DpiAwarenessScope dpi;
    return toPng(captureRegion(windowRegion(hwnd)));
}

}  // namespace jarvis::os

// Design notes:
// - GDI + WIC are in every Windows installation; no third-party encoder or
//   DirectX device is needed for a few captures per minute. A Desktop
//   Duplication (DXGI) path would be the upgrade for video-rate capture.
// - WritePixels takes a non-const pointer but never writes through it; the
//   const_cast is confined to that one call and marked.
