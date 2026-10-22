#include "jarvis/overlay/window.hpp"

#include "jarvis/overlay/errors.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"

#include <imgui.h>
#include <imgui_impl_dx11.h>
#include <imgui_impl_win32.h>

#include "panel.hpp"
#include <d3d11.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

// The backend keeps this declaration behind '#if 0' in its header so the
// header stays free of <windows.h>; applications declare it themselves.
// NOLINTNEXTLINE(readability-identifier-naming) - symbol name is Dear ImGui's
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam,
                                                             LPARAM lParam);

namespace jarvis::overlay {

namespace {

using Microsoft::WRL::ComPtr;

constexpr float kBaseDpi = 96.0F;
constexpr float kFontSizePx = 16.0F;
constexpr std::wstring_view kFontRelativePath = L"\\Fonts\\segoeui.ttf";
constexpr std::wstring_view kClassNamePrefix = L"JarvisOverlay.";
constexpr std::uint64_t kWordMask = 0xFFFF;
constexpr int kMaxOpacity = 255;
constexpr unsigned kWordBits = 16;
/// How long after the last mutation the renderer keeps the active frame rate.
constexpr std::chrono::seconds kActiveAfterChange{1};
constexpr std::array<D3D_FEATURE_LEVEL, 2> kFeatureLevels{D3D_FEATURE_LEVEL_11_0,
                                                          D3D_FEATURE_LEVEL_10_0};

UINT lowWord(LPARAM value) noexcept {
    return static_cast<UINT>(static_cast<std::uint64_t>(value) & kWordMask);
}

UINT highWord(LPARAM value) noexcept {
    return static_cast<UINT>((static_cast<std::uint64_t>(value) >> kWordBits) & kWordMask);
}

std::string lastErrorText() {
    return "Win32 error " + std::to_string(GetLastError());
}

std::wstring toWide(std::string_view utf8) {
    if (utf8.empty()) {
        return {};
    }
    const std::string owned{utf8};  // the Win32 call wants a sized, contiguous buffer
    const int size = static_cast<int>(owned.size());
    const int needed = MultiByteToWideChar(CP_UTF8, 0, owned.c_str(), size, nullptr, 0);
    std::wstring wide(static_cast<std::size_t>(std::max(needed, 0)), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, owned.c_str(), size, wide.data(), needed);
    return wide;
}

std::string toUtf8(std::wstring_view wide) {
    if (wide.empty()) {
        return {};
    }
    const std::wstring owned{wide};
    const int size = static_cast<int>(owned.size());
    const int needed =
        WideCharToMultiByte(CP_UTF8, 0, owned.c_str(), size, nullptr, 0, nullptr, nullptr);
    std::string utf8(static_cast<std::size_t>(std::max(needed, 0)), '\0');
    WideCharToMultiByte(CP_UTF8, 0, owned.c_str(), size, utf8.data(), needed, nullptr, nullptr);
    return utf8;
}

/// Segoe UI as UTF-8 for ImGui's file loader; empty if Windows has no such font.
std::string systemFontPath() {
    std::array<wchar_t, MAX_PATH> windowsDir{};
    const UINT length =
        GetWindowsDirectoryW(windowsDir.data(), static_cast<UINT>(windowsDir.size()));
    if (length == 0 || length >= windowsDir.size()) {
        return {};
    }
    std::wstring path{windowsDir.data(), length};
    path.append(kFontRelativePath);
    return std::filesystem::exists(path) ? toUtf8(path) : std::string{};
}

/// Segoe UI when Windows has it, ImGui's embedded font otherwise.
void loadFont() {
    const ImGuiIO& io = ImGui::GetIO();
    const std::string path = systemFontPath();
    const ImFont* font = path.empty() ? nullptr : io.Fonts->AddFontFromFileTTF(path.c_str());
    if (font == nullptr) {
        io.Fonts->AddFontDefault();
    }
}

std::chrono::milliseconds framePeriod(int fps) noexcept {
    constexpr int kMillisPerSecond = 1000;
    return std::chrono::milliseconds(kMillisPerSecond / std::max(fps, 1));
}

}  // namespace

/// Everything that needs Win32, Direct3D or ImGui. A plain pimpl struct:
/// the first four members are shared with the facade; everything after
/// uiThread is touched only by the UI thread.
struct OverlayWindow::Impl {
    explicit Impl(OverlayConfig initial);
    ~Impl();

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

    void close();

    OverlayConfig config;
    OverlayModel model;
    std::atomic<bool> running{false};
    std::atomic<HWND> hwnd{nullptr};

    std::promise<void> ready;
    std::jthread uiThread;
    std::wstring className;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGISwapChain> swapChain;
    ComPtr<ID3D11RenderTargetView> renderTarget;
    Theme theme;
    float dpiScale{1.0F};
    bool win32BackendReady{false};
    bool dx11BackendReady{false};
    bool shown{false};
    bool quit{false};
    std::uint64_t lastRevision{0};
    OverlayModel::Clock::time_point lastChange{OverlayModel::Clock::now()};
    OverlayModel::Clock::time_point nextFrame{OverlayModel::Clock::now()};

private:
    void threadMain();
    void createWindow();
    void createDevice();
    void createRenderTarget();
    void initImGui();
    void applyDpi() const;
    void placeWindow() const;
    void messageLoop();
    void pumpMessages();
    void updateVisibility(bool visible);
    void renderFrame(const OverlaySnapshot& snapshot);
    [[nodiscard]] std::chrono::milliseconds framePeriodFor(const OverlaySnapshot& snapshot);
    void shutdown();

    static LRESULT CALLBACK wndProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam);
    LRESULT handleMessage(HWND window, UINT message, WPARAM wParam, LPARAM lParam);
    void onResize(UINT width, UINT height);
    void onDpiChanged(UINT dpi);
};

// --- lifecycle (caller thread) -----------------------------------------------

OverlayWindow::Impl::Impl(OverlayConfig initial)
    : config(std::move(initial)), model(config.limits) {
    std::future<void> started = ready.get_future();
    uiThread = std::jthread([this] { threadMain(); });
    started.get();  // rethrows OverlayError from the UI thread
    running.store(true);
}

OverlayWindow::Impl::~Impl() {
    close();
}

void OverlayWindow::Impl::close() {
    if (auto* const window = hwnd.load(); window != nullptr) {
        PostMessageW(window, WM_CLOSE, 0, 0);
    }
    if (uiThread.joinable()) {
        uiThread.join();
    }
    running.store(false);
}

// --- UI thread -------------------------------------------------------------

void OverlayWindow::Impl::threadMain() {
    // Thread-scoped so the Python process (and any Tk or pystray window it
    // owns) keeps whatever DPI awareness it already has.
    SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    try {
        createWindow();
        createDevice();
        createRenderTarget();
        initImGui();
        placeWindow();
    } catch (...) {
        shutdown();
        ready.set_exception(std::current_exception());
        return;
    }
    ready.set_value();
    messageLoop();
    shutdown();
    running.store(false);
}

void OverlayWindow::Impl::createWindow() {
    // Unique per instance so two windows never fight over one class.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) - pointer identity as text
    const auto identity = reinterpret_cast<std::uintptr_t>(this);
    className = std::wstring{kClassNamePrefix} + std::to_wstring(identity);
    WNDCLASSEXW windowClass{};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.lpfnWndProc = &Impl::wndProc;
    windowClass.hInstance = GetModuleHandleW(nullptr);
    windowClass.lpszClassName = className.c_str();
    if (RegisterClassExW(&windowClass) == 0) {
        throw OverlayError("overlay window class registration failed: " + lastErrorText());
    }

    // NOLINTBEGIN(bugprone-signed-bitwise) - Win32 defines style flags as signed longs
    DWORD exStyle = WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_LAYERED;
    if (config.clickThrough) {
        exStyle |= WS_EX_TRANSPARENT;
    }
    // NOLINTEND(bugprone-signed-bitwise)
    const std::wstring title = toWide(config.title);
    auto* const window =
        CreateWindowExW(exStyle, className.c_str(), title.c_str(), WS_POPUP, 0, 0, config.width,
                        config.height, nullptr, nullptr, windowClass.hInstance, this);
    if (window == nullptr) {
        throw OverlayError("overlay window creation failed: " + lastErrorText());
    }
    hwnd.store(window);
    const auto alpha = static_cast<BYTE>(std::clamp(config.opacity, 0, kMaxOpacity));
    SetLayeredWindowAttributes(window, 0, alpha, LWA_ALPHA);
    dpiScale = static_cast<float>(GetDpiForWindow(window)) / kBaseDpi;
}

void OverlayWindow::Impl::createDevice() {
    DXGI_SWAP_CHAIN_DESC desc{};
    desc.BufferCount = 2;
    desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.OutputWindow = hwnd.load();
    desc.SampleDesc.Count = 1;
    desc.Windowed = TRUE;
    // Flip-model swap chains cannot be used on layered windows, so the
    // overlay uses the blit model that DWM composes with the window alpha.
    desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    // A headless or virtualised machine has no GPU; WARP is the software
    // rasteriser that ships with Windows and keeps the overlay (and CI) working.
    constexpr std::array<D3D_DRIVER_TYPE, 2> kDriverTypes{D3D_DRIVER_TYPE_HARDWARE,
                                                          D3D_DRIVER_TYPE_WARP};
    HRESULT result = E_FAIL;
    for (const D3D_DRIVER_TYPE driverType : kDriverTypes) {
        result = D3D11CreateDeviceAndSwapChain(
            nullptr, driverType, nullptr, 0, kFeatureLevels.data(),
            static_cast<UINT>(kFeatureLevels.size()), D3D11_SDK_VERSION, &desc, &swapChain, &device,
            nullptr, &context);
        if (SUCCEEDED(result)) {
            return;
        }
    }
    throw OverlayError("Direct3D 11 device creation failed on hardware and WARP (HRESULT " +
                       std::to_string(static_cast<long>(result)) + ")");
}

void OverlayWindow::Impl::createRenderTarget() {
    ComPtr<ID3D11Texture2D> backBuffer;
    if (FAILED(swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) {
        throw OverlayError("could not acquire the swap chain back buffer");
    }
    if (FAILED(device->CreateRenderTargetView(backBuffer.Get(), nullptr, &renderTarget))) {
        throw OverlayError("could not create the render target view");
    }
}

void OverlayWindow::Impl::initImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    // The window never receives input (WS_EX_TRANSPARENT / NOACTIVATE), so
    // ImGui must not try to shape the cursor or track the mouse.
    // NOLINTNEXTLINE(bugprone-signed-bitwise) - ImGui flags are plain ints by design
    io.ConfigFlags |= ImGuiConfigFlags_NoMouse | ImGuiConfigFlags_NoMouseCursorChange;
    loadFont();
    applyDpi();

    win32BackendReady = ImGui_ImplWin32_Init(hwnd.load());
    if (!win32BackendReady) {
        throw OverlayError("ImGui Win32 backend initialisation failed");
    }
    dx11BackendReady = ImGui_ImplDX11_Init(device.Get(), context.Get());
    if (!dx11BackendReady) {
        throw OverlayError("ImGui DirectX 11 backend initialisation failed");
    }
}

void OverlayWindow::Impl::applyDpi() const {
    // Rebuilding from a fresh style is the documented way to rescale: calling
    // ScaleAllSizes repeatedly would compound rounding.
    ImGuiStyle& style = ImGui::GetStyle();
    style = ImGuiStyle{};
    ImGui::StyleColorsDark(&style);
    style.WindowBorderSize = 0.0F;
    style.ScaleAllSizes(dpiScale);
    style.FontSizeBase = kFontSizePx;
    style.FontScaleDpi = dpiScale;
}

void OverlayWindow::Impl::placeWindow() const {
    auto* const window = hwnd.load();
    MONITORINFO monitor{};
    monitor.cbSize = sizeof(monitor);
    GetMonitorInfoW(MonitorFromWindow(window, MONITOR_DEFAULTTOPRIMARY), &monitor);
    const PixelRect workArea{.left = monitor.rcWork.left,
                             .top = monitor.rcWork.top,
                             .right = monitor.rcWork.right,
                             .bottom = monitor.rcWork.bottom};
    const Placement placement =
        anchorToCorner(config.corner, workArea, scaled(config.width, dpiScale),
                       scaled(config.height, dpiScale), scaled(config.margin, dpiScale));
    SetWindowPos(window, HWND_TOPMOST, placement.x, placement.y, placement.width, placement.height,
                 SWP_NOACTIVATE);
}

void OverlayWindow::Impl::messageLoop() {
    while (!quit) {
        pumpMessages();
        if (quit) {
            break;
        }
        const OverlaySnapshot snapshot = model.snapshot();
        updateVisibility(snapshot.visible);
        if (snapshot.visible) {
            renderFrame(snapshot);
        }
        nextFrame = std::max(nextFrame + framePeriodFor(snapshot), OverlayModel::Clock::now());
        const auto wait = std::chrono::duration_cast<std::chrono::milliseconds>(
            nextFrame - OverlayModel::Clock::now());
        // Wakes early for any message (close, DPI change) and otherwise
        // sleeps until the next frame is due.
        MsgWaitForMultipleObjectsEx(0, nullptr,
                                    static_cast<DWORD>(std::max<long long>(wait.count(), 0)),
                                    QS_ALLINPUT, MWMO_INPUTAVAILABLE);
    }
}

void OverlayWindow::Impl::pumpMessages() {
    MSG message{};
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE) != 0) {
        if (message.message == WM_QUIT) {
            quit = true;
            return;
        }
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

void OverlayWindow::Impl::updateVisibility(bool visible) {
    if (visible == shown) {
        return;
    }
    ShowWindow(hwnd.load(), visible ? SW_SHOWNOACTIVATE : SW_HIDE);
    shown = visible;
}

void OverlayWindow::Impl::renderFrame(const OverlaySnapshot& snapshot) {
    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    drawPanel(snapshot, theme, PanelMetrics::forScale(dpiScale));
    ImGui::Render();

    const std::array<float, 4> clear{theme.background.r, theme.background.g, theme.background.b,
                                     theme.background.a};
    context->OMSetRenderTargets(1, renderTarget.GetAddressOf(), nullptr);
    context->ClearRenderTargetView(renderTarget.Get(), clear.data());
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

    const HRESULT presented = swapChain->Present(0, 0);
    if (presented == DXGI_ERROR_DEVICE_REMOVED || presented == DXGI_ERROR_DEVICE_RESET) {
        // The GPU went away (driver update, remote session change). Rather
        // than recreate everything mid-conversation the window retires and
        // running() turns false so the Python side can fall back to text.
        quit = true;
    }
}

std::chrono::milliseconds OverlayWindow::Impl::framePeriodFor(const OverlaySnapshot& snapshot) {
    const auto now = OverlayModel::Clock::now();
    if (snapshot.revision != lastRevision) {
        lastRevision = snapshot.revision;
        lastChange = now;
    }
    const bool busy = snapshot.state != AssistantState::Idle || snapshot.wakeFlash > 0.0F ||
                      (now - lastChange) < kActiveAfterChange;
    return framePeriod(busy && snapshot.visible ? config.activeFps : config.idleFps);
}

void OverlayWindow::Impl::shutdown() {
    // Reverse order of creation: ImGui backends, ImGui, D3D views and
    // device, then the window and its class.
    if (dx11BackendReady) {
        ImGui_ImplDX11_Shutdown();
        dx11BackendReady = false;
    }
    if (win32BackendReady) {
        ImGui_ImplWin32_Shutdown();
        win32BackendReady = false;
    }
    if (ImGui::GetCurrentContext() != nullptr) {
        ImGui::DestroyContext();
    }
    renderTarget.Reset();
    swapChain.Reset();
    context.Reset();
    device.Reset();
    if (auto* const window = hwnd.exchange(nullptr); window != nullptr) {
        DestroyWindow(window);
    }
    if (!className.empty()) {
        UnregisterClassW(className.c_str(), GetModuleHandleW(nullptr));
    }
}

// --- window procedure --------------------------------------------------------

LRESULT CALLBACK OverlayWindow::Impl::wndProc(HWND window, UINT message, WPARAM wParam,
                                              LPARAM lParam) {
    if (message == WM_NCCREATE) {
        // Win32 passes the CREATESTRUCT and stores user data as integers.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
        const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lParam);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) - GWLP_USERDATA is a LONG_PTR
        const auto userData = reinterpret_cast<LONG_PTR>(create->lpCreateParams);
        SetWindowLongPtrW(window, GWLP_USERDATA, userData);
    }
    const LONG_PTR userData = GetWindowLongPtrW(window, GWLP_USERDATA);
    // Inverse of the store above.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    auto* self = reinterpret_cast<Impl*>(userData);
    if (self == nullptr) {
        return DefWindowProcW(window, message, wParam, lParam);
    }
    return self->handleMessage(window, message, wParam, lParam);
}

LRESULT OverlayWindow::Impl::handleMessage(HWND window, UINT message, WPARAM wParam,
                                           LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(window, message, wParam, lParam) != 0) {
        return 1;
    }
    switch (message) {
        case WM_SIZE:
            if (wParam != SIZE_MINIMIZED) {
                onResize(lowWord(lParam), highWord(lParam));
            }
            return 0;
        case WM_DPICHANGED:
            onDpiChanged(highWord(static_cast<LPARAM>(wParam)));
            return 0;
        case WM_DISPLAYCHANGE:
        case WM_SETTINGCHANGE:
            placeWindow();  // the work area or resolution changed
            return 0;
        case WM_MOUSEACTIVATE:
            return MA_NOACTIVATE;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_NCDESTROY:
            hwnd.store(nullptr);
            return 0;
        default:
            return DefWindowProcW(window, message, wParam, lParam);
    }
}

void OverlayWindow::Impl::onResize(UINT width, UINT height) {
    if (swapChain == nullptr || width == 0 || height == 0) {
        return;
    }
    renderTarget.Reset();
    swapChain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0);
    createRenderTarget();
}

void OverlayWindow::Impl::onDpiChanged(UINT dpi) {
    dpiScale = static_cast<float>(dpi) / kBaseDpi;
    if (ImGui::GetCurrentContext() != nullptr) {
        applyDpi();
    }
    placeWindow();
}

// --- public facade -----------------------------------------------------------

OverlayWindow::OverlayWindow(OverlayConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

OverlayWindow::~OverlayWindow() = default;

void OverlayWindow::setState(AssistantState state) {
    impl_->model.setState(state);
}

void OverlayWindow::pushLevel(float level) {
    impl_->model.pushLevel(level);
}

void OverlayWindow::setTranscript(std::string_view utf8) {
    impl_->model.setTranscript(utf8);
}

void OverlayWindow::setResponse(std::string_view utf8) {
    impl_->model.setResponse(utf8);
}

void OverlayWindow::setVisible(bool visible) {
    impl_->model.setVisible(visible);
}

void OverlayWindow::flashWakeWord() {
    impl_->model.flashWakeWord();
}

bool OverlayWindow::visible() const {
    return impl_->model.visible();
}

AssistantState OverlayWindow::state() const {
    return impl_->model.state();
}

bool OverlayWindow::running() const noexcept {
    return impl_->running.load();
}

OverlaySnapshot OverlayWindow::snapshot() const {
    return impl_->model.snapshot();
}

void OverlayWindow::close() {
    impl_->close();
}

const OverlayConfig& OverlayWindow::config() const noexcept {
    return impl_->config;
}

}  // namespace jarvis::overlay

// Design notes:
// - The UI thread owns the window, the device and the ImGui context; the
//   constructor blocks on a promise so failures surface as OverlayError at
//   the call site instead of as a window that never appears.
// - Layered window + LWA_ALPHA gives whole-panel translucency with the blit
//   swap chain; per-pixel alpha would need DirectComposition and a flip
//   chain, which is more machinery than a status panel justifies.
// - Dear ImGui keeps one global context per process, so one OverlayWindow at
//   a time is supported; the Python side only ever creates one.
// - Frames are paced by a deadline, not by vsync: Present(0) never blocks,
//   so a hidden or idle panel costs a few wakeups per second and nothing more.
