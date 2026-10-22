#include "jarvis/overlay/window.hpp"

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/errors.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"

#include <imgui.h>
#include <imgui_impl_dx11.h>
#include <imgui_impl_win32.h>

#include "hud.hpp"
#include "surface.hpp"
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
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

// The backend keeps this declaration behind '#if 0' in its header so the
// header stays free of <windows.h>; applications declare it themselves.
// NOLINTNEXTLINE(readability-identifier-naming) - symbol name is Dear ImGui's
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam,
                                                             LPARAM lParam);

namespace jarvis::overlay {

namespace {

using Microsoft::WRL::ComPtr;
using Clock = OverlayModel::Clock;

constexpr float kBaseDpi = 96.0F;
constexpr std::wstring_view kBodyFont = L"\\Fonts\\segoeui.ttf";
constexpr std::wstring_view kDisplayFont = L"\\Fonts\\bahnschrift.ttf";
constexpr std::wstring_view kClassNamePrefix = L"JarvisOverlay.";
constexpr std::uint64_t kWordMask = 0xFFFF;
constexpr int kMaxOpacity = 255;
constexpr unsigned kWordBits = 16;
/// Side of the square boot canvas, 96-DPI pixels.
constexpr int kBootCanvas = 440;
/// Transparent margin around the panel so glows and the wake ripple are not
/// clipped; only on the transparent surface, where it is invisible.
constexpr int kBleed = 20;
/// Panel padding, 96-DPI pixels.
constexpr float kPanelPadding = 14.0F;
/// How long after the last mutation the renderer keeps the active frame rate.
constexpr std::chrono::seconds kActiveAfterChange{1};
constexpr std::array<D3D_FEATURE_LEVEL, 2> kFeatureLevels{D3D_FEATURE_LEVEL_11_0,
                                                          D3D_FEATURE_LEVEL_10_0};
/// Waitable timers count in 100 ns units; negative means relative.
constexpr long long kTimerTicksPerMicrosecond = 10;

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

/// A font under %WINDIR% as UTF-8 for ImGui's loader; empty if it is missing.
std::string systemFontPath(std::wstring_view relative) {
    std::array<wchar_t, MAX_PATH> windowsDir{};
    const UINT length =
        GetWindowsDirectoryW(windowsDir.data(), static_cast<UINT>(windowsDir.size()));
    if (length == 0 || length >= windowsDir.size()) {
        return {};
    }
    std::wstring path{windowsDir.data(), length};
    path.append(relative);
    return std::filesystem::exists(path) ? toUtf8(path) : std::string{};
}

ImFont* loadFont(std::wstring_view relative) {
    const ImGuiIO& io = ImGui::GetIO();
    const std::string path = systemFontPath(relative);
    return path.empty() ? nullptr : io.Fonts->AddFontFromFileTTF(path.c_str());
}

/// Segoe UI for text and Bahnschrift for captions when Windows has them,
/// ImGui's embedded font for whatever is missing.
hud::HudFonts loadFonts() {
    ImFont* body = loadFont(kBodyFont);
    if (body == nullptr) {
        body = ImGui::GetIO().Fonts->AddFontDefault();
    }
    ImFont* display = loadFont(kDisplayFont);
    return hud::HudFonts{.body = body, .display = display != nullptr ? display : body};
}

std::chrono::microseconds framePeriod(int fps) noexcept {
    constexpr int kMicrosPerSecond = 1'000'000;
    return std::chrono::microseconds(kMicrosPerSecond / std::max(fps, 1));
}

double millisecondsBetween(Clock::time_point from, Clock::time_point to) noexcept {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

/// Closes a Win32 HANDLE on destruction.
struct HandleCloser {
    void operator()(HANDLE handle) const noexcept {
        if (handle != nullptr) {
            CloseHandle(handle);
        }
    }
};
using UniqueHandle = std::unique_ptr<std::remove_pointer_t<HANDLE>, HandleCloser>;

/// High-resolution waitable timer (Windows 10 1803+), a normal one before
/// that; both let the loop sleep precisely without raising the global
/// timer resolution or spinning.
UniqueHandle createFrameTimer() {
    constexpr DWORD kAccess = TIMER_ALL_ACCESS;
    HANDLE timer =
        CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, kAccess);
    if (timer == nullptr) {
        timer = CreateWaitableTimerExW(nullptr, nullptr, 0, kAccess);
    }
    if (timer == nullptr) {
        throw OverlayError("frame timer creation failed: " + lastErrorText());
    }
    return UniqueHandle{timer};
}

/// Primary monitor's work area (the screen minus the taskbar).
PixelRect primaryWorkArea() noexcept {
    MONITORINFO monitor{};
    monitor.cbSize = sizeof(monitor);
    GetMonitorInfoW(MonitorFromPoint(POINT{.x = 0, .y = 0}, MONITOR_DEFAULTTOPRIMARY), &monitor);
    return PixelRect{.left = monitor.rcWork.left,
                     .top = monitor.rcWork.top,
                     .right = monitor.rcWork.right,
                     .bottom = monitor.rcWork.bottom};
}

/// Where the window goes and what is drawn where, for one frame.
struct FrameLayout {
    Placement window;
    /// Everything below is in client coordinates of @c window.
    HudGeometry geometry;
    RectF panel;
    RectF panelText;
    RectF canvas;
};

}  // namespace

/// Everything that needs Win32, Direct3D or ImGui. A plain pimpl struct:
/// the members up to statsMutex are shared with the facade; everything
/// after uiThread is touched only by the UI thread.
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
    std::atomic<bool> transparent{false};
    std::string fallbackReason;  // written before `ready`, read-only afterwards
    mutable std::mutex statsMutex;
    anim::FrameStatsAccumulator stats;  // guarded by statsMutex

    std::promise<void> ready;
    std::jthread uiThread;
    std::wstring className;
    HINSTANCE instance{GetModuleHandleW(nullptr)};
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    std::unique_ptr<RenderSurface> surface;
    anim::OverlayAnimator animator{Clock::now()};
    std::unique_ptr<hud::HudRenderer> hud;
    hud::HudFonts fonts;
    UniqueHandle frameTimer;
    PixelRect workArea;
    Placement placement;
    float dpiScale{1.0F};
    bool win32BackendReady{false};
    bool dx11BackendReady{false};
    bool shown{false};
    bool forcedClickThrough{false};
    bool quit{false};
    std::uint64_t lastRevision{0};
    Clock::time_point lastChange{Clock::now()};
    Clock::time_point nextFrame{Clock::now()};
    std::optional<Clock::time_point> lastFrameStart;

private:
    void threadMain();
    void registerClass();
    void createWindow(DWORD exStyle);
    void destroyWindow();
    void createDevice();
    void createSurface();
    void initImGui();
    void messageLoop();
    void pumpMessages();
    void waitUntil(Clock::time_point deadline) const;
    void updateVisibility(bool visible);
    void renderFrame(const OverlaySnapshot& snapshot, Clock::time_point now);
    void drawFrame(const OverlaySnapshot& snapshot, const anim::AnimationFrame& frame,
                   const FrameLayout& layout);
    [[nodiscard]] FrameLayout layoutFor(const anim::AnimationFrame& frame) const;
    void applyPlacement(const Placement& target);
    void setClickThrough(bool on) const;
    void syncBootClickThrough(bool booting);
    void recordFrame(Clock::time_point start, Clock::time_point end);
    [[nodiscard]] std::chrono::microseconds framePeriodFor(const OverlaySnapshot& snapshot);
    void shutdown();

    static LRESULT CALLBACK wndProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam);
    LRESULT handleMessage(HWND window, UINT message, WPARAM wParam, LPARAM lParam);
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

// --- UI thread: creation -------------------------------------------------------

void OverlayWindow::Impl::threadMain() {
    // Thread-scoped so the Python process (and any Tk or pystray window it
    // owns) keeps whatever DPI awareness it already has.
    SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    try {
        registerClass();
        createDevice();
        createSurface();
        initImGui();
        hud = std::make_unique<hud::HudRenderer>(*device.Get(), themeFor(config.theme), animator);
        frameTimer = createFrameTimer();
        workArea = primaryWorkArea();
        applyPlacement(layoutFor(animator.frame()).window);
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

void OverlayWindow::Impl::registerClass() {
    // Unique per instance so two windows never fight over one class.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) - pointer identity as text
    const auto identity = reinterpret_cast<std::uintptr_t>(this);
    className = std::wstring{kClassNamePrefix} + std::to_wstring(identity);
    WNDCLASSEXW windowClass{};
    windowClass.cbSize = sizeof(windowClass);
    windowClass.lpfnWndProc = &Impl::wndProc;
    windowClass.hInstance = instance;
    windowClass.lpszClassName = className.c_str();
    if (RegisterClassExW(&windowClass) == 0) {
        className.clear();
        throw OverlayError("overlay window class registration failed: " + lastErrorText());
    }
}

void OverlayWindow::Impl::createWindow(DWORD exStyle) {
    const std::wstring title = toWide(config.title);
    auto* const window =
        CreateWindowExW(exStyle, className.c_str(), title.c_str(), WS_POPUP, 0, 0, config.width,
                        config.height, nullptr, nullptr, instance, this);
    if (window == nullptr) {
        throw OverlayError("overlay window creation failed: " + lastErrorText());
    }
    hwnd.store(window);
    placement = Placement{.x = 0, .y = 0, .width = config.width, .height = config.height};
    dpiScale = static_cast<float>(GetDpiForWindow(window)) / kBaseDpi;
}

void OverlayWindow::Impl::destroyWindow() {
    if (auto* const window = hwnd.exchange(nullptr); window != nullptr) {
        DestroyWindow(window);
    }
}

void OverlayWindow::Impl::createDevice() {
    // A headless or virtualised machine has no GPU; WARP is the software
    // rasteriser that ships with Windows and keeps the overlay (and CI) working.
    constexpr std::array<D3D_DRIVER_TYPE, 2> kDriverTypes{D3D_DRIVER_TYPE_HARDWARE,
                                                          D3D_DRIVER_TYPE_WARP};
    HRESULT result = E_FAIL;
    for (const D3D_DRIVER_TYPE driverType : kDriverTypes) {
        result = D3D11CreateDevice(nullptr, driverType, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                                   kFeatureLevels.data(), static_cast<UINT>(kFeatureLevels.size()),
                                   D3D11_SDK_VERSION, &device, nullptr, &context);
        if (SUCCEEDED(result)) {
            return;
        }
    }
    throw OverlayError("Direct3D 11 device creation failed on hardware and WARP (HRESULT " +
                       std::to_string(static_cast<long>(result)) + ")");
}

void OverlayWindow::Impl::createSurface() {
    if (config.transparent) {
        try {
            createWindow(compositionWindowStyle());
            setClickThrough(config.clickThrough);
            surface = createCompositionSurface(hwnd.load(), *device.Get());
            transparent.store(true);
            return;
        } catch (const OverlayError& error) {
            // Not swallowed: the reason is kept and reported through
            // fallbackReason(), and the layered path below is the
            // documented degradation (ADR-0022).
            fallbackReason = error.what();
            destroyWindow();
        }
    } else {
        fallbackReason = "per-pixel transparency disabled in the configuration";
    }
    createWindow(layeredWindowStyle());
    setClickThrough(config.clickThrough);
    surface = createLayeredSurface(hwnd.load(), *device.Get(), config.opacity);
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
    fonts = loadFonts();

    win32BackendReady = ImGui_ImplWin32_Init(hwnd.load());
    if (!win32BackendReady) {
        throw OverlayError("ImGui Win32 backend initialisation failed");
    }
    dx11BackendReady = ImGui_ImplDX11_Init(device.Get(), context.Get());
    if (!dx11BackendReady) {
        throw OverlayError("ImGui DirectX 11 backend initialisation failed");
    }
}

// --- UI thread: per frame --------------------------------------------------------

void OverlayWindow::Impl::messageLoop() {
    while (!quit) {
        pumpMessages();
        if (quit) {
            break;
        }
        const Clock::time_point now = Clock::now();
        const OverlaySnapshot snapshot = model.snapshot(now);
        updateVisibility(snapshot.visible);
        if (snapshot.visible) {
            renderFrame(snapshot, now);
        }
        nextFrame = std::max(nextFrame + framePeriodFor(snapshot), Clock::now());
        waitUntil(nextFrame);
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

void OverlayWindow::Impl::waitUntil(Clock::time_point deadline) const {
    const auto wait =
        std::chrono::duration_cast<std::chrono::microseconds>(deadline - Clock::now());
    if (wait.count() <= 0) {
        return;
    }
    LARGE_INTEGER due{};
    due.QuadPart = -(wait.count() * kTimerTicksPerMicrosecond);
    HANDLE timer = frameTimer.get();
    SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
    // Wakes for the timer or for any message (close, DPI change), whichever
    // comes first; no polling, no busy loop.
    MsgWaitForMultipleObjectsEx(1, &timer, INFINITE, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
}

void OverlayWindow::Impl::updateVisibility(bool visible) {
    if (visible == shown) {
        return;
    }
    ShowWindow(hwnd.load(), visible ? SW_SHOWNOACTIVATE : SW_HIDE);
    shown = visible;
}

void OverlayWindow::Impl::renderFrame(const OverlaySnapshot& snapshot, Clock::time_point now) {
    const anim::AnimationFrame& frame = animator.advance(snapshot, now);
    syncBootClickThrough(frame.boot.has_value());
    const FrameLayout layout = layoutFor(frame);
    applyPlacement(layout.window);
    drawFrame(snapshot, frame, layout);
    recordFrame(now, Clock::now());
}

void OverlayWindow::Impl::drawFrame(const OverlaySnapshot& snapshot,
                                    const anim::AnimationFrame& frame, const FrameLayout& layout) {
    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    const float opacity = transparent.load()
                              ? static_cast<float>(std::clamp(config.opacity, 0, kMaxOpacity)) /
                                    static_cast<float>(kMaxOpacity)
                              : 1.0F;
    const hud::HudScene scene{.snapshot = snapshot,
                              .frame = frame,
                              .geometry = layout.geometry,
                              .panel = layout.panel,
                              .panelText = layout.panelText,
                              .canvas = layout.canvas,
                              .scale = dpiScale,
                              .opacity = opacity};
    hud->draw(*ImGui::GetBackgroundDrawList(), scene, fonts);
    ImGui::Render();

    const Rgba& background = hud->theme().background;
    const std::array<float, 4> clear =
        clearColourFor(*surface, {background.r, background.g, background.b, background.a});
    ID3D11RenderTargetView* const target = surface->target();
    context->OMSetRenderTargets(1, &target, nullptr);
    context->ClearRenderTargetView(target, clear.data());
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

    const HRESULT presented = surface->present();
    if (presented == DXGI_ERROR_DEVICE_REMOVED || presented == DXGI_ERROR_DEVICE_RESET) {
        // The GPU went away (driver update, remote session change). Rather
        // than recreate everything mid-conversation the window retires and
        // running() turns false so the Python side can fall back to text.
        quit = true;
    }
}

FrameLayout OverlayWindow::Impl::layoutFor(const anim::AnimationFrame& frame) const {
    const Placement panelPx =
        anchorToCorner(config.corner, workArea, scaled(config.width, dpiScale),
                       scaled(config.height, dpiScale), scaled(config.margin, dpiScale));
    const RectF panel = toRectF(panelPx);
    const HudGeometry panelHud = panelGeometry(panel, kPanelPadding * dpiScale);
    FrameLayout layout{.window = panelPx,
                       .geometry = panelHud,
                       .panel = panel,
                       .panelText = panelHud.text,
                       .canvas = panel};
    if (frame.boot.has_value()) {
        const int side = scaled(kBootCanvas, dpiScale);
        const RectF boot = toRectF(centredIn(workArea, side, side));
        layout.canvas = lerpRect(boot, panel, frame.boot->handoff);
        layout.geometry = lerpGeometry(bootGeometry(boot), panelHud, frame.boot->handoff);
        // Transparent: one work-area-sized window, so the move is pure drawing.
        // Fallback: the opaque window itself follows the canvas.
        layout.window = transparent.load() ? Placement{.x = workArea.left,
                                                       .y = workArea.top,
                                                       .width = workArea.right - workArea.left,
                                                       .height = workArea.bottom - workArea.top}
                                           : enclosingPlacement(layout.canvas);
    } else if (transparent.load()) {
        layout.window = inflated(panelPx, scaled(kBleed, dpiScale));
    }
    const auto dx = static_cast<float>(-layout.window.x);
    const auto dy = static_cast<float>(-layout.window.y);
    layout.geometry = translated(layout.geometry, dx, dy);
    layout.panel = translated(layout.panel, dx, dy);
    layout.panelText = translated(layout.panelText, dx, dy);
    layout.canvas = translated(layout.canvas, dx, dy);
    return layout;
}

void OverlayWindow::Impl::applyPlacement(const Placement& target) {
    if (target == placement) {
        return;
    }
    placement = target;
    // WM_SIZE arrives synchronously and resizes the swap chain before the
    // frame is drawn, so the new size is used immediately.
    SetWindowPos(hwnd.load(), HWND_TOPMOST, target.x, target.y, target.width, target.height,
                 SWP_NOACTIVATE);
}

void OverlayWindow::Impl::setClickThrough(bool on) const {
    auto* const window = hwnd.load();
    const LONG_PTR current = GetWindowLongPtrW(window, GWL_EXSTYLE);
    // NOLINTBEGIN(bugprone-signed-bitwise,hicpp-signed-bitwise) - Win32 style flags are signed
    constexpr LONG_PTR kPassThrough = WS_EX_TRANSPARENT | WS_EX_LAYERED;
    // The fallback is always layered (that is its transparency); the
    // composition window only needs WS_EX_LAYERED for click-through.
    const bool layeredSurface = (current & WS_EX_NOREDIRECTIONBITMAP) == 0;
    LONG_PTR style = on ? (current | kPassThrough) : (current & ~WS_EX_TRANSPARENT);
    if (!on && !layeredSurface) {
        style &= ~WS_EX_LAYERED;
    }
    // NOLINTEND(bugprone-signed-bitwise,hicpp-signed-bitwise)
    SetWindowLongPtrW(window, GWL_EXSTYLE, style);
    if (on && !layeredSurface) {
        // A layered window stays invisible until its attributes are set;
        // full alpha leaves the composition visual's per-pixel alpha alone.
        SetLayeredWindowAttributes(window, 0, kMaxOpacity, LWA_ALPHA);
    }
}

void OverlayWindow::Impl::syncBootClickThrough(bool booting) {
    // The boot canvas sits in the middle of the screen; it must never eat a
    // click there, even when the panel itself is configured to take them.
    if (config.clickThrough || booting == forcedClickThrough) {
        return;
    }
    forcedClickThrough = booting;
    setClickThrough(booting);
}

void OverlayWindow::Impl::recordFrame(Clock::time_point start, Clock::time_point end) {
    const double interval =
        lastFrameStart.has_value() ? millisecondsBetween(*lastFrameStart, start) : 0.0;
    lastFrameStart = start;
    const std::scoped_lock lock(statsMutex);
    stats.record(millisecondsBetween(start, end), interval);
}

std::chrono::microseconds OverlayWindow::Impl::framePeriodFor(const OverlaySnapshot& snapshot) {
    const auto now = Clock::now();
    if (snapshot.revision != lastRevision) {
        lastRevision = snapshot.revision;
        lastChange = now;
    }
    const bool busy =
        animator.pace(snapshot) == anim::Pace::Active || (now - lastChange) < kActiveAfterChange;
    return framePeriod(busy && snapshot.visible ? config.activeFps : config.idleFps);
}

void OverlayWindow::Impl::shutdown() {
    // Reverse order of creation: ImGui backends, ImGui, the HUD's GPU
    // objects, the surface, the device, then the window and its class.
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
    hud.reset();
    surface.reset();
    context.Reset();
    device.Reset();
    destroyWindow();
    if (!className.empty()) {
        UnregisterClassW(className.c_str(), instance);
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
    if (ImGui::GetCurrentContext() != nullptr &&
        ImGui_ImplWin32_WndProcHandler(window, message, wParam, lParam) != 0) {
        return 1;
    }
    switch (message) {
        case WM_SIZE:
            if (wParam != SIZE_MINIMIZED && surface != nullptr) {
                surface->resize(lowWord(lParam), highWord(lParam));
            }
            return 0;
        case WM_DPICHANGED:
            dpiScale = static_cast<float>(highWord(static_cast<LPARAM>(wParam))) / kBaseDpi;
            return 0;  // the next frame's layout uses the new scale
        case WM_DISPLAYCHANGE:
        case WM_SETTINGCHANGE:
            workArea = primaryWorkArea();  // the work area or resolution changed
            return 0;
        case WM_MOUSEACTIVATE:
            return MA_NOACTIVATE;
        case WM_DESTROY:
            // Only the live window ends the loop; the composition attempt
            // that failed and was destroyed during creation must not.
            if (window == hwnd.load()) {
                PostQuitMessage(0);
            }
            return 0;
        case WM_NCDESTROY:
            if (window == hwnd.load()) {
                hwnd.store(nullptr);
            }
            return 0;
        default:
            return DefWindowProcW(window, message, wParam, lParam);
    }
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

bool OverlayWindow::playBoot() {
    return impl_->model.playBoot();
}

bool OverlayWindow::skipBoot() {
    return impl_->model.skipBoot();
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

bool OverlayWindow::transparent() const noexcept {
    return impl_->transparent.load();
}

std::string OverlayWindow::fallbackReason() const {
    return impl_->fallbackReason;
}

anim::FrameStats OverlayWindow::frameStats() const {
    const std::scoped_lock lock(impl_->statsMutex);
    return impl_->stats.stats();
}

void OverlayWindow::resetFrameStats() {
    const std::scoped_lock lock(impl_->statsMutex);
    impl_->stats.reset();
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
// - Transparency is chosen once, at creation: WS_EX_NOREDIRECTIONBITMAP
//   cannot be added or removed later, so a failed composition attempt
//   destroys that window and creates the layered fallback from scratch.
// - During boot the transparent window spans the work area and the handoff
//   to the corner is pure drawing; the window is resized exactly twice. The
//   opaque fallback instead moves its window with the canvas, which costs a
//   swap-chain resize per frame for under half a second.
// - Dear ImGui keeps one global context per process, so one OverlayWindow at
//   a time is supported; the Python side only ever creates one.
// - Frames are paced by a high-resolution waitable timer, not by vsync:
//   Present(0) never blocks, the flip-model swap chain is shown by DWM on
//   the next vblank anyway, and a hidden or idle panel costs a few wakeups
//   per second and nothing more.
