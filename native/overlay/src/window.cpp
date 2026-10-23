#include "jarvis/overlay/window.hpp"

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/animator.hpp"
#include "jarvis/overlay/errors.hpp"
#include "jarvis/overlay/hud_data.hpp"
#include "jarvis/overlay/hud_layout.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/palette.hpp"
#include "jarvis/overlay/ui_events.hpp"

#include <imgui.h>
#include <imgui_impl_dx11.h>
#include <imgui_impl_win32.h>

#include "frame_clock.hpp"
#include "hud.hpp"
#include "hud_interact.hpp"
#include "postfx.hpp"
#include "surface.hpp"
#include "window_impl.hpp"
#include <d3d11.h>
#include <dxgi.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace jarvis::overlay {

namespace {

using Microsoft::WRL::ComPtr;
using Clock = OverlayModel::Clock;

constexpr std::wstring_view kBodyFont = L"\\Fonts\\segoeui.ttf";
constexpr std::wstring_view kDisplayFont = L"\\Fonts\\bahnschrift.ttf";
constexpr std::wstring_view kClassNamePrefix = L"JarvisOverlay.";
constexpr int kMaxOpacity = 255;
/// Length of the whip-pan streak at the fastest point of the hand-off,
/// 96-DPI pixels.
constexpr float kWhipLength = 70.0F;
constexpr std::size_t kAdapterNameChars = 30;
/// Light reaches this share of the canvas side beyond the boot canvas.
constexpr float kRegionMargin = 0.2F;
/// How long after the last mutation the renderer keeps the active frame rate.
constexpr std::chrono::seconds kActiveAfterChange{1};
constexpr std::array<D3D_FEATURE_LEVEL, 2> kFeatureLevels{D3D_FEATURE_LEVEL_11_0,
                                                          D3D_FEATURE_LEVEL_10_0};
/// Waitable timers count in 100 ns units; negative means relative.
constexpr long long kTimerTicksPerMicrosecond = 10;
constexpr double kMillisPerSecond = 1000.0;
/// Longest frame step the interaction fades integrate (after a hidden spell).
constexpr float kMaxInteractionStep = 0.1F;

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

Appearance initialAppearance(const OverlayConfig& config) noexcept {
    return Appearance{.theme = config.theme,
                      .accent = config.accent,
                      .intensity = config.intensity,
                      .size = config.size};
}

double millisecondsBetween(Clock::time_point from, Clock::time_point to) noexcept {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

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

/// The wall clock for the idle readout, formatted once per frame.
hud::HudClock localClock() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    if (localtime_s(&local, &now) != 0) {
        return {};
    }
    return hud::HudClock{.time = formatClock(local), .date = formatDate(local)};
}

/// The GPU's name as DXGI reports it, e.g. "NVIDIA GeForce RTX 4070".
std::string adapterName(ID3D11Device& device) {
    ComPtr<IDXGIDevice> dxgiDevice;
    ComPtr<IDXGIAdapter> adapter;
    DXGI_ADAPTER_DESC desc{};
    if (FAILED(device.QueryInterface(IID_PPV_ARGS(&dxgiDevice))) ||
        FAILED(dxgiDevice->GetAdapter(&adapter)) || FAILED(adapter->GetDesc(&desc))) {
        return "Direct3D 11";
    }
    return toUtf8(std::wstring_view{static_cast<const wchar_t*>(desc.Description)});
}

}  // namespace

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

// --- lifecycle (caller thread) -----------------------------------------------

OverlayWindow::Impl::Impl(OverlayConfig initial)
    : config(std::move(initial)),
      metrics(withOverride(panelMetrics(config.size), config.width, config.height)),
      model(config.limits),
      hudData(config.hudLimits, initialAppearance(config)),
      animator(Clock::now(), anim::AnimatorConfig{.intensity = config.intensity}),
      appearance(initialAppearance(config)),
      cornerSaved(config.position),
      expandedSaved(config.expandedBounds),
      orbSaved(config.orbPosition),
      orbTarget(config.startMinimized),
      autoMinimizer(config.autoMinimizeSeconds) {
    minimizedNow.store(config.startMinimized);
    hudData.setAutoMinimize(config.autoMinimizeSeconds);
    hudData.setHideInFullscreen(config.hideInFullscreen);
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
        createPostProcessing();
        describeSystem();
        hud = std::make_unique<hud::HudRenderer>(
            *device.Get(), themeFor(config.theme, config.accent), animator, post.get());
        frameTimer = createFrameTimer();
        refreshMonitors();
        resolvePlacements();
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
    // Double-clicks on the ring reset the HUD to its default corner.
    windowClass.style = CS_DBLCLKS;
    windowClass.lpfnWndProc = &Impl::wndProc;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursorA(nullptr, IDC_ARROW);
    windowClass.lpszClassName = className.c_str();
    if (RegisterClassExW(&windowClass) == 0) {
        className.clear();
        throw OverlayError("overlay window class registration failed: " + lastErrorText());
    }
}

void OverlayWindow::Impl::createWindow(DWORD exStyle) {
    const std::wstring title = toWide(config.title);
    auto* const window =
        CreateWindowExW(exStyle, className.c_str(), title.c_str(), WS_POPUP, 0, 0, metrics.width,
                        metrics.height, nullptr, nullptr, instance, this);
    if (window == nullptr) {
        throw OverlayError("overlay window creation failed: " + lastErrorText());
    }
    hwnd.store(window);
    placement = Placement{.x = 0, .y = 0, .width = metrics.width, .height = metrics.height};
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
    // An interactive HUD starts click-through too; the cursor poll opens
    // the window to the mouse only over the HUD's own shapes.
    const bool startThrough = config.clickThrough || config.interactive;
    if (config.transparent) {
        try {
            createWindow(compositionWindowStyle());
            setClickThrough(startThrough);
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
    setClickThrough(startThrough);
    surface = createLayeredSurface(hwnd.load(), *device.Get(), config.opacity);
}

void OverlayWindow::Impl::createPostProcessing() {
    gpuTimer = std::make_unique<GpuTimer>(*device.Get());
    if (!config.bloom) {
        bloomReason = "bloom disabled in the configuration";
        return;
    }
    try {
        post = std::make_unique<PostProcessor>(*device.Get());
        bloomActive.store(true);
    } catch (const OverlayError& error) {
        // Not swallowed: the reason is reported through bloomReason() and
        // logged by the Python facade; the HUD then draws its light
        // additively without post-processing (the ADR-0022 look).
        bloomReason = error.what();
    }
}

void OverlayWindow::Impl::describeSystem() {
    const std::string gpu = truncateUtf8Head(adapterName(*device.Get()), kAdapterNameChars);
    const std::string surfaceName = transparent.load() ? "DIRECTCOMPOSITION" : "LAYERED WINDOW";
    const std::string bloomName =
        post != nullptr ? std::to_string(animator.scale().bloomLevels) + "-LEVEL BLOOM" : "OFF";
    systemReadout = {
        ReadoutLine{.label = "GPU", .value = gpu},
        ReadoutLine{.label = "SURFACE", .value = surfaceName},
        ReadoutLine{.label = "POST FX", .value = bloomName},
    };
}

void OverlayWindow::Impl::initImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    // The window procedure shapes the cursor itself (move, resize, text).
    // NOLINTBEGIN(bugprone-signed-bitwise,hicpp-signed-bitwise) - ImGui flags are plain ints
    io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
    if (config.interactive) {
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    } else {
        io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
    }
    // NOLINTEND(bugprone-signed-bitwise,hicpp-signed-bitwise)
    fonts = loadFonts();
    hud::applyHudStyle(themeFor(config.theme, config.accent), dpiScale);

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
        const HudData data = hudData.snapshot();
        checkFullscreen(now, data.hideInFullscreen);
        const bool visible = snapshot.visible && !fullscreenHidden;
        updateVisibility(visible);
        if (visible) {
            renderFrame(snapshot, data, now);
        }
        const std::chrono::microseconds period = framePeriodFor(snapshot, visible);
        scheduledMs = std::chrono::duration<double, std::milli>(period).count();
        nextFrame = std::max(nextFrame + period, Clock::now());
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
    // Wakes for the timer or for any message (close, DPI change, input),
    // whichever comes first; no polling, no busy loop.
    MsgWaitForMultipleObjectsEx(1, &timer, INFINITE, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
}

void OverlayWindow::Impl::updateVisibility(bool visible) {
    if (visible == shown) {
        return;
    }
    if (!visible) {
        releaseFocus();
    }
    ShowWindow(hwnd.load(), visible ? SW_SHOWNOACTIVATE : SW_HIDE);
    shown = visible;
}

void OverlayWindow::Impl::renderFrame(const OverlaySnapshot& snapshot, const HudData& data,
                                      Clock::time_point now) {
    applyHudData(data);
    applyDemo();
    applyMinimizeRequest(data);
    const anim::AnimationFrame& frame = animator.advance(snapshot, now);
    const bool booting = frame.boot.has_value();
    syncBootClickThrough(booting);
    runAutoMinimize(snapshot, data, booting, now);
    advanceInteraction(now, booting);
    if (placementDirty) {
        resolvePlacements();
    }
    const FrameLayout layout = layoutFor(frame);
    applyPlacement(layout.window);
    syncWindowRegion(layout);
    updateHitMap(layout, booting);
    pollCursor();
    drawFrame(snapshot, frame, layout, data);
    recordFrame(now, Clock::now());
    collectGpuTime();
    collectPresentation(presentedAt);
}

void OverlayWindow::Impl::applyHudData(const HudData& data) {
    if (data.appearance == appearance) {
        return;
    }
    if (data.appearance.theme != appearance.theme || data.appearance.accent != appearance.accent) {
        hud->setTheme(themeFor(data.appearance.theme, data.appearance.accent));
        hud::applyHudStyle(hud->theme(), dpiScale);
    }
    animator.setIntensity(data.appearance.intensity);
    if (data.appearance.size != appearance.size) {
        metrics = withOverride(panelMetrics(data.appearance.size), config.width, config.height);
        placementDirty = true;
    }
    appearance = data.appearance;
}

void OverlayWindow::Impl::applyDemo() {
    const std::uint64_t revision = demoRevision.load();
    if (revision == seenDemo) {
        return;
    }
    seenDemo = revision;
    const std::scoped_lock lock(demoMutex);
    demoHover = demo.hover;
    if (demo.input.has_value()) {
        interaction.showDemoInput(*demo.input);
    }
}

void OverlayWindow::Impl::advanceInteraction(Clock::time_point now, bool booting) {
    const float dt = lastAdvance.has_value()
                         ? std::min(std::chrono::duration<float>(now - *lastAdvance).count(),
                                    kMaxInteractionStep)
                         : 0.0F;
    lastAdvance = now;
    const bool wanted = demoHover || hoverKind != HitKind::None || interaction.inputOpen() ||
                        interaction.confirming() || drag.mode != DragMode::None || keyboardFocus;
    if (wanted) {
        hoverSeen = now;
    }
    const bool linger = std::chrono::duration<float>(now - hoverSeen).count() < kHoverLingerSeconds;
    hover = approach(hover, linger ? 1.0F : 0.0F, dt, kHoverFadeSeconds);
    advanceOrb(dt, booting);
    // The two morphs run one after the other, never together: the panel
    // collapses before the HUD shrinks, and the orb grows back first.
    const bool expandTarget = interaction.expanded() && orbLinear <= 0.0F;
    expandLinear = approach(expandLinear, expandTarget ? 1.0F : 0.0F, dt, kMorphSeconds);
    expandedNow.store(interaction.expanded());
}

void OverlayWindow::Impl::drawFrame(const OverlaySnapshot& snapshot,
                                    const anim::AnimationFrame& frame, const FrameLayout& layout,
                                    const HudData& data) {
    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    const float opacity = transparent.load()
                              ? static_cast<float>(std::clamp(config.opacity, 0, kMaxOpacity)) /
                                    static_cast<float>(kMaxOpacity)
                              : 1.0F;
    const bool collapsed = layout.expand <= 0.0F;
    const hud::HudScene scene{.snapshot = snapshot,
                              .frame = frame,
                              .geometry = layout.geometry,
                              .panel = layout.panel,
                              .panelText = layout.panelText,
                              .canvas = layout.canvas,
                              .scale = dpiScale,
                              .opacity = opacity,
                              .textScale = metrics.textScale,
                              .systemReadout = systemReadout,
                              .clock = localClock(),
                              .expand = layout.expand,
                              .panelFrame = layout.frame,
                              .hover = config.interactive ? hover : 0.0F,
                              .inputOpen = collapsed && interaction.inputOpen(),
                              .confirming = collapsed && interaction.confirming(),
                              .orb = layout.orb};
    if (post != nullptr) {
        post->setFrame(postSettings(frame, layout));
    }
    hud->draw(*ImGui::GetBackgroundDrawList(), scene, fonts);
    // The orb and the morph towards it have no widgets: the window only
    // answers clicks on the orb's disc (window_input.cpp).
    if (config.interactive && !frame.boot.has_value() && layout.orb <= 0.0F) {
        runInteraction(scene, layout, data);
    }
    ImGui::Render();
    submitFrame();
}

void OverlayWindow::Impl::runInteraction(const hud::HudScene& scene, const FrameLayout& layout,
                                         const HudData& data) {
    const hud::InteractFrame in{.data = data,
                                .collapsed = layout.collapsed,
                                .expanded = layout.expanded,
                                .expand = layout.expand,
                                .hover = hover,
                                .scale = dpiScale,
                                .textScale = metrics.textScale,
                                .accent = hud->accent(scene),
                                .warning = hud->theme().warning,
                                .opacity = scene.opacity};
    const hud::InteractResult result = interaction.frame(in, fonts, hud->theme());
    if (result.takeFocus) {
        takeFocus();
    }
    if (result.releaseFocus) {
        releaseFocus();
    }
    if (result.minimize) {
        setOrbTarget(true, false, true);
    }
}

void OverlayWindow::Impl::submitFrame() {
    const Rgba& background = hud->theme().background;
    const std::array<float, 4> clear =
        clearColourFor(*surface, {background.r, background.g, background.b, background.a});
    ID3D11RenderTargetView* const target = surface->target();
    gpuTimer->begin(*context.Get());
    context->OMSetRenderTargets(1, &target, nullptr);
    context->ClearRenderTargetView(target, clear.data());
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
    gpuTimer->end(*context.Get());
    if (post != nullptr && !post->failure().empty()) {
        retirePostProcessing();
    }

    const HRESULT presented = surface->present();
    presentedAt = Clock::now();
    if (presented == DXGI_ERROR_DEVICE_REMOVED || presented == DXGI_ERROR_DEVICE_RESET) {
        // The GPU went away (driver update, remote session change). Rather
        // than recreate everything mid-conversation the window retires and
        // running() turns false so the Python side can fall back to text.
        quit = true;
    }
}

PostSettings OverlayWindow::Impl::postSettings(const anim::AnimationFrame& frame,
                                               const FrameLayout& layout) const noexcept {
    const float whip = frame.boot.has_value() ? frame.boot->whip * kWhipLength * dpiScale : 0.0F;
    RECT region{.left = 0, .top = 0, .right = 0, .bottom = 0};
    if (frame.boot.has_value()) {
        // The boot's light stays near its canvas (particles, shock ring and
        // the whip trail reach a little beyond it).
        const RectF& c = layout.canvas;
        const float margin = (std::min(c.width, c.height) * kRegionMargin) + whip;
        region = RECT{.left = static_cast<LONG>(c.x - margin),
                      .top = static_cast<LONG>(c.y - margin),
                      .right = static_cast<LONG>(c.right() + margin),
                      .bottom = static_cast<LONG>(c.bottom() + margin)};
    }
    return PostSettings{.bloomStrength = frame.bloom * hud->theme().bloom,
                        .glitch = frame.glitch,
                        .time = frame.time,
                        .whipX = layout.handoffX * whip,
                        .whipY = layout.handoffY * whip,
                        .bloomLevels = animator.scale().bloomLevels,
                        .region = region};
}

void OverlayWindow::Impl::retirePostProcessing() {
    // The light targets could not be (re)created mid-run, e.g. out of video
    // memory after a resize. Keep the overlay and drop the bloom.
    {
        const std::scoped_lock lock(statsMutex);
        bloomReason = post->failure();
    }
    bloomActive.store(false);
    hud = std::make_unique<hud::HudRenderer>(*device.Get(), hud->theme(), animator, nullptr);
    post.reset();
}

void OverlayWindow::Impl::collectGpuTime() {
    if (const std::optional<double> gpu = gpuTimer->collect(*context.Get()); gpu.has_value()) {
        const std::scoped_lock lock(statsMutex);
        stats.recordGpu(*gpu);
    }
}

void OverlayWindow::Impl::collectPresentation(Clock::time_point presented) {
    const std::optional<VBlankGrid> grid = compositionGrid();
    if (!grid.has_value() || grid->period.count() <= 0) {
        lastVblank.reset();
        return;
    }
    // DWM composes a frame on the first vblank after it was presented.
    // Landing times, not vblank indices: the period DWM reports is a
    // measurement and a dynamic refresh rate changes it, so only the gap
    // between two consecutive frames is counted in refreshes.
    const double period = std::chrono::duration<double>(grid->period).count();
    const double since = std::chrono::duration<double>(presented - grid->vblank).count();
    const double landing = std::chrono::duration<double>(grid->vblank - epoch).count() +
                           (std::ceil(since / period) * period);
    const std::optional<double> previous = std::exchange(lastVblank, landing);
    if (!previous.has_value() || landing <= *previous) {
        return;
    }
    const double refreshMs = period * kMillisPerSecond;
    const auto perFrame =
        static_cast<std::uint32_t>(std::max(std::lround(scheduledMs / refreshMs), 1L));
    const auto refreshes = static_cast<std::uint32_t>(std::lround((landing - *previous) / period));
    const std::scoped_lock lock(statsMutex);
    stats.recordPresentation(1, refreshes, perFrame, 1.0 / period);
}

void OverlayWindow::Impl::recordFrame(Clock::time_point start, Clock::time_point end) {
    const double interval =
        lastFrameStart.has_value() ? millisecondsBetween(*lastFrameStart, start) : 0.0;
    lastFrameStart = start;
    const std::scoped_lock lock(statsMutex);
    stats.record(millisecondsBetween(start, end), interval, scheduledMs);
}

std::chrono::microseconds OverlayWindow::Impl::framePeriodFor(const OverlaySnapshot& snapshot,
                                                              bool visible) {
    const auto now = Clock::now();
    const std::uint64_t hudRevision = hudData.snapshot().revision;
    if (snapshot.revision != lastRevision || hudRevision != lastHudRevision) {
        lastRevision = snapshot.revision;
        lastHudRevision = hudRevision;
        lastChange = now;
    }
    // Hover and morph fades, a drag, typing or the cursor over the HUD all
    // need the active rate; the idle corner HUD with the cursor elsewhere
    // stays at the idle rate (ADR-0023 budget).
    const bool fading = (hover > 0.0F && hover < 1.0F) ||
                        (expandLinear > 0.0F && expandLinear < 1.0F) ||
                        expandLinear != (interaction.expanded() ? 1.0F : 0.0F) || orbMorphing();
    const bool interacting =
        fading || drag.mode != DragMode::None || keyboardFocus || hoverKind != HitKind::None;
    const bool busy = animator.pace(snapshot) == anim::Pace::Active ||
                      (now - lastChange) < kActiveAfterChange || interacting;
    return framePeriod(busy && visible ? config.activeFps : config.idleFps);
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
    post.reset();
    gpuTimer.reset();
    surface.reset();
    context.Reset();
    device.Reset();
    destroyWindow();
    if (!className.empty()) {
        UnregisterClassW(className.c_str(), instance);
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

bool OverlayWindow::playBoot(BootReadout readout) {
    return impl_->model.playBoot(std::move(readout));
}

void OverlayWindow::replayBoot(BootReadout readout) {
    impl_->model.replayBoot(std::move(readout));
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

bool OverlayWindow::bloom() const noexcept {
    return impl_->bloomActive.load();
}

std::string OverlayWindow::bloomReason() const {
    const std::scoped_lock lock(impl_->statsMutex);
    return impl_->bloomReason;
}

anim::FrameStats OverlayWindow::frameStats() const {
    const std::scoped_lock lock(impl_->statsMutex);
    return impl_->stats.stats();
}

void OverlayWindow::resetFrameStats() {
    const std::scoped_lock lock(impl_->statsMutex);
    impl_->stats.reset();
}

std::vector<UiEvent> OverlayWindow::pollUiEvents() {
    return impl_->events.drain();
}

std::uint64_t OverlayWindow::droppedUiEvents() const {
    return impl_->events.dropped();
}

void OverlayWindow::injectUiEvent(UiEvent event) {
    impl_->events.push(std::move(event));
}

void OverlayWindow::setMuted(bool muted) {
    impl_->hudData.setMuted(muted);
}

void OverlayWindow::addHistory(HistoryEntry entry) {
    impl_->hudData.addHistory(std::move(entry));
}

void OverlayWindow::clearHistory() {
    impl_->hudData.clearHistory();
}

void OverlayWindow::setSettings(SettingsPage page) {
    impl_->hudData.setSettings(std::move(page));
}

void OverlayWindow::setAppearance(Appearance appearance) {
    impl_->hudData.setAppearance(appearance);
}

void OverlayWindow::setExpanded(bool expanded) {
    impl_->hudData.requestExpanded(expanded);
    impl_->expandedNow.store(expanded);
}

bool OverlayWindow::expanded() const noexcept {
    return impl_->expandedNow.load();
}

HudData OverlayWindow::hudData() const {
    return impl_->hudData.snapshot();
}

void OverlayWindow::setMinimized(bool minimized) {
    impl_->hudData.requestMinimized(minimized);
    impl_->minimizedNow.store(minimized);
}

bool OverlayWindow::minimized() const noexcept {
    return impl_->minimizedNow.load();
}

void OverlayWindow::setAutoMinimize(float seconds) {
    impl_->hudData.setAutoMinimize(seconds);
}

void OverlayWindow::setHideInFullscreen(bool hide) {
    impl_->hudData.setHideInFullscreen(hide);
}

bool OverlayWindow::fullscreenHidden() const noexcept {
    return impl_->fullscreenNow.load();
}

void OverlayWindow::setDemo(DemoOverrides demo) {
    {
        const std::scoped_lock lock(impl_->demoMutex);
        impl_->demo = std::move(demo);
    }
    impl_->demoRevision.fetch_add(1);
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
// - Dear ImGui keeps one global context per process, so one OverlayWindow at
//   a time is supported; the Python side only ever creates one.
// - The bloom pipeline is optional at every step: it can be switched off,
//   it may fail to compile or allocate at creation, and it may fail later
//   (light targets on a resize); each case leaves a reason in bloomReason()
//   and the HUD carries on drawing its light without post-processing.
// - GPU time comes from timestamp queries read back a few frames late
//   without flushing, so measuring it never stalls the UI thread.
// - Frames are paced by a high-resolution waitable timer, not by vsync:
//   Present(0) never blocks, the flip-model swap chain is shown by DWM on
//   the next vblank anyway, and a hidden or idle panel costs a few wakeups
//   per second and nothing more.
// - The interactive layer (ADR-0024) only adds work while it is used: the
//   idle corner HUD with the cursor elsewhere costs one GetCursorPos and a
//   hit test of a dozen shapes per frame, and stays at the idle rate.
// - Minimized, the window shrinks to the orb and draws nothing else
//   (window_orb.cpp, hud_orb.cpp): the idle orb keeps the idle rate and
//   its frame is a few glows and one tiny bloom pass.
// - User actions leave the UI thread only as UiEvents in a queue that
//   Python polls; settings, history and appearance arrive as data. The
//   UI thread therefore never waits for, or calls into, Python.
