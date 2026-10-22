/// @file window_impl.hpp
/// @brief OverlayWindow's pimpl, shared by the window's translation units
/// (private).
///
/// window.cpp holds the lifecycle, the frame loop and the public facade;
/// window_input.cpp the window procedure, hit-testing, hover, dragging and
/// keyboard focus; window_place.cpp monitors, placement and the per-frame
/// layout. Nothing outside those three files includes this header.

#pragma once

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/animator.hpp"
#include "jarvis/overlay/hud_data.hpp"
#include "jarvis/overlay/hud_layout.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/placement.hpp"
#include "jarvis/overlay/ui_events.hpp"
#include "jarvis/overlay/window.hpp"

#include "hud.hpp"
#include "hud_interact.hpp"
#include "postfx.hpp"
#include "surface.hpp"
#include <d3d11.h>
#include <windows.h>
#include <wrl/client.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace jarvis::overlay {

inline constexpr float kBaseDpi = 96.0F;
/// Side of the square boot canvas, 96-DPI pixels.
inline constexpr int kBootCanvas = 520;
/// Transparent margin around the panel so glows, the wake ripple and the
/// particle field are not clipped; only on the transparent surface, where
/// it is invisible (and click-through like the rest of the window).
inline constexpr int kBleed = 44;
/// Panel padding, 96-DPI pixels.
inline constexpr float kPanelPadding = 14.0F;

/// Closes a Win32 HANDLE on destruction.
struct HandleCloser {
    void operator()(HANDLE handle) const noexcept {
        if (handle != nullptr) {
            CloseHandle(handle);
        }
    }
};
using UniqueHandle = std::unique_ptr<std::remove_pointer_t<HANDLE>, HandleCloser>;

/// Where the window goes and what is drawn where, for one frame.
struct FrameLayout {
    Placement window;
    /// Everything below is in client coordinates of @c window.
    HudGeometry geometry;
    RectF panel;
    RectF panelText;
    RectF canvas;
    /// Direction of the hand-off (boot centre to panel centre), unit length.
    float handoffX{0.0F};
    float handoffY{0.0F};
    /// Eased expand/collapse progress and the panel frame at that point.
    float expand{0.0F};
    RectF frame;
    CollapsedLayout collapsed;
    ExpandedLayout expanded;
};

/// What a mouse drag is doing.
enum class DragMode : std::uint8_t { None, Move, Resize };

/// A drag in progress (UI thread only).
struct DragState {
    DragMode mode{DragMode::None};
    /// Which panel: the corner HUD or the expanded panel.
    bool expanded{false};
    DragAnchor anchor;
    POINT start{.x = 0, .y = 0};
    Placement startRect;
};

/// UTF-8 to UTF-16 for Win32 calls.
[[nodiscard]] std::wstring toWide(std::string_view utf8);
/// UTF-16 from Win32 to UTF-8.
[[nodiscard]] std::string toUtf8(std::wstring_view wide);
/// "Win32 error N" for the calling thread's last error.
[[nodiscard]] std::string lastErrorText();

/// Everything that needs Win32, Direct3D or ImGui. A plain pimpl struct:
/// the members up to hudData are shared with the facade (each is
/// thread-safe or written before `ready`); everything after uiThread is
/// touched only by the UI thread.
struct OverlayWindow::Impl {
    using Clock = OverlayModel::Clock;

    explicit Impl(OverlayConfig initial);
    ~Impl();

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

    void close();

    OverlayConfig config;
    /// The panel's size preset with any explicit width/height applied.
    PanelMetrics metrics;
    OverlayModel model;
    std::atomic<bool> running{false};
    std::atomic<HWND> hwnd{nullptr};
    std::atomic<bool> transparent{false};
    std::string fallbackReason;  // written before `ready`, read-only afterwards
    mutable std::mutex statsMutex;
    anim::FrameStatsAccumulator stats;  // guarded by statsMutex
    UiEventQueue events;
    HudDataModel hudData;
    std::atomic<bool> expandedNow{false};
    mutable std::mutex demoMutex;
    DemoOverrides demo;  // guarded by demoMutex
    std::atomic<std::uint64_t> demoRevision{0};

    std::promise<void> ready;
    std::jthread uiThread;

    // --- UI thread only from here -------------------------------------------
    std::wstring className;
    HINSTANCE instance{GetModuleHandleW(nullptr)};
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
    std::unique_ptr<RenderSurface> surface;
    anim::OverlayAnimator animator;
    std::unique_ptr<PostProcessor> post;
    std::unique_ptr<GpuTimer> gpuTimer;
    std::string bloomReason;  // guarded by statsMutex once the thread runs
    std::atomic<bool> bloomActive{false};
    std::vector<ReadoutLine> systemReadout;
    std::unique_ptr<hud::HudRenderer> hud;
    hud::HudInteraction interaction{events};
    hud::HudFonts fonts;
    UniqueHandle frameTimer;
    /// Primary monitor's work area (the default corner's reference).
    PixelRect workArea;
    std::vector<MonitorArea> monitors;
    Placement placement;
    float dpiScale{1.0F};
    bool win32BackendReady{false};
    bool dx11BackendReady{false};
    bool shown{false};
    bool forcedClickThrough{false};
    bool quit{false};
    std::uint64_t lastRevision{0};
    std::uint64_t lastHudRevision{0};
    Clock::time_point lastChange{Clock::now()};
    Clock::time_point nextFrame{Clock::now()};
    std::optional<Clock::time_point> lastFrameStart;

    // Interaction (ADR-0024).
    Appearance appearance;
    std::optional<SavedPlacement> cornerSaved;
    std::optional<SavedPlacement> expandedSaved;
    /// The visible panels in screen pixels (no bleed).
    Placement cornerPanel;
    Placement expandedPanel;
    bool placementDirty{true};
    HitMap hitMap;
    HitKind hoverKind{HitKind::None};
    /// The current frame's ring and text field, client coordinates.
    HudGeometry hitRing;
    std::optional<RectF> inputRect;
    /// WS_EX_TRANSPARENT is off (the cursor is over the HUD's shapes).
    bool mouseOpen{false};
    bool keyboardFocus{false};
    HWND previousForeground{nullptr};
    DragState drag;
    float hover{0.0F};
    float expandLinear{0.0F};
    Clock::time_point hoverSeen{};
    std::optional<Clock::time_point> lastAdvance;
    std::uint64_t seenDemo{0};
    bool demoHover{false};

private:
    // window.cpp: creation and the frame loop.
    void threadMain();
    void registerClass();
    void createWindow(DWORD exStyle);
    void destroyWindow();
    void createDevice();
    void createSurface();
    void createPostProcessing();
    void describeSystem();
    void initImGui();
    void messageLoop();
    void pumpMessages();
    void waitUntil(Clock::time_point deadline) const;
    void updateVisibility(bool visible);
    void renderFrame(const OverlaySnapshot& snapshot, Clock::time_point now);
    void drawFrame(const OverlaySnapshot& snapshot, const anim::AnimationFrame& frame,
                   const FrameLayout& layout, const HudData& data);
    void runInteraction(const hud::HudScene& scene, const FrameLayout& layout, const HudData& data);
    void submitFrame();
    void applyHudData(const HudData& data);
    void applyDemo();
    void advanceInteraction(Clock::time_point now);
    void recordFrame(Clock::time_point start, Clock::time_point end);
    void collectGpuTime();
    void retirePostProcessing();
    [[nodiscard]] PostSettings postSettings(const anim::AnimationFrame& frame,
                                            const FrameLayout& layout) const noexcept;
    [[nodiscard]] std::chrono::microseconds framePeriodFor(const OverlaySnapshot& snapshot);
    void shutdown();

    // window_place.cpp: monitors, placement, layout.
    void refreshMonitors();
    void resolvePlacements();
    [[nodiscard]] Placement defaultCorner() const;
    [[nodiscard]] Placement defaultExpanded() const;
    [[nodiscard]] float expandEased() const noexcept;
    [[nodiscard]] FrameLayout layoutFor(const anim::AnimationFrame& frame) const;
    void placeBoot(FrameLayout& layout, const anim::BootFrame& boot,
                   const HudGeometry& panelHud) const;
    void placeExpanded(FrameLayout& layout, const HudGeometry& panelHud) const;
    void applyPlacement(const Placement& target);
    void resetPlacement(bool expanded);
    void reportPlacement(bool expanded);
    [[nodiscard]] const MonitorArea& monitorOf(const Placement& rect) const;

    // window_input.cpp: input, hit-testing, focus, dragging.
    void setClickThrough(bool on) const;
    void syncBootClickThrough(bool booting);
    void updateHitMap(const FrameLayout& layout, bool booting);
    void pollCursor();
    void syncMouseThrough();
    void takeFocus();
    void releaseFocus();
    [[nodiscard]] HitKind hitAt(LPARAM screenPoint) const;
    [[nodiscard]] HitKind hitAtClient(POINT client) const;
    bool beginDrag(POINT client, bool doubleClick);
    void continueDrag();
    void endDrag();
    [[nodiscard]] bool onSetCursor() const;
    [[nodiscard]] std::optional<LRESULT> handleInput(UINT message, WPARAM wParam, LPARAM lParam);
    [[nodiscard]] std::optional<LRESULT> handleMouse(UINT message, LPARAM lParam);
    void onDpiChanged(WPARAM wParam);

    static LRESULT CALLBACK wndProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam);
    LRESULT handleMessage(HWND window, UINT message, WPARAM wParam, LPARAM lParam);
};

}  // namespace jarvis::overlay
