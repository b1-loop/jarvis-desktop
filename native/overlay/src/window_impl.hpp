/// @file window_impl.hpp
/// @brief OverlayWindow's pimpl, shared by the window's translation units
/// (private).
///
/// window.cpp holds the lifecycle, the frame loop and the public facade;
/// window_input.cpp the window procedure, hit-testing, hover, dragging and
/// keyboard focus; window_place.cpp monitors, placement and the per-frame
/// layout; window_orb.cpp minimizing to the orb, auto-minimize and hiding
/// in full screen. Nothing outside those four files includes this header.

#pragma once

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/animator.hpp"
#include "jarvis/overlay/hud_data.hpp"
#include "jarvis/overlay/hud_layout.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/minimize_policy.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/pacing.hpp"
#include "jarvis/overlay/placement.hpp"
#include "jarvis/overlay/ui_events.hpp"
#include "jarvis/overlay/window.hpp"

#include "frame_clock.hpp"
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
    /// Linear HUD-to-orb morph progress (orb.hpp), 0 = HUD, 1 = orb.
    float orb{0.0F};
};

/// What a mouse drag is doing.
enum class DragMode : std::uint8_t { None, Move, Resize };

/// The three movable things, each with its own remembered placement.
enum class HudPanel : std::uint8_t { Corner, Expanded, Orb };

/// A drag in progress (UI thread only).
struct DragState {
    DragMode mode{DragMode::None};
    HudPanel panel{HudPanel::Corner};
    DragAnchor anchor;
    POINT start{.x = 0, .y = 0};
    Placement startRect;
    /// The cursor left the system drag rectangle. A press on the orb is a
    /// click (restore) until it does, then a drag.
    bool moved{true};
};

/// Seconds a dragged panel takes to ease onto the edge it snaps to (99 %).
inline constexpr float kSnapGlideSeconds = 0.2F;
/// Seconds a reset (double-click on the ring) takes to glide home (99 %).
inline constexpr float kResetGlideSeconds = 0.45F;

/// A panel easing from where it is drawn onto its real placement.
///
/// The placement itself is always exact (snapped, clamped, reported to
/// Python); only the drawing glides. While dragging, the drawn position is
/// the cursor's unsnapped position plus an offset that springs towards the
/// snap, so the panel eases onto an edge instead of jumping 16 px, and
/// eases off it again; after the release, or after a reset, it springs the
/// rest of the way.
struct SnapGlide {
    HudPanel panel{HudPanel::Corner};
    /// The unsnapped position the offset is measured from, screen pixels.
    float fromX{0.0F};
    float fromY{0.0F};
    /// Offset from (fromX, fromY) towards the placement.
    anim::Spring dx;
    anim::Spring dy;
};

/// The three panels where they are drawn this frame.
struct PanelRects {
    Placement corner;
    Placement expanded;
    Placement orb;
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
    std::atomic<bool> minimizedNow{false};
    std::atomic<bool> fullscreenNow{false};
    mutable std::mutex demoMutex;
    DemoOverrides demo;  // guarded by demoMutex
    std::atomic<std::uint64_t> demoRevision{0};
    /// Signalled by the facade after every mutation, so a change made while
    /// the loop sleeps out an idle frame is drawn on the next vblank slot.
    UniqueHandle wakeEvent;

    /// Wake the UI thread (any thread).
    void wake() const noexcept;

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
    /// The vblank slot (seconds after epoch) of the frame just scheduled.
    std::optional<double> lastSlot;
    /// The model's state for this frame; reused so a frame allocates nothing.
    OverlaySnapshot frameSnapshot;
    std::optional<Clock::time_point> lastFrameStart;
    /// The period the frame being drawn was scheduled with, ms (hitch detection).
    double scheduledMs{0.0};
    /// When the window was created; vblanks are counted from here.
    Clock::time_point epoch{Clock::now()};
    /// When the last frame was handed to DWM, and when (seconds after
    /// epoch) the vblank it landed on was.
    Clock::time_point presentedAt{};
    std::optional<double> lastVblank;

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
    std::optional<SnapGlide> glide;
    /// A reset's glide, started once the new placement is resolved.
    std::optional<SnapGlide> pendingResetGlide;
    float hover{0.0F};
    float expandLinear{0.0F};
    Clock::time_point hoverSeen{};
    std::optional<Clock::time_point> lastAdvance;
    std::uint64_t seenDemo{0};
    bool demoHover{false};

    // Minimize to the orb (ADR-0024, update).
    std::optional<SavedPlacement> orbSaved;
    /// The orb in screen pixels (no bleed), wherever it docks or was dragged.
    Placement orbPanel;
    /// Someone asked for the orb (the control, Python, auto-minimize).
    bool orbTarget{false};
    /// What the morph heads for: the orb waits for the boot and a collapse.
    bool orbGoal{false};
    float orbLinear{0.0F};
    std::uint64_t seenMinimizeRequest{0};
    AutoMinimizer autoMinimizer;
    bool fullscreenHidden{false};
    Clock::time_point nextForegroundCheck{};
    /// Size of the elliptic window region on the opaque fallback, if any.
    std::optional<SIZE> regionSize;

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
    void runFrame();
    /// Sleep until the next frame is due, a change arrives or a message does.
    void waitForFrame();
    /// Plan the next frame at @p fps: on the vblank grid when visible.
    void scheduleNext(int fps, bool visible);
    /// Bring the next frame forward to the next slot (after a change).
    void wakeSoon();
    [[nodiscard]] double secondsSinceEpoch(Clock::time_point at) const noexcept;
    [[nodiscard]] pacing::Grid gridSince(const VBlankGrid& grid) const noexcept;
    void updateVisibility(bool visible);
    void renderFrame(const OverlaySnapshot& snapshot, const HudData& data, Clock::time_point now);
    void drawFrame(const OverlaySnapshot& snapshot, const anim::AnimationFrame& frame,
                   const FrameLayout& layout, const HudData& data);
    void runInteraction(const hud::HudScene& scene, const FrameLayout& layout, const HudData& data);
    void submitFrame();
    void applyHudData(const HudData& data);
    void applyDemo();
    void advanceInteraction(Clock::time_point now, bool booting);

    // window_orb.cpp: minimize to the orb, auto-minimize, full-screen hiding.
    void applyMinimizeRequest(const HudData& data);
    void setOrbTarget(bool minimized, bool automatic, bool report);
    void runAutoMinimize(const OverlaySnapshot& snapshot, const HudData& data, bool booting,
                         Clock::time_point now);
    void advanceOrb(float dt, bool booting);
    [[nodiscard]] Placement orbHome() const;
    void syncWindowRegion(const FrameLayout& layout);
    void checkFullscreen(Clock::time_point now, bool enabled);
    [[nodiscard]] bool orbMorphing() const noexcept;
    void recordFrame(Clock::time_point start, Clock::time_point end);
    void collectGpuTime();
    /// Fold the vblank the frame presented at @p presented lands on into
    /// the cadence statistics (frame_clock.hpp).
    void collectPresentation(Clock::time_point presented);
    void retirePostProcessing();
    [[nodiscard]] PostSettings postSettings(const anim::AnimationFrame& frame,
                                            const FrameLayout& layout) const noexcept;
    /// The frame rate this frame calls for: active while anything moves.
    [[nodiscard]] int targetFps(const OverlaySnapshot& snapshot, const HudData& data, bool visible);
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
    void placeExpanded(FrameLayout& layout, const HudGeometry& panelHud,
                       const PanelRects& drawn) const;
    void placeOrb(FrameLayout& layout, const HudGeometry& panelHud, const PanelRects& drawn) const;
    /// Where @p panel is drawn: its placement, offset while it glides.
    [[nodiscard]] Placement shownRect(HudPanel panel) const;
    [[nodiscard]] PanelRects shownPanels() const;
    /// Spring the glide's offset on by @p dt; drop it once it has landed.
    void advanceGlide(float dt);
    void startResetGlide();
    void applyPlacement(const Placement& target);
    void resetPlacement(bool expanded);
    void reportPlacement(HudPanel panel);
    void forgetOrbPlacement();
    [[nodiscard]] const MonitorArea& monitorOf(const Placement& rect) const;
    /// The placement of @p panel (cornerPanel, expandedPanel or orbPanel).
    [[nodiscard]] Placement& panelRect(HudPanel panel) noexcept;
    [[nodiscard]] const Placement& panelRect(HudPanel panel) const noexcept;
    /// Which panel is fully on screen (mid-morph: the one it comes from).
    [[nodiscard]] HudPanel activePanel() const noexcept;

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
    /// The cursor moved beyond the system's drag threshold since the press.
    [[nodiscard]] bool leftDragRectangle(POINT cursor) const noexcept;
    void continueDrag();
    /// The drag put @p panel at @p free before snapping: glide towards the snap.
    void followSnap(HudPanel panel, const Placement& free);
    /// Grabbing a gliding panel: make where it is drawn its placement.
    void settleGlide(HudPanel panel);
    /// @p released: the button went up (a click on the orb restores);
    /// false when the capture was taken away.
    void endDrag(bool released);
    [[nodiscard]] bool onSetCursor() const;
    [[nodiscard]] std::optional<LRESULT> handleInput(UINT message, WPARAM wParam, LPARAM lParam);
    [[nodiscard]] std::optional<LRESULT> handleMouse(UINT message, LPARAM lParam);
    void onDpiChanged(WPARAM wParam);

    static LRESULT CALLBACK wndProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam);
    LRESULT handleMessage(HWND window, UINT message, WPARAM wParam, LPARAM lParam);
};

}  // namespace jarvis::overlay
