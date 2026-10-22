/// @file window.hpp
/// @brief Always-on-top status window on its own UI thread.

#pragma once

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/effects.hpp"
#include "jarvis/overlay/hud_data.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/placement.hpp"
#include "jarvis/overlay/ui_events.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::overlay {

/// Geometry and behaviour of the window. Sizes are in device-independent
/// pixels at 96 DPI; the window scales them by the monitor's DPI itself.
struct OverlayConfig {
    Corner corner = Corner::BottomRight;
    /// Preset dimensions of the corner HUD (and its text scale).
    PanelSize size = PanelSize::Large;
    /// Explicit panel width / height overriding the preset; 0 keeps it.
    int width = 0;
    int height = 0;
    /// Distance from the two edges of the chosen corner.
    int margin = 24;
    /// Let clicks fall through to whatever is underneath (WS_EX_TRANSPARENT).
    /// The boot sequence is always click-through, whatever this says.
    bool clickThrough = true;
    /// Overall opacity, 0 (invisible) to 255 (opaque): a global alpha on the
    /// transparent surface, the whole-window alpha on the fallback.
    int opacity = 235;
    /// Frame rate while something animates (boot, cross-fade, not Idle).
    int activeFps = 60;
    /// Frame rate while the idle ring only breathes.
    int idleFps = 15;
    /// Prefer per-pixel transparency (DirectComposition); false forces the
    /// whole-window-alpha fallback, which is also used when composition fails.
    bool transparent = true;
    /// Palette of the HUD.
    ThemeName theme = ThemeName::Cyan;
    /// How rich the HUD is: particle count, bloom strength and motion.
    anim::Intensity intensity = anim::Intensity::Normal;
    /// Run the GPU bloom pipeline; false (or a pipeline that fails to start)
    /// draws the light additively without post-processing.
    bool bloom = true;
    /// Window title; never shown (frameless) but visible to window tools.
    std::string title = "Jarvis";
    ModelLimits limits;
    /// Make the HUD's own shapes interactive (hover controls, text field,
    /// dragging, the expanded view) while transparent pixels stay
    /// click-through (ADR-0024). False keeps the display-only ADR-0022
    /// behaviour, where clickThrough alone decides.
    bool interactive = true;
    /// Remembered position of the corner HUD; unset anchors it to @c corner.
    std::optional<SavedPlacement> position;
    /// Remembered position and logical size of the expanded panel; unset
    /// opens it at kExpandedWidth x kExpandedHeight next to the corner HUD.
    std::optional<SavedPlacement> expandedBounds;
    HudDataLimits hudLimits;
};

/// Demo and test hooks that show interactive states without touching the
/// real mouse or keyboard focus (scripts/demo_overlay.py records with them).
struct DemoOverrides {
    /// Show the hover controls as if the cursor were over the HUD.
    bool hover{false};
    /// Open the text field showing this text; unset leaves the field alone.
    std::optional<std::string> input;
};

/// The overlay window: Win32 + Direct3D 11 + Dear ImGui, on a thread of its own.
///
/// The constructor starts the UI thread, waits until the window and the
/// device exist, and throws OverlayError if either could not be created, so
/// a successfully constructed object is always on screen (or hidden by
/// choice). Every setter is cheap and thread-safe: it updates the model and
/// returns; the UI thread picks the change up on its next frame. No code
/// here ever calls back into the caller, which is what keeps the Python
/// binding free of GIL concerns.
///
/// Rendering goes to a DirectComposition visual with per-pixel alpha when
/// available, otherwise to a layered window with whole-window alpha; which
/// one was chosen is reported by transparent() and fallbackReason().
///
/// Destruction posts WM_CLOSE, joins the thread and releases the Direct3D
/// objects in reverse creation order on that thread.
class OverlayWindow {
public:
    explicit OverlayWindow(OverlayConfig config = {});
    ~OverlayWindow();

    OverlayWindow(const OverlayWindow&) = delete;
    OverlayWindow& operator=(const OverlayWindow&) = delete;
    OverlayWindow(OverlayWindow&&) = delete;
    OverlayWindow& operator=(OverlayWindow&&) = delete;

    void setState(AssistantState state);
    void pushLevel(float level);
    void setTranscript(std::string_view utf8);
    void setResponse(std::string_view utf8);
    void setVisible(bool visible);
    /// Pulse the ring outwards to acknowledge the wake word.
    void flashWakeWord();

    /// Play the boot sequence (about 4 s) once, with @p readout shown in
    /// its systems check. Non-blocking and idempotent: later calls return
    /// false and change nothing.
    bool playBoot(BootReadout readout = {});
    /// Cut the boot sequence short; it still glides into the corner.
    bool skipBoot();

    /// True when the window has per-pixel transparency (DirectComposition).
    [[nodiscard]] bool transparent() const noexcept;
    /// Why the fallback was used; empty when transparent() is true.
    [[nodiscard]] std::string fallbackReason() const;

    /// True while the GPU bloom pipeline renders the light layer.
    [[nodiscard]] bool bloom() const noexcept;
    /// Why there is no bloom; empty when bloom() is true.
    [[nodiscard]] std::string bloomReason() const;

    /// Smoothed frame timings measured on the UI thread.
    [[nodiscard]] anim::FrameStats frameStats() const;
    /// Start measuring afresh, e.g. at the beginning of a demo phase.
    void resetFrameStats();

    [[nodiscard]] bool visible() const;
    [[nodiscard]] AssistantState state() const;

    /// True from construction until close() or until the UI thread exits.
    [[nodiscard]] bool running() const noexcept;

    /// Copy of the current model state; for tests and the dashboard.
    [[nodiscard]] OverlaySnapshot snapshot() const;

    // --- interactive HUD (ADR-0024) -------------------------------------------

    /// Everything the user did since the last call, oldest first. Python
    /// polls this every ~30 ms; the UI thread never calls back.
    [[nodiscard]] std::vector<UiEvent> pollUiEvents();
    /// Events lost to the queue's overflow policy (see UiEventQueue).
    [[nodiscard]] std::uint64_t droppedUiEvents() const;
    /// Test hook: enqueue @p event as if the user had caused it.
    void injectUiEvent(UiEvent event);

    /// Show the microphone control as muted.
    void setMuted(bool muted);
    /// Append a line to the expanded view's conversation history.
    void addHistory(HistoryEntry entry);
    void clearHistory();
    /// Replace the settings page (choices and current values, as data).
    void setSettings(SettingsPage page);
    /// Switch theme, intensity and size live.
    void setAppearance(Appearance appearance);
    /// Expand to the panel (true) or collapse to the corner HUD (false).
    void setExpanded(bool expanded);
    /// Whether the HUD is (or is animating towards) the expanded panel.
    [[nodiscard]] bool expanded() const noexcept;
    /// Copy of the data pushed so far; for tests.
    [[nodiscard]] HudData hudData() const;
    /// Demo hook, see DemoOverrides.
    void setDemo(DemoOverrides demo);

    /// Close the window and join the UI thread. Idempotent.
    void close();

    [[nodiscard]] const OverlayConfig& config() const noexcept;

private:
    struct Impl;  // keeps Win32, Direct3D and ImGui out of the public header
    std::unique_ptr<Impl> impl_;
};

}  // namespace jarvis::overlay
