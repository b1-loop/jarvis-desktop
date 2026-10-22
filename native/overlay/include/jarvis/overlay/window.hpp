/// @file window.hpp
/// @brief Always-on-top status window on its own UI thread.

#pragma once

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/effects.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

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

    /// Close the window and join the UI thread. Idempotent.
    void close();

    [[nodiscard]] const OverlayConfig& config() const noexcept;

private:
    struct Impl;  // keeps Win32, Direct3D and ImGui out of the public header
    std::unique_ptr<Impl> impl_;
};

}  // namespace jarvis::overlay
