/// @file window.hpp
/// @brief Always-on-top status window on its own UI thread.

#pragma once

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
    int width = 380;
    int height = 132;
    /// Distance from the two edges of the chosen corner.
    int margin = 24;
    /// Let clicks fall through to whatever is underneath (WS_EX_TRANSPARENT).
    bool clickThrough = true;
    /// Whole-window opacity, 0 (invisible) to 255 (opaque).
    int opacity = 235;
    /// Frame rate while something is happening (state not Idle, fresh data).
    int activeFps = 60;
    /// Frame rate once the panel has been static for a second.
    int idleFps = 10;
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
    void flashWakeWord();

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
