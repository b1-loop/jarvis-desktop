/// @file window_manager.hpp
/// @brief Active window lookup and top-level window management.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::os {

/// An opaque top-level window handle (an HWND widened to 64 bits so the
/// public API and Python never see Win32 types).
using WindowHandle = std::uint64_t;

/// What a tool needs to know about a window to describe or pick it.
struct WindowInfo {
    WindowHandle handle{0};
    std::string title;
    /// Executable file name without directory, e.g. "Code.exe".
    std::string processName;
    /// Full path of the executable; empty when the process is more
    /// privileged than we are and refuses the query.
    std::string executablePath;
    std::uint32_t pid{0};
};

/// Interface for window queries and management.
///
/// Exists so the Python facade and future macOS/Linux backends share one
/// contract and so tests can substitute a fake. Handles are validated on
/// every call: a window the user closed since listWindows() raises
/// WindowNotFoundError instead of acting on a recycled handle.
class WindowManager {
public:
    WindowManager() = default;
    WindowManager(const WindowManager&) = delete;
    WindowManager& operator=(const WindowManager&) = delete;
    WindowManager(WindowManager&&) = delete;
    WindowManager& operator=(WindowManager&&) = delete;
    virtual ~WindowManager() = default;

    /// The window that has keyboard focus, or nullopt when none has (lock
    /// screen, desktop transition).
    [[nodiscard]] virtual std::optional<WindowInfo> activeWindow() const = 0;

    /// Visible, titled, top-level windows in z-order (front first).
    [[nodiscard]] virtual std::vector<WindowInfo> listWindows() const = 0;

    /// Describe one window. Throws WindowNotFoundError if the handle is stale.
    [[nodiscard]] virtual WindowInfo describe(WindowHandle handle) const = 0;

    /// Bring @p handle to the foreground, restoring it if minimised.
    virtual void focus(WindowHandle handle) = 0;

    /// Focus the front-most window whose title contains @p substring
    /// (case-insensitive) and return it. Throws WindowNotFoundError.
    virtual WindowInfo focusByTitle(std::string_view substring) = 0;

    virtual void minimize(WindowHandle handle) = 0;
    virtual void maximize(WindowHandle handle) = 0;
    virtual void restore(WindowHandle handle) = 0;

    /// Ask the window to close (WM_CLOSE). The application may still prompt
    /// to save; this call returns immediately either way.
    virtual void close(WindowHandle handle) = 0;
};

/// Win32 implementation over EnumWindows / SetForegroundWindow / ShowWindow.
class Win32WindowManager final : public WindowManager {
public:
    [[nodiscard]] std::optional<WindowInfo> activeWindow() const override;
    [[nodiscard]] std::vector<WindowInfo> listWindows() const override;
    [[nodiscard]] WindowInfo describe(WindowHandle handle) const override;
    void focus(WindowHandle handle) override;
    WindowInfo focusByTitle(std::string_view substring) override;
    void minimize(WindowHandle handle) override;
    void maximize(WindowHandle handle) override;
    void restore(WindowHandle handle) override;
    void close(WindowHandle handle) override;
};

}  // namespace jarvis::os
