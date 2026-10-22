#include "jarvis/os/window_manager.hpp"

#include "jarvis/os/errors.hpp"
#include "jarvis/os/text_encoding.hpp"

#include "win32_util.hpp"
#include <dwmapi.h>
#include <windows.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::os {

namespace {

std::string windowTitle(HWND hwnd) {
    const int length = GetWindowTextLengthW(hwnd);
    if (length <= 0) {
        return {};
    }
    std::wstring title(static_cast<std::size_t>(length) + 1, L'\0');
    const int copied = GetWindowTextW(hwnd, title.data(), length + 1);
    title.resize(static_cast<std::size_t>(std::max(copied, 0)));
    return wideToUtf8(title);
}

/// Full image path of @p pid, or empty if the process is more privileged
/// than we are (an elevated installer, a protected system process).
std::string executablePath(DWORD pid) {
    const win32::UniqueHandle process{OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid)};
    if (process == nullptr) {
        return {};
    }
    std::array<wchar_t, MAX_PATH> buffer{};
    auto size = static_cast<DWORD>(buffer.size());
    if (QueryFullProcessImageNameW(process.get(), 0, buffer.data(), &size) == FALSE) {
        return {};
    }
    return wideToUtf8(std::wstring_view{buffer.data(), size});
}

std::string fileName(std::string_view path) {
    const std::size_t slash = path.find_last_of("\\/");
    return std::string{slash == std::string_view::npos ? path : path.substr(slash + 1)};
}

WindowInfo describeWindow(HWND hwnd) {
    DWORD pid = 0;
    GetWindowThreadProcessId(hwnd, &pid);
    WindowInfo info{.handle = win32::fromHwnd(hwnd),
                    .title = windowTitle(hwnd),
                    .processName = {},
                    .executablePath = executablePath(pid),
                    .pid = pid};
    info.processName = fileName(info.executablePath);
    return info;
}

/// DWM keeps suspended UWP apps and minimised-to-nothing windows "cloaked":
/// IsWindowVisible says yes, but the user cannot see them.
bool isCloaked(HWND hwnd) {
    DWORD cloaked = 0;
    const HRESULT result = DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, &cloaked, sizeof(cloaked));
    return SUCCEEDED(result) && cloaked != 0;
}

/// The windows a user would call "open": visible, titled, not a tool
/// window, not owned by another window (dialogs list under their owner).
bool isUserFacing(HWND hwnd) {
    if (IsWindowVisible(hwnd) == FALSE || GetWindow(hwnd, GW_OWNER) != nullptr) {
        return false;
    }
    constexpr auto kToolWindow = static_cast<DWORD>(WS_EX_TOOLWINDOW);
    const auto exStyle = static_cast<DWORD>(GetWindowLongPtrW(hwnd, GWL_EXSTYLE));
    if ((exStyle & kToolWindow) != 0 || isCloaked(hwnd)) {
        return false;
    }
    return GetWindowTextLengthW(hwnd) > 0;
}

BOOL CALLBACK collectWindow(HWND hwnd, LPARAM lParam) {
    // EnumWindows gives us back the LPARAM we passed in; it is our vector.
    // NOLINTNEXTLINE(performance-no-int-to-ptr,cppcoreguidelines-pro-type-reinterpret-cast)
    auto* windows = reinterpret_cast<std::vector<WindowInfo>*>(lParam);
    if (isUserFacing(hwnd)) {
        windows->push_back(describeWindow(hwnd));
    }
    return TRUE;
}

std::string lowered(std::string_view text) {
    std::string result{text};
    std::ranges::transform(result, result.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

/// SetForegroundWindow only succeeds for the thread that currently owns the
/// foreground (or was the last to receive input). Attaching our input queue
/// to that thread temporarily makes us count as that thread.
void bringToForeground(HWND hwnd) {
    const DWORD foregroundThread = GetWindowThreadProcessId(GetForegroundWindow(), nullptr);
    const DWORD ourThread = GetCurrentThreadId();
    AllowSetForegroundWindow(ASFW_ANY);
    const bool attached = foregroundThread != 0 && foregroundThread != ourThread &&
                          AttachThreadInput(ourThread, foregroundThread, TRUE) != FALSE;
    SetForegroundWindow(hwnd);
    BringWindowToTop(hwnd);
    if (attached) {
        AttachThreadInput(ourThread, foregroundThread, FALSE);
    }
}

void showWindow(WindowHandle handle, int command) {
    ShowWindow(win32::requireWindow(handle), command);  // returns the previous state, not success
}

}  // namespace

std::optional<WindowInfo> Win32WindowManager::activeWindow() const {
    HWND hwnd = GetForegroundWindow();
    if (hwnd == nullptr) {
        return std::nullopt;
    }
    return describeWindow(hwnd);
}

std::vector<WindowInfo> Win32WindowManager::listWindows() const {
    std::vector<WindowInfo> windows;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    if (EnumWindows(&collectWindow, reinterpret_cast<LPARAM>(&windows)) == FALSE) {
        win32::throwLastError("EnumWindows");
    }
    return windows;
}

WindowInfo Win32WindowManager::describe(WindowHandle handle) const {
    return describeWindow(win32::requireWindow(handle));
}

void Win32WindowManager::focus(WindowHandle handle) {
    HWND hwnd = win32::requireWindow(handle);
    if (IsIconic(hwnd) != FALSE) {
        ShowWindow(hwnd, SW_RESTORE);
    }
    bringToForeground(hwnd);
    if (GetForegroundWindow() != hwnd) {
        throw OsError("Windows refused to bring window '" + windowTitle(hwnd) +
                      "' to the foreground (foreground lock)");
    }
}

WindowInfo Win32WindowManager::focusByTitle(std::string_view substring) {
    const std::string needle = lowered(substring);
    for (const WindowInfo& window : listWindows()) {
        if (lowered(window.title).find(needle) != std::string::npos) {
            focus(window.handle);
            return window;
        }
    }
    throw WindowNotFoundError("no window with '" + std::string{substring} + "' in its title");
}

void Win32WindowManager::minimize(WindowHandle handle) {
    showWindow(handle, SW_MINIMIZE);
}

void Win32WindowManager::maximize(WindowHandle handle) {
    showWindow(handle, SW_MAXIMIZE);
}

void Win32WindowManager::restore(WindowHandle handle) {
    showWindow(handle, SW_RESTORE);
}

void Win32WindowManager::close(WindowHandle handle) {
    HWND hwnd = win32::requireWindow(handle);
    // PostMessage, not SendMessage: a "save changes?" prompt must not block
    // the caller.
    if (PostMessageW(hwnd, WM_CLOSE, 0, 0) == FALSE) {
        win32::throwLastError("PostMessage(WM_CLOSE)");
    }
}

}  // namespace jarvis::os

// Design notes:
// - listWindows() filters the way Alt+Tab does (visible, titled, unowned,
//   not cloaked) so the agent sees the same set the user would name.
// - focus() verifies the result and raises instead of pretending: Windows
//   may still deny the foreground to a background process, and a tool must
//   be able to tell the user that instead of claiming success.
