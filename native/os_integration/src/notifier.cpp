#include "jarvis/os/notifier.hpp"

#include "jarvis/os/errors.hpp"
#include "jarvis/os/text_encoding.hpp"

#include "message_loop_thread.hpp"
#include "win32_util.hpp"
#include <shellapi.h>
#include <windows.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace jarvis::os {

namespace {

constexpr const wchar_t* kWindowClass = L"JarvisNotifierWindow";
constexpr UINT kIconId = 1;
constexpr UINT kCallbackMessage = WM_APP + 2;
constexpr UINT kIconFlags =
    static_cast<UINT>(NIF_ICON) | static_cast<UINT>(NIF_TIP) | static_cast<UINT>(NIF_MESSAGE);
constexpr DWORD kBalloonFlags =
    static_cast<DWORD>(NIIF_INFO) | static_cast<DWORD>(NIIF_RESPECT_QUIET_TIME);

/// Copy @p text into a fixed Win32 buffer, truncating and NUL-terminating.
void copyTruncated(std::span<wchar_t> destination, std::wstring_view text) {
    const auto count = static_cast<std::ptrdiff_t>(std::min(text.size(), destination.size() - 1));
    const auto end = std::copy_n(text.begin(), count, destination.begin());
    *end = L'\0';
}

LRESULT CALLBACK notifierWindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    return DefWindowProcW(hwnd, message, wParam, lParam);
}

void registerWindowClass() {
    WNDCLASSW windowClass{};
    windowClass.lpfnWndProc = &notifierWindowProc;
    windowClass.hInstance = GetModuleHandleW(nullptr);
    windowClass.lpszClassName = kWindowClass;
    if (RegisterClassW(&windowClass) == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        win32::throwLastError("RegisterClass");
    }
}

/// The hidden window that owns the notification icon. Message-only windows
/// cannot own tray icons, so it is a real but never-shown top-level window.
HWND createHiddenWindow() {
    registerWindowClass();
    HWND hwnd = CreateWindowExW(0, kWindowClass, L"Jarvis", WS_OVERLAPPED, 0, 0, 0, 0, nullptr,
                                nullptr, GetModuleHandleW(nullptr), nullptr);
    if (hwnd == nullptr) {
        win32::throwLastError("CreateWindowEx");
    }
    return hwnd;
}

NOTIFYICONDATAW baseIconData(HWND hwnd) {
    NOTIFYICONDATAW data{};
    data.cbSize = sizeof(data);
    data.hWnd = hwnd;
    data.uID = kIconId;
    return data;
}

}  // namespace

struct BalloonNotifier::Impl {
    std::wstring tooltip;
    HWND hwnd{nullptr};
    // Declared last so it is destroyed first; its exit hook tears the
    // window down on the loop thread while hwnd is still valid.
    win32::MessageLoopThread loop{nullptr, [this] { destroyWindow(); }};

    void addIcon() const {
        NOTIFYICONDATAW data = baseIconData(hwnd);
        data.uFlags = kIconFlags;
        data.uCallbackMessage = kCallbackMessage;
        data.hIcon = LoadIconW(nullptr, IDI_INFORMATION);
        copyTruncated(data.szTip, tooltip);
        if (Shell_NotifyIconW(NIM_ADD, &data) == FALSE) {
            win32::throwLastError("Shell_NotifyIcon(NIM_ADD)");
        }
        // uVersion shares a union with uTimeout by Win32 design.
        data.uVersion = NOTIFYICON_VERSION_4;  // NOLINT(cppcoreguidelines-pro-type-union-access)
        Shell_NotifyIconW(NIM_SETVERSION, &data);
    }

    /// Runs on the loop thread as it exits.
    void destroyWindow() const noexcept {
        if (hwnd == nullptr) {
            return;
        }
        NOTIFYICONDATAW data = baseIconData(hwnd);
        Shell_NotifyIconW(NIM_DELETE, &data);
        DestroyWindow(hwnd);
    }

    void showBalloon(std::wstring_view title, std::wstring_view message) const {
        NOTIFYICONDATAW data = baseIconData(hwnd);
        data.uFlags = NIF_INFO;
        data.dwInfoFlags = kBalloonFlags;
        copyTruncated(data.szInfoTitle, title);
        copyTruncated(data.szInfo, message);
        if (Shell_NotifyIconW(NIM_MODIFY, &data) == FALSE) {
            win32::throwLastError("Shell_NotifyIcon(NIM_MODIFY)");
        }
    }
};

BalloonNotifier::BalloonNotifier(std::string_view tooltip) : impl_(std::make_unique<Impl>()) {
    impl_->tooltip = utf8ToWide(tooltip);
    impl_->loop.run([this] {
        impl_->hwnd = createHiddenWindow();
        impl_->addIcon();
    });
}

BalloonNotifier::~BalloonNotifier() = default;

void BalloonNotifier::notify(std::string_view title, std::string_view message) {
    const std::wstring wideTitle = utf8ToWide(title);
    const std::wstring wideMessage = utf8ToWide(message);
    impl_->loop.run(
        [this, &wideTitle, &wideMessage] { impl_->showBalloon(wideTitle, wideMessage); });
}

}  // namespace jarvis::os

// Design notes:
// - The window and icon live on a MessageLoopThread because Windows destroys
//   a window when its creating thread exits, and Python's executor threads
//   are not guaranteed to live as long as the notifier. The loop's exit hook
//   removes the icon and the window, so the destructor never has to throw.
// - NIIF_RESPECT_QUIET_TIME keeps Jarvis polite after a fresh login; Focus
//   Assist may still hide the toast, which is the user's choice, not an error.
