#include "jarvis/os/clipboard.hpp"

#include "jarvis/os/errors.hpp"
#include "jarvis/os/text_encoding.hpp"

#include "win32_util.hpp"
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <thread>

namespace jarvis::os {

namespace {

/// Another application may hold the clipboard open for a few milliseconds;
/// retrying briefly is what every well-behaved clipboard client does.
class ClipboardScope {
public:
    ClipboardScope() {
        constexpr int kAttempts = 10;
        constexpr auto kRetryDelay = std::chrono::milliseconds(10);
        for (int attempt = 0; attempt < kAttempts; ++attempt) {
            if (OpenClipboard(nullptr) != FALSE) {
                return;
            }
            std::this_thread::sleep_for(kRetryDelay);
        }
        win32::throwLastError("OpenClipboard");
    }
    ~ClipboardScope() { CloseClipboard(); }

    ClipboardScope(const ClipboardScope&) = delete;
    ClipboardScope& operator=(const ClipboardScope&) = delete;
    ClipboardScope(ClipboardScope&&) = delete;
    ClipboardScope& operator=(ClipboardScope&&) = delete;
};

/// GlobalLock/GlobalUnlock pair for a movable memory block.
class GlobalLockScope {
public:
    explicit GlobalLockScope(HGLOBAL block) : block_(block), pointer_(GlobalLock(block)) {}
    ~GlobalLockScope() {
        if (pointer_ != nullptr) {
            GlobalUnlock(block_);
        }
    }

    GlobalLockScope(const GlobalLockScope&) = delete;
    GlobalLockScope& operator=(const GlobalLockScope&) = delete;
    GlobalLockScope(GlobalLockScope&&) = delete;
    GlobalLockScope& operator=(GlobalLockScope&&) = delete;

    [[nodiscard]] void* get() const noexcept { return pointer_; }

private:
    HGLOBAL block_;
    void* pointer_;
};

}  // namespace

std::string Win32Clipboard::text() const {
    const ClipboardScope clipboard;
    if (IsClipboardFormatAvailable(CF_UNICODETEXT) == FALSE) {
        return {};
    }
    HANDLE data = GetClipboardData(CF_UNICODETEXT);
    if (data == nullptr) {
        win32::throwLastError("GetClipboardData");
    }
    const GlobalLockScope locked{data};
    if (locked.get() == nullptr) {
        win32::throwLastError("GlobalLock");
    }
    return wideToUtf8(static_cast<const wchar_t*>(locked.get()));
}

void Win32Clipboard::setText(std::string_view utf8) {
    const std::wstring wide = utf8ToWide(utf8);
    const std::size_t bytes = (wide.size() + 1) * sizeof(wchar_t);
    const HGLOBAL block = GlobalAlloc(GMEM_MOVEABLE, bytes);
    if (block == nullptr) {
        win32::throwLastError("GlobalAlloc");
    }
    {
        const GlobalLockScope locked{block};
        if (locked.get() == nullptr) {
            GlobalFree(block);
            win32::throwLastError("GlobalLock");
        }
        const std::span<wchar_t> destination{static_cast<wchar_t*>(locked.get()), wide.size() + 1};
        std::ranges::copy(wide, destination.begin());
        destination.back() = L'\0';
    }
    const ClipboardScope clipboard;
    EmptyClipboard();
    if (SetClipboardData(CF_UNICODETEXT, block) == nullptr) {
        GlobalFree(block);  // still ours; the clipboard only takes it on success
        win32::throwLastError("SetClipboardData");
    }
}

}  // namespace jarvis::os
