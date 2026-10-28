/// @file clipboard.hpp
/// @brief Plain-text clipboard access.

#pragma once

#include <chrono>
#include <functional>
#include <string>
#include <string_view>
#include <utility>

namespace jarvis::os {

/// Interface for reading and writing clipboard text.
class Clipboard {
public:
    Clipboard() = default;
    Clipboard(const Clipboard&) = delete;
    Clipboard& operator=(const Clipboard&) = delete;
    Clipboard(Clipboard&&) = delete;
    Clipboard& operator=(Clipboard&&) = delete;
    virtual ~Clipboard() = default;

    /// Current clipboard text, or an empty string if it holds no text.
    [[nodiscard]] virtual std::string text() const = 0;

    /// Replace the clipboard contents with @p utf8.
    virtual void setText(std::string_view utf8) = 0;
};

/// How Win32Clipboard opens the clipboard, and how long it retries.
///
/// Another process (a clipboard manager, a remote-desktop client, a second
/// app copying) may hold the clipboard for a few hundred ms. The default
/// exponential backoff waits about one second in total before giving up,
/// still well below what a spoken request notices.
struct ClipboardOpenPolicy {
    /// One attempt to open the clipboard. On failure it returns false with
    /// the thread's last error set, exactly like OpenClipboard. Empty means
    /// the real OpenClipboard(nullptr); tests inject failures here.
    std::function<bool()> open;
    /// Total attempts before giving up.
    int attempts = 8;
    /// Wait after the first failed attempt; doubled after each later one.
    std::chrono::milliseconds firstDelay{8};
};

/// Win32 implementation over OpenClipboard / CF_UNICODETEXT.
///
/// When OpenClipboard keeps failing with access denied (another process
/// holds the clipboard), text() and setText() throw ClipboardBusyError;
/// any other failure is a plain OsError.
class Win32Clipboard final : public Clipboard {
public:
    /// Uses the real OpenClipboard with the default retry policy.
    Win32Clipboard() = default;
    /// Uses @p policy, for tests that simulate a busy clipboard.
    explicit Win32Clipboard(ClipboardOpenPolicy policy) : policy_(std::move(policy)) {}

    [[nodiscard]] std::string text() const override;
    void setText(std::string_view utf8) override;

private:
    ClipboardOpenPolicy policy_;
};

}  // namespace jarvis::os
