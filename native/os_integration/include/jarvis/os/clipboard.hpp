/// @file clipboard.hpp
/// @brief Plain-text clipboard access.

#pragma once

#include <string>
#include <string_view>

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

/// Win32 implementation over OpenClipboard / CF_UNICODETEXT.
class Win32Clipboard final : public Clipboard {
public:
    [[nodiscard]] std::string text() const override;
    void setText(std::string_view utf8) override;
};

}  // namespace jarvis::os
