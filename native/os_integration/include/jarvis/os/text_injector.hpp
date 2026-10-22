/// @file text_injector.hpp
/// @brief Type text into whatever control currently has keyboard focus.

#pragma once

#include <string_view>

namespace jarvis::os {

/// Interface for text injection at the cursor.
class TextInjector {
public:
    TextInjector() = default;
    TextInjector(const TextInjector&) = delete;
    TextInjector& operator=(const TextInjector&) = delete;
    TextInjector(TextInjector&&) = delete;
    TextInjector& operator=(TextInjector&&) = delete;
    virtual ~TextInjector() = default;

    /// Type @p utf8 as if on the keyboard; newlines become Enter presses.
    /// With @p pressEnter an Enter press follows the text.
    virtual void typeText(std::string_view utf8, bool pressEnter) = 0;
};

/// SendInput with KEYEVENTF_UNICODE, so any character types correctly
/// regardless of the active keyboard layout.
class SendInputTextInjector final : public TextInjector {
public:
    void typeText(std::string_view utf8, bool pressEnter) override;
};

}  // namespace jarvis::os
