/// @file text_entry.hpp
/// @brief The HUD's single-line text field over Dear ImGui (private).
///
/// Characters reach ImGui through the Win32 backend: WM_CHAR (already
/// composed by TranslateMessage from dead keys and keyboard layouts, or
/// committed by the IME) becomes io.AddInputCharacterUTF16, surrogate pairs
/// included, and InputText stores UTF-8. Ctrl+V uses ImGui's Win32
/// clipboard (CF_UNICODETEXT). This class adds what the HUD needs on top:
/// a bounded buffer, Enter to submit a trimmed line, Escape to cancel, and
/// focus on request. Only ImGui is used, so the Catch2 suite drives it with
/// a headless ImGui context.

#pragma once

#include <imgui.h>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::overlay::hud {

/// @p text without leading and trailing ASCII white space.
[[nodiscard]] std::string_view trimmed(std::string_view text) noexcept;

/// One text field with its own buffer.
class TextEntry {
public:
    /// Bytes of UTF-8 the field holds, terminator included; a paste beyond
    /// this is cut at a character boundary by ImGui.
    static constexpr std::size_t kCapacity = 4096;

    /// What happened to the field this frame.
    struct Result {
        /// The trimmed line when Enter was pressed on non-blank text.
        std::optional<std::string> submitted;
        /// Escape was pressed while editing.
        bool cancelled{false};
        /// The field has keyboard focus (is being edited).
        bool active{false};
    };

    TextEntry();

    /// Draw the field at the cursor, @p width pixels wide, inside the
    /// current ImGui window. @p focus moves keyboard focus into it.
    Result draw(const char* id, const char* hint, float width, bool focus);

    /// Replace the contents (demo hook and tests); cut to kCapacity - 1 bytes.
    void set(std::string_view text);
    void clear() noexcept;
    [[nodiscard]] std::string_view text() const noexcept;

private:
    std::vector<char> buffer_;
};

}  // namespace jarvis::overlay::hud
