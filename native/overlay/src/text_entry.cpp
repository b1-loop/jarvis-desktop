#include "text_entry.hpp"

#include <imgui.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>

namespace jarvis::overlay::hud {

namespace {

constexpr std::string_view kWhiteSpace = " \t\r\n\v\f";

/// Largest prefix of @p text of at most @p bytes that ends on a UTF-8
/// character boundary (never inside a multi-byte sequence).
std::size_t utf8Fit(std::string_view text, std::size_t bytes) {
    if (text.size() <= bytes) {
        return text.size();
    }
    constexpr unsigned char kContinuationMask = 0xC0;
    constexpr unsigned char kContinuation = 0x80;
    std::size_t cut = bytes;
    while (cut > 0 &&
           (static_cast<unsigned char>(text.at(cut)) & kContinuationMask) == kContinuation) {
        --cut;
    }
    return cut;
}

}  // namespace

std::string_view trimmed(std::string_view text) noexcept {
    const std::size_t first = text.find_first_not_of(kWhiteSpace);
    if (first == std::string_view::npos) {
        return {};
    }
    const std::size_t last = text.find_last_not_of(kWhiteSpace);
    return text.substr(first, last - first + 1);
}

TextEntry::TextEntry() : buffer_(kCapacity, '\0') {}

TextEntry::Result TextEntry::draw(const char* id, const char* hint, float width, bool focus) {
    Result result;
    if (focus) {
        ImGui::SetKeyboardFocusHere();
    }
    ImGui::SetNextItemWidth(width);
    const bool entered = ImGui::InputTextWithHint(id, hint, buffer_.data(), buffer_.size(),
                                                  ImGuiInputTextFlags_EnterReturnsTrue);
    result.active = ImGui::IsItemActive();
    if (entered) {
        const std::string_view line = trimmed(text());
        if (!line.empty()) {
            result.submitted = std::string{line};
        }
        clear();
    }
    // InputText itself drops the focus on Escape, so the field is already
    // inactive by now: what matters is that it just was deactivated by it.
    if ((result.active || ImGui::IsItemDeactivated()) &&
        ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
        result.cancelled = true;
    }
    return result;
}

void TextEntry::set(std::string_view text) {
    const std::size_t length = utf8Fit(text, kCapacity - 1);
    std::ranges::fill(buffer_, '\0');
    std::ranges::copy_n(text.begin(), static_cast<std::ptrdiff_t>(length), buffer_.begin());
}

void TextEntry::clear() noexcept {
    std::ranges::fill(buffer_, '\0');
}

std::string_view TextEntry::text() const noexcept {
    return {buffer_.data(), std::strlen(buffer_.data())};
}

}  // namespace jarvis::overlay::hud

// Design notes:
// - A fixed buffer rather than std::string with ImGui's resize callback:
//   a request is a sentence, 4 KB bounds what a careless paste can push
//   into the agent, and there is no reallocation while the user types.
// - Characters are not filtered: ImGui already drops control characters,
//   and anything printable (Swedish letters, emoji, CJK from the IME) is a
//   legitimate request; the font simply shows a fallback glyph for what it
//   does not cover, while the text sent to Python stays exact.
