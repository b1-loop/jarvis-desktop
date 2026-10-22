/// @file hotkey_parser.cpp
/// @brief Pure string <-> Hotkey conversion. No Win32 headers: the virtual-key
/// codes are spelled out here so the parser is unit-testable anywhere.

#include "jarvis/os/hotkey.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace jarvis::os {

namespace {

using NamedKey = std::pair<std::string_view, std::uint32_t>;
using NamedModifier = std::pair<std::string_view, HotkeyModifier>;

/// Virtual-key codes from winuser.h for the keys people name in a hotkey.
/// Letters and digits are handled in code (their codes equal the ASCII
/// upper-case character), function keys by arithmetic. The first entry per
/// code is the canonical name used by formatHotkey().
constexpr std::array<NamedKey, 34> kNamedKeys{{
    {"space", 0x20},      {"enter", 0x0D},     {"return", 0x0D},    {"tab", 0x09},
    {"esc", 0x1B},        {"escape", 0x1B},    {"backspace", 0x08}, {"delete", 0x2E},
    {"del", 0x2E},        {"insert", 0x2D},    {"home", 0x24},      {"end", 0x23},
    {"pageup", 0x21},     {"pagedown", 0x22},  {"up", 0x26},        {"down", 0x28},
    {"left", 0x25},       {"right", 0x27},     {"pause", 0x13},     {"printscreen", 0x2C},
    {"scrolllock", 0x91}, {"numlock", 0x90},   {"capslock", 0x14},  {"plus", 0xBB},
    {"minus", 0xBD},      {"comma", 0xBC},     {"period", 0xBE},    {"slash", 0xBF},
    {"backslash", 0xDC},  {"semicolon", 0xBA}, {"quote", 0xDE},     {"backquote", 0xC0},
    {"lbracket", 0xDB},   {"rbracket", 0xDD},
}};

constexpr std::uint32_t kVkF1 = 0x70;
constexpr std::uint32_t kFunctionKeyCount = 24;

/// The first entry per modifier is the canonical spelling.
constexpr std::array<NamedModifier, 7> kModifiers{{
    {"ctrl", HotkeyModifier::Ctrl},
    {"control", HotkeyModifier::Ctrl},
    {"alt", HotkeyModifier::Alt},
    {"shift", HotkeyModifier::Shift},
    {"win", HotkeyModifier::Win},
    {"super", HotkeyModifier::Win},
    {"cmd", HotkeyModifier::Win},
}};

std::string normalise(std::string_view token) {
    std::string text;
    for (const char c : token) {
        if (std::isspace(static_cast<unsigned char>(c)) == 0) {
            text += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return text;
}

std::optional<HotkeyModifier> modifierFor(std::string_view token) {
    const auto found = std::ranges::find(kModifiers, token, &NamedModifier::first);
    return found == kModifiers.end() ? std::nullopt : std::optional{found->second};
}

std::optional<std::uint32_t> functionKeyFor(std::string_view token) {
    if (token.size() < 2 || token.size() > 3 || token.front() != 'f') {
        return std::nullopt;
    }
    std::uint32_t number = 0;
    for (const char digit : token.substr(1)) {
        if (std::isdigit(static_cast<unsigned char>(digit)) == 0) {
            return std::nullopt;
        }
        number = (number * 10) + static_cast<std::uint32_t>(digit - '0');
    }
    if (number < 1 || number > kFunctionKeyCount) {
        return std::nullopt;
    }
    return kVkF1 + number - 1;
}

bool isLetterOrDigitCode(std::uint32_t virtualKey) {
    return (virtualKey >= '0' && virtualKey <= '9') || (virtualKey >= 'A' && virtualKey <= 'Z');
}

std::optional<std::uint32_t> keyFor(std::string_view token) {
    if (token.size() == 1) {
        const auto code =
            static_cast<std::uint32_t>(std::toupper(static_cast<unsigned char>(token.front())));
        return isLetterOrDigitCode(code) ? std::optional{code} : std::nullopt;
    }
    if (const auto function = functionKeyFor(token)) {
        return function;
    }
    const auto named = std::ranges::find(kNamedKeys, token, &NamedKey::first);
    return named == kNamedKeys.end() ? std::nullopt : std::optional{named->second};
}

std::string keyName(std::uint32_t virtualKey) {
    if (isLetterOrDigitCode(virtualKey)) {
        return std::string{static_cast<char>(std::tolower(static_cast<int>(virtualKey)))};
    }
    if (virtualKey >= kVkF1 && virtualKey < kVkF1 + kFunctionKeyCount) {
        return "f" + std::to_string(virtualKey - kVkF1 + 1);
    }
    const auto named = std::ranges::find(kNamedKeys, virtualKey, &NamedKey::second);
    if (named == kNamedKeys.end()) {
        throw std::invalid_argument("virtual key " + std::to_string(virtualKey) + " has no name");
    }
    return std::string{named->first};
}

void applyToken(Hotkey& hotkey, bool& haveKey, std::string_view spec, const std::string& token) {
    if (const auto modifier = modifierFor(token)) {
        if ((hotkey.modifiers & toBits(*modifier)) != 0) {
            throw std::invalid_argument("duplicate modifier in hotkey '" + std::string{spec} + "'");
        }
        hotkey.modifiers |= toBits(*modifier);
        return;
    }
    if (const auto key = keyFor(token)) {
        if (haveKey) {
            throw std::invalid_argument("more than one key in hotkey '" + std::string{spec} + "'");
        }
        hotkey.virtualKey = *key;
        haveKey = true;
        return;
    }
    throw std::invalid_argument("unknown key '" + token + "' in hotkey '" + std::string{spec} +
                                "'");
}

}  // namespace

Hotkey parseHotkey(std::string_view spec) {
    Hotkey hotkey;
    bool haveKey = false;
    std::size_t start = 0;
    while (start <= spec.size()) {
        const std::size_t end = std::min(spec.find('+', start), spec.size());
        const std::string token = normalise(spec.substr(start, end - start));
        if (token.empty()) {
            throw std::invalid_argument("empty token in hotkey '" + std::string{spec} + "'");
        }
        applyToken(hotkey, haveKey, spec, token);
        start = end + 1;
    }
    if (!haveKey) {
        throw std::invalid_argument("hotkey '" + std::string{spec} + "' names no key");
    }
    return hotkey;
}

std::string formatHotkey(const Hotkey& hotkey) {
    std::string text;
    std::uint32_t emitted = 0;
    for (const auto& [name, modifier] : kModifiers) {
        const std::uint32_t bit = toBits(modifier);
        if ((hotkey.modifiers & bit) != 0 && (emitted & bit) == 0) {
            text += std::string{name} + "+";
            emitted |= bit;
        }
    }
    return text + keyName(hotkey.virtualKey);
}

}  // namespace jarvis::os
