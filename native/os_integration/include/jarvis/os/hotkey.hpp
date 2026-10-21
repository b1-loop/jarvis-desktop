/// @file hotkey.hpp
/// @brief Global hotkeys: a pure parser and a thread-affine manager.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::os {

/// Modifier bits, numerically identical to the Win32 MOD_* flags so a parsed
/// hotkey can be handed to RegisterHotKey without translation.
enum class HotkeyModifier : std::uint32_t {
    None = 0x0000,
    Alt = 0x0001,
    Ctrl = 0x0002,
    Shift = 0x0004,
    Win = 0x0008,
};

[[nodiscard]] constexpr std::uint32_t toBits(HotkeyModifier modifier) noexcept {
    return static_cast<std::uint32_t>(modifier);
}

/// A key combination: a bitmask of HotkeyModifier and a Win32 virtual-key code.
struct Hotkey {
    std::uint32_t modifiers{0};
    std::uint32_t virtualKey{0};

    [[nodiscard]] bool operator==(const Hotkey&) const noexcept = default;
};

/// Parse a spec such as "ctrl+alt+j", "Ctrl + Shift + F5" or "win+space".
///
/// Tokens are case-insensitive and separated by '+'. Exactly one token must
/// be a key; the rest are modifiers (ctrl/control, alt, shift, win/super).
/// Throws std::invalid_argument (ValueError in Python) for anything else.
[[nodiscard]] Hotkey parseHotkey(std::string_view spec);

/// Canonical lower-case spelling of @p hotkey, e.g. "ctrl+alt+j".
/// Throws std::invalid_argument if the virtual key has no name in our table.
[[nodiscard]] std::string formatHotkey(const Hotkey& hotkey);

/// One hotkey press, identified by the id registerHotkey() returned.
struct HotkeyEvent {
    int id{0};
    Hotkey hotkey;
};

/// Registers system-wide hotkeys and reports presses through a poll queue.
///
/// RegisterHotKey is thread-affine: WM_HOTKEY is posted to the message
/// queue of the thread that registered the key, so that thread must pump
/// messages for the hotkey's whole lifetime. Python has no such thread, so
/// the manager owns one. Registration requests are marshalled onto it and
/// presses are appended to a queue that pollEvents() drains, which keeps
/// the native thread from ever needing the GIL.
class HotkeyManager {
public:
    /// Start the message-loop thread. Throws OsError if it cannot start.
    HotkeyManager();
    ~HotkeyManager();

    HotkeyManager(const HotkeyManager&) = delete;
    HotkeyManager& operator=(const HotkeyManager&) = delete;
    HotkeyManager(HotkeyManager&&) = delete;
    HotkeyManager& operator=(HotkeyManager&&) = delete;

    /// Register @p hotkey system-wide and return its id.
    /// Throws OsError if another application (or this one) already owns it.
    int registerHotkey(const Hotkey& hotkey);

    /// Release the hotkey with @p id. Unknown ids are ignored.
    void unregisterHotkey(int id);

    /// Drain and return every press reported since the previous poll.
    [[nodiscard]] std::vector<HotkeyEvent> pollEvents();

    /// Number of hotkeys currently registered.
    [[nodiscard]] std::size_t registeredCount() const;

private:
    struct Impl;  // keeps the message loop and Win32 headers out of the public header
    std::unique_ptr<Impl> impl_;
};

}  // namespace jarvis::os
