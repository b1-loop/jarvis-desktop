/// @file ui_events.hpp
/// @brief What the user did in the HUD, queued for Python to poll.
///
/// The overlay never calls into Python from its UI thread (ADR-0011,
/// ADR-0024). When the user clicks a control, types a request or changes a
/// setting, the UI thread pushes a typed UiEvent into a UiEventQueue and
/// carries on drawing; Python drains the queue every ~30 ms on its event
/// loop and decides what the event means. Nothing here knows Win32, ImGui
/// or pybind11, so ordering, coalescing and overflow are unit-tested.

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace jarvis::overlay {

/// The user pressed Enter in the text field.
struct SubmitText {
    /// What was typed, UTF-8, trimmed of surrounding white space.
    std::string text;
    [[nodiscard]] bool operator==(const SubmitText&) const = default;
};

/// The close control: hide the HUD (the tray brings it back).
struct CloseRequested {
    [[nodiscard]] bool operator==(const CloseRequested&) const = default;
};

/// The quit control, after the user confirmed it in the HUD.
struct QuitRequested {
    [[nodiscard]] bool operator==(const QuitRequested&) const = default;
};

/// The microphone control.
struct ToggleMute {
    [[nodiscard]] bool operator==(const ToggleMute&) const = default;
};

/// The expand/collapse control (or Escape in the expanded view).
struct SetExpanded {
    bool expanded{false};
    [[nodiscard]] bool operator==(const SetExpanded&) const = default;
};

/// A setting was edited in the settings page. Values travel as text; the
/// Python side owns the types, validation and persistence.
struct SettingChanged {
    std::string key;
    std::string value;
    [[nodiscard]] bool operator==(const SettingChanged&) const = default;
};

/// "Open log file" in the settings page.
struct OpenLog {
    [[nodiscard]] bool operator==(const OpenLog&) const = default;
};

/// "Restart" in the settings page (settings that apply after a restart).
struct RestartRequested {
    [[nodiscard]] bool operator==(const RestartRequested&) const = default;
};

/// The settings page was opened; Python refreshes lists that may have
/// changed (installed Ollama models) and pushes them back.
struct SettingsOpened {
    [[nodiscard]] bool operator==(const SettingsOpened&) const = default;
};

/// The user moved (or reset) the corner HUD or the expanded panel.
///
/// Positions are logical (96-DPI) pixels from the top-left of the named
/// monitor's work area, so a saved position means the same place after a
/// DPI change and survives the monitor order changing.
struct PlacementChanged {
    /// Which of the remembered placements this is: the expanded panel, the
    /// orb (when @c orb is set), otherwise the corner HUD.
    bool expanded{false};
    /// False after a double-click on the ring (back to the default corner)
    /// or, for the orb, after the HUD moved (it docks next to it again).
    bool custom{true};
    /// Monitor device name, e.g. "\\\\.\\DISPLAY2".
    std::string monitor;
    float x{0.0F};
    float y{0.0F};
    /// Logical size; only meaningful for the expanded panel.
    float width{0.0F};
    float height{0.0F};
    /// The minimized orb's position (ADR-0024, update).
    bool orb{false};
    [[nodiscard]] bool operator==(const PlacementChanged&) const = default;
};

/// The HUD shrank to the orb: the minimize control, or auto-minimize.
struct Minimized {
    /// True when the auto-minimize policy did it (not to be remembered).
    bool automatic{false};
    [[nodiscard]] bool operator==(const Minimized&) const = default;
};

/// The orb grew back into the HUD: a click on it, or auto-minimize
/// popping it back for a turn.
struct Restored {
    /// True when the auto-minimize policy did it (not to be remembered).
    bool automatic{false};
    [[nodiscard]] bool operator==(const Restored&) const = default;
};

/// One thing the user did.
using UiEvent =
    std::variant<SubmitText, CloseRequested, QuitRequested, ToggleMute, SetExpanded, SettingChanged,
                 OpenLog, RestartRequested, SettingsOpened, PlacementChanged, Minimized, Restored>;

/// Snake-case name of the event's alternative, e.g. "submit_text"; the
/// spelling the Python binding uses.
[[nodiscard]] std::string_view kindOf(const UiEvent& event);

/// Bounded, thread-safe FIFO between the UI thread and Python.
///
/// Policy, in this order:
/// 1. Coalescing: a SettingChanged for the same key, or a PlacementChanged
///    for the same panel, that directly follows the newest queued event
///    replaces it instead of being appended. A slider drag or a window drag
///    yields one event per poll, not one per frame, and ordering relative
///    to every other event is preserved exactly.
/// 2. Overflow: when the queue holds @c capacity events, the oldest is
///    dropped and counted in dropped(). Python polls every ~30 ms, so this
///    only happens when the event loop is stuck; the freshest intent wins.
class UiEventQueue {
public:
    static constexpr std::size_t kDefaultCapacity = 256;

    explicit UiEventQueue(std::size_t capacity = kDefaultCapacity);

    /// Enqueue @p event (or coalesce it, see the class notes). Never blocks
    /// for longer than a short critical section.
    void push(UiEvent event);

    /// Everything queued so far, oldest first, leaving the queue empty.
    [[nodiscard]] std::vector<UiEvent> drain();

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    /// Events dropped by the overflow policy since construction.
    [[nodiscard]] std::uint64_t dropped() const;

private:
    std::size_t capacity_;
    mutable std::mutex mutex_;  // guards events_ and dropped_
    std::deque<UiEvent> events_;
    std::uint64_t dropped_{0};
};

/// Whether @p next may replace @p newest under the coalescing rule.
[[nodiscard]] bool coalesces(const UiEvent& newest, const UiEvent& next) noexcept;

}  // namespace jarvis::overlay
