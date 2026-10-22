/// @file hud_data.hpp
/// @brief What Python pushes to the interactive HUD as data: the session's
/// conversation, the settings page, the microphone state and appearance.
///
/// Python owns every decision (which settings exist, what their choices
/// are, whether a change applies live); the native side only renders this
/// data, lets the user edit it and reports edits as UiEvents. Like
/// OverlayModel, every setter is a short critical section and the UI thread
/// takes one snapshot per frame; the history and settings travel as
/// shared immutable blocks so a snapshot never copies them.

#pragma once

#include "jarvis/overlay/effects.hpp"
#include "jarvis/overlay/layout.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace jarvis::overlay {

/// Who a history line belongs to.
enum class HistoryRole : std::uint8_t { User, Jarvis, Tool, Notice };

/// One line of the conversation shown in the expanded view.
struct HistoryEntry {
    HistoryRole role{HistoryRole::User};
    /// UTF-8; bounded by HudDataLimits::maxEntryChars.
    std::string text;
    /// Wall-clock time as Python formatted it, e.g. "14:32".
    std::string time;

    [[nodiscard]] bool operator==(const HistoryEntry&) const = default;
};

/// How a setting is edited.
enum class SettingKind : std::uint8_t {
    /// One of a list of choices (a combo box).
    Choice,
    /// On or off; the value is "true" or "false".
    Toggle,
    /// A number between minimum and maximum.
    Slider,
    /// A button (open the log, restart); has no value.
    Action,
};

/// One choice of a Choice setting.
struct SettingChoice {
    /// What is reported back in SettingChanged::value.
    std::string value;
    /// What the user sees.
    std::string label;

    [[nodiscard]] bool operator==(const SettingChoice&) const = default;
};

/// One row of the settings page.
struct SettingItem {
    std::string key;
    std::string label;
    /// Heading the row is grouped under, e.g. "VOICE".
    std::string section;
    SettingKind kind{SettingKind::Choice};
    std::vector<SettingChoice> choices;
    /// Current value as text ("true", "0.80", "sv-SE-MattiasNeural").
    std::string value;
    float minimum{0.0F};
    float maximum{1.0F};
    float step{0.0F};
    /// Shows "applies after restart" next to the row.
    bool restartRequired{false};
    /// Short extra line under the label (why a choice is unavailable).
    std::string note;

    [[nodiscard]] bool operator==(const SettingItem&) const = default;
};

/// The whole settings page.
struct SettingsPage {
    std::vector<SettingItem> items;
    /// A restart-required setting changed since start: show the Restart button.
    bool restartPending{false};
    /// Status line, e.g. "Saved." or why a value was rejected.
    std::string notice;

    [[nodiscard]] bool operator==(const SettingsPage&) const = default;
};

/// Palette, richness and size, changeable while the HUD runs.
struct Appearance {
    ThemeName theme{ThemeName::Cyan};
    anim::Intensity intensity{anim::Intensity::Normal};
    PanelSize size{PanelSize::Large};

    [[nodiscard]] bool operator==(const Appearance&) const noexcept = default;
};

/// Bounds on what Python can make the HUD keep.
struct HudDataLimits {
    /// History lines kept for the expanded view; the oldest go first.
    std::size_t maxEntries = 500;
    /// Characters (code points) kept per line; the head wins.
    std::size_t maxEntryChars = 2000;
};

/// Immutable copy of the HUD data for one frame.
struct HudData {
    bool muted{false};
    std::shared_ptr<const std::vector<HistoryEntry>> history;
    /// Incremented by every history change; the view scrolls to the end on change.
    std::uint64_t historyRevision{0};
    std::shared_ptr<const SettingsPage> settings;
    std::uint64_t settingsRevision{0};
    Appearance appearance;
    /// The latest expand/collapse request from Python and its sequence number.
    std::optional<bool> expandRequest;
    std::uint64_t expandRequestId{0};
    /// Incremented by every mutation.
    std::uint64_t revision{0};
};

/// The HUD's data, owned by the window and mutated from Python.
class HudDataModel {
public:
    explicit HudDataModel(HudDataLimits limits = {}, Appearance appearance = {});

    void setMuted(bool muted);
    /// Append @p entry (its text cut to the limit), dropping the oldest
    /// line when the history is full.
    void addHistory(HistoryEntry entry);
    void clearHistory();
    void setSettings(SettingsPage page);
    void setAppearance(Appearance appearance);
    /// Ask the UI thread to expand (true) or collapse (false) the HUD.
    void requestExpanded(bool expanded);

    [[nodiscard]] HudData snapshot() const;
    [[nodiscard]] const HudDataLimits& limits() const noexcept { return limits_; }

private:
    HudDataLimits limits_;
    mutable std::mutex mutex_;  // guards data_
    HudData data_;
};

}  // namespace jarvis::overlay
