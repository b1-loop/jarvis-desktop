/// @file hud_interact.hpp
/// @brief The interactive layer of the HUD: hover controls, text field,
/// quit confirmation, expanded conversation view and settings page (private).
///
/// Only window.cpp drives this. It runs on the UI thread after the scene
/// has been drawn and builds Dear ImGui widgets at the rectangles of
/// hud_layout.hpp. It never decides anything that belongs to Python: every
/// action becomes a UiEvent in the queue (submit, close, quit, mute,
/// expand, setting changed, open log, restart), and only purely visual
/// state (which page is open, whether the field is shown, scroll position)
/// lives here. Members are split over hud_interact.cpp (frame, controls,
/// field, confirmation), hud_expanded_ui.cpp (tabs and history) and
/// hud_settings_ui.cpp (settings page).

#pragma once

#include "jarvis/overlay/hud_data.hpp"
#include "jarvis/overlay/hud_layout.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/ui_events.hpp"

#include <imgui.h>

#include "hud.hpp"
#include "text_entry.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace jarvis::overlay::hud {

/// What the expanded panel shows below its header.
enum class Page : std::uint8_t { Conversation, Settings };

/// Everything one frame of the interactive layer needs (client coordinates).
struct InteractFrame {
    const HudData& data;
    const CollapsedLayout& collapsed;
    const ExpandedLayout& expanded;
    /// Expand/collapse morph progress, 0 (corner HUD) to 1 (panel).
    float expand{0.0F};
    /// Fade of the hover controls in the corner HUD, 0 to 1.
    float hover{0.0F};
    float scale{1.0F};
    float textScale{1.0F};
    /// Current accent (state colour) and global opacity.
    Rgba accent;
    float opacity{1.0F};
};

/// What the window should do after the frame.
struct InteractResult {
    /// Take keyboard focus (the user wants to type).
    bool takeFocus{false};
    /// Give keyboard focus back to the previous window.
    bool releaseFocus{false};
};

/// UI-thread-only state and widgets of the interactive HUD.
class HudInteraction {
public:
    explicit HudInteraction(UiEventQueue& events);

    /// Build this frame's widgets in a full-window ImGui window.
    InteractResult frame(const InteractFrame& in, const HudFonts& fonts, const Theme& theme);

    /// Expand or collapse; @p report pushes a SetExpanded event.
    void setExpanded(bool expanded, bool report);
    /// Show the text field; @p focus also takes keyboard focus for it.
    void openInput(bool focus);
    /// Hide the text field (corner HUD) and drop focus.
    void closeInput();
    /// Demo hook: show @p text in the field without taking focus.
    void showDemoInput(std::string_view text);

    [[nodiscard]] bool expanded() const noexcept { return expanded_; }
    [[nodiscard]] bool inputOpen() const noexcept { return inputOpen_; }
    [[nodiscard]] bool confirming() const noexcept { return confirming_; }
    [[nodiscard]] Page page() const noexcept { return page_; }
    /// True while a widget is being edited (the field, a slider, a combo).
    [[nodiscard]] bool editing() const noexcept { return editing_; }

private:
    // hud_interact.cpp
    void syncFromData(const HudData& data);
    void drawControls(const InteractFrame& in, const ControlBar& bar, float alpha);
    void drawControl(const InteractFrame& in, HudControl control, RectF rect, float alpha);
    void act(HudControl control);
    void drawCollapsed(const InteractFrame& in, const HudFonts& fonts);
    void drawField(RectF rect, const HudFonts& fonts, float fontSize);
    void drawConfirm(const InteractFrame& in, RectF text, RectF yes, RectF no,
                     const HudFonts& fonts, float fontSize);
    void push(UiEvent event);

    // hud_expanded_ui.cpp
    void drawExpanded(const InteractFrame& in, const HudFonts& fonts, const Theme& theme);
    void drawTabs(const InteractFrame& in, const HudFonts& fonts);
    void drawTab(const InteractFrame& in, RectF rect, std::string_view label, Page target,
                 const HudFonts& fonts);
    void drawHistory(const InteractFrame& in, const HudFonts& fonts, const Theme& theme);
    static void drawHistoryEntry(const HistoryEntry& entry, const InteractFrame& in,
                                 const HudFonts& fonts, const Theme& theme);

    // hud_settings_ui.cpp
    void drawSettings(const InteractFrame& in, const HudFonts& fonts, const Theme& theme);
    void drawSettingRow(SettingItem& item, const InteractFrame& in, const HudFonts& fonts,
                        const Theme& theme);
    void drawSettingWidget(SettingItem& item, float width);
    void drawChoice(SettingItem& item, float width);
    void drawToggle(SettingItem& item);
    void drawSlider(SettingItem& item, float width);
    void drawAction(const SettingItem& item);
    void drawSettingsFooter(const InteractFrame& in, const Theme& theme);

    UiEventQueue& events_;
    TextEntry entry_;
    /// Placeholder of the text field (ImGui wants a terminated string).
    std::string fieldHint_{"Type a request and press Enter"};
    InteractResult pending_;
    bool expanded_{false};
    bool inputOpen_{false};
    bool focusField_{false};
    bool confirming_{false};
    bool editing_{false};
    Page page_{Page::Conversation};
    std::uint64_t seenExpandRequest_{0};
    std::uint64_t seenHistory_{0};
    bool scrollToEnd_{true};
    std::uint64_t seenSettings_{0};
    /// Local, editable copy of the page Python pushed last.
    SettingsPage settings_;
};

/// Dear ImGui colours and metrics in the HUD's style for @p theme.
void applyHudStyle(const Theme& theme, float scale);

/// Draw the vector glyph of @p control into @p rect.
void drawControlIcon(ImDrawList& draw, HudControl control, RectF rect, Rgba colour, float thickness,
                     bool muted, bool expanded);

}  // namespace jarvis::overlay::hud
