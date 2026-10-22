#include "jarvis/overlay/hud_data.hpp"
#include "jarvis/overlay/hud_layout.hpp"
#include "jarvis/overlay/layout.hpp"

#include <imgui.h>

#include "hud.hpp"
#include "hud_draw.hpp"
#include "hud_interact.hpp"

#include <string_view>

namespace jarvis::overlay::hud {

namespace {

// Expanded panel typography, 96-DPI pixels.
constexpr float kHistorySize = 15.0F;
constexpr float kMetaSize = 11.5F;
constexpr float kTabSize = 12.0F;
constexpr float kTabTracking = 2.0F;
constexpr float kConfirmWidth = 90.0F;
constexpr float kConfirmGap = 8.0F;

constexpr std::string_view kEmptyHistory =
    "Nothing said yet this session. Type below, or say the wake word.";

ImVec4 toVec4(Rgba colour) noexcept {
    return ImVec4{colour.r, colour.g, colour.b, colour.a};
}

std::string_view roleLabel(HistoryRole role) noexcept {
    switch (role) {
        case HistoryRole::User:
            return "YOU";
        case HistoryRole::Jarvis:
            return "JARVIS";
        case HistoryRole::Tool:
            return "TOOL";
        case HistoryRole::Notice:
            break;
    }
    return "NOTE";
}

void coloured(std::string_view text, Rgba colour) {
    ImGui::PushStyleColor(ImGuiCol_Text, toVec4(colour));
    ImGui::TextUnformatted(text.data(), text.data() + text.size());
    ImGui::PopStyleColor();
}

}  // namespace

void HudInteraction::drawExpanded(const InteractFrame& in, const HudFonts& fonts,
                                  const Theme& theme) {
    const ExpandedLayout& layout = in.expanded;
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, in.opacity);
    drawControls(in, layout.controls, in.opacity);
    drawTabs(in, fonts);
    if (page_ == Page::Conversation) {
        drawHistory(in, fonts, theme);
    } else {
        drawSettings(in, fonts, theme);
    }
    const float body = kHistorySize * in.scale;
    if (confirming_) {
        const RectF& row = layout.input;
        const float width = kConfirmWidth * in.scale;
        const float gap = kConfirmGap * in.scale;
        const RectF no{.x = row.right() - width, .y = row.y, .width = width, .height = row.height};
        const RectF yes{.x = no.x - gap - width, .y = row.y, .width = width, .height = row.height};
        const RectF text{
            .x = row.x, .y = row.y + (row.height * 0.25F), .width = 0.0F, .height = row.height};
        drawConfirm(in, text, yes, no, fonts, body);
    } else {
        drawField(layout.input, fonts, body);
    }
    ImGui::PopStyleVar();
}

void HudInteraction::drawTabs(const InteractFrame& in, const HudFonts& fonts) {
    drawTab(in, in.expanded.tabConversation, "CONVERSATION", Page::Conversation, fonts);
    drawTab(in, in.expanded.tabSettings, "SETTINGS", Page::Settings, fonts);
}

void HudInteraction::drawTab(const InteractFrame& in, RectF rect, std::string_view label,
                             Page target, const HudFonts& fonts) {
    ImGui::SetCursorScreenPos({rect.x, rect.y});
    ImGui::PushID(static_cast<int>(target));
    const bool clicked = ImGui::InvisibleButton("##tab", {rect.width, rect.height});
    const bool hot = ImGui::IsItemHovered() || ImGui::IsItemFocused();
    ImGui::PopID();
    const bool selected = page_ == target;
    ImDrawList& draw = *ImGui::GetWindowDrawList();
    const float size = kTabSize * in.scale;
    float alpha = hot ? 0.8F : 0.5F;
    if (selected) {
        alpha = 1.0F;
    }
    alpha *= in.opacity;
    spacedText(draw, *fonts.display, size, {rect.x, rect.y + ((rect.height - size) * 0.5F)}, label,
               kTabTracking * in.scale, withAlpha(in.accent, alpha));
    if (selected) {
        glowLine(draw, {rect.x, rect.bottom() - in.scale}, {rect.right(), rect.bottom() - in.scale},
                 Stroke{.colour = withAlpha(in.accent, 0.8F * in.opacity),
                        .thickness = 1.4F * in.scale,
                        .glow = kSoftGlow});
    }
    if (clicked && !selected) {
        page_ = target;
        if (target == Page::Settings) {
            push(SettingsOpened{});
        }
    }
}

void HudInteraction::drawHistory(const InteractFrame& in, const HudFonts& fonts,
                                 const Theme& theme) {
    const RectF& body = in.expanded.body;
    ImGui::SetCursorScreenPos({body.x, body.y});
    ImGui::PushFont(fonts.body, kHistorySize * in.scale);
    ImGui::BeginChild("##history", {body.width, body.height}, ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoBackground);
    const bool atEnd = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0F;
    const auto& history = *in.data.history;
    if (history.empty()) {
        coloured(kEmptyHistory, theme.mutedText);
    }
    for (const HistoryEntry& entry : history) {
        drawHistoryEntry(entry, in, fonts, theme);
    }
    if (in.data.historyRevision != seenHistory_) {
        seenHistory_ = in.data.historyRevision;
        // Follow new lines only when the user has not scrolled up to read.
        scrollToEnd_ = scrollToEnd_ || atEnd;
    }
    if (scrollToEnd_) {
        ImGui::SetScrollHereY(1.0F);
        scrollToEnd_ = false;
    }
    ImGui::EndChild();
    ImGui::PopFont();
}

void HudInteraction::drawHistoryEntry(const HistoryEntry& entry, const InteractFrame& in,
                                      const HudFonts& fonts, const Theme& theme) {
    const bool tool = entry.role == HistoryRole::Tool;
    const Rgba roleColour = entry.role == HistoryRole::Jarvis ? in.accent : theme.mutedText;
    ImGui::PushFont(fonts.display, kMetaSize * in.scale);
    coloured(entry.time, withAlpha(theme.mutedText, 0.7F));
    ImGui::SameLine();
    coloured(roleLabel(entry.role), roleColour);
    ImGui::PopFont();
    if (tool) {
        // Tool calls are context, not conversation: one dim line each.
        ImGui::SameLine();
        ImGui::PushFont(fonts.body, kMetaSize * in.scale);
        coloured(entry.text, withAlpha(theme.mutedText, 0.85F));
        ImGui::PopFont();
    } else {
        ImGui::PushTextWrapPos(0.0F);
        coloured(entry.text, theme.text);
        ImGui::PopTextWrapPos();
    }
    ImGui::Spacing();
}

}  // namespace jarvis::overlay::hud

// Design notes:
// - The history is drawn in full every frame inside a scrolling child
//   window: a session is a few hundred lines at most (HudDataLimits caps
//   it), which ImGui lays out in well under a millisecond, and the
//   expanded view only exists while the user is looking at it.
// - New lines scroll the view to the end only when it already was at the
//   end, so reading back through a long answer is not interrupted.
