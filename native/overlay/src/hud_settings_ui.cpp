#include "jarvis/overlay/hud_data.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/palette.hpp"
#include "jarvis/overlay/ui_events.hpp"

#include <imgui.h>

#include "hud.hpp"
#include "hud_interact.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <format>
#include <string>
#include <string_view>
#include <system_error>

namespace jarvis::overlay::hud {

namespace {

constexpr float kRowSize = 14.0F;
constexpr float kSectionSize = 11.5F;
constexpr float kNoteSize = 11.5F;
constexpr float kLabelShare = 0.42F;
constexpr std::string_view kTrue = "true";
constexpr std::string_view kFalse = "false";
constexpr std::string_view kRestartNote = "applies after restart";
constexpr std::string_view kOpenLogKey = "open_log";
constexpr std::string_view kRestartKey = "restart";
constexpr std::string_view kPressed = "pressed";
// A swatch that opens a hue bar and saturation/value square; no numeric
// inputs, since a typed half-hex would be reported on every keystroke.
// NOLINTBEGIN(bugprone-signed-bitwise,hicpp-signed-bitwise) - ImGui flags are plain ints
constexpr ImGuiColorEditFlags kColourFlags =
    ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueBar;
// NOLINTEND(bugprone-signed-bitwise,hicpp-signed-bitwise)

ImVec4 toVec4(Rgba colour) noexcept {
    return ImVec4{colour.r, colour.g, colour.b, colour.a};
}

void coloured(std::string_view text, Rgba colour) {
    ImGui::PushStyleColor(ImGuiCol_Text, toVec4(colour));
    ImGui::TextUnformatted(text.data(), text.data() + text.size());
    ImGui::PopStyleColor();
}

float parseFloat(std::string_view text, float fallback) noexcept {
    float value = fallback;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    static_cast<void>(end);
    return error == std::errc{} ? value : fallback;
}

/// The label of @p item's current choice, or the raw value if it has none.
std::string_view choiceLabel(const SettingItem& item) noexcept {
    const auto found = std::ranges::find_if(
        item.choices, [&](const SettingChoice& c) { return c.value == item.value; });
    return found != item.choices.end() ? std::string_view{found->label}
                                       : std::string_view{item.value};
}

}  // namespace

void HudInteraction::drawSettings(const InteractFrame& in, const HudFonts& fonts,
                                  const Theme& theme) {
    const RectF& body = in.expanded.body;
    ImGui::SetCursorScreenPos({body.x, body.y});
    ImGui::PushFont(fonts.body, kRowSize * in.scale);
    ImGui::BeginChild("##settings", {body.width, body.height}, ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoBackground);
    std::string_view section;
    for (SettingItem& item : settings_.items) {
        if (item.section != section) {
            section = item.section;
            ImGui::Spacing();
            ImGui::PushFont(fonts.display, kSectionSize * in.scale);
            coloured(section, in.accent);
            ImGui::PopFont();
            ImGui::Separator();
        }
        drawSettingRow(item, in, fonts, theme);
    }
    drawSettingsFooter(in, theme);
    ImGui::EndChild();
    ImGui::PopFont();
}

void HudInteraction::drawSettingRow(SettingItem& item, const InteractFrame& in,
                                    const HudFonts& fonts, const Theme& theme) {
    ImGui::PushID(item.key.c_str());
    const float width = ImGui::GetContentRegionAvail().x;
    const float labelWidth = width * kLabelShare;
    const float top = ImGui::GetCursorPosY();
    coloured(item.label, theme.text);
    if (item.restartRequired || !item.note.empty()) {
        ImGui::PushFont(fonts.body, kNoteSize * in.scale);
        ImGui::PushTextWrapPos(labelWidth);
        coloured(item.note.empty() ? kRestartNote : std::string_view{item.note}, theme.mutedText);
        ImGui::PopTextWrapPos();
        ImGui::PopFont();
    }
    const float bottom = ImGui::GetCursorPosY();
    ImGui::SetCursorPos({labelWidth, top});
    drawSettingWidget(item, width - labelWidth);
    ImGui::SetCursorPosY(std::max(bottom, ImGui::GetCursorPosY()));
    ImGui::Spacing();
    ImGui::PopID();
}

void HudInteraction::drawSettingWidget(SettingItem& item, float width) {
    switch (item.kind) {
        case SettingKind::Choice:
            drawChoice(item, width);
            return;
        case SettingKind::Toggle:
            drawToggle(item);
            return;
        case SettingKind::Slider:
            drawSlider(item, width);
            return;
        case SettingKind::Colour:
            drawColour(item);
            return;
        case SettingKind::Action:
            break;
    }
    drawAction(item);
}

void HudInteraction::drawChoice(SettingItem& item, float width) {
    ImGui::SetNextItemWidth(width);
    const std::string preview{choiceLabel(item)};
    if (!ImGui::BeginCombo("##choice", preview.c_str())) {
        return;
    }
    editing_ = true;
    for (const SettingChoice& choice : item.choices) {
        const bool selected = choice.value == item.value;
        if (ImGui::Selectable(choice.label.c_str(), selected) && !selected) {
            item.value = choice.value;
            push(SettingChanged{.key = item.key, .value = item.value});
        }
        if (selected) {
            ImGui::SetItemDefaultFocus();
        }
    }
    ImGui::EndCombo();
}

void HudInteraction::drawToggle(SettingItem& item) {
    bool on = item.value == kTrue;
    if (ImGui::Checkbox("##toggle", &on)) {
        item.value = on ? kTrue : kFalse;
        push(SettingChanged{.key = item.key, .value = item.value});
    }
}

void HudInteraction::drawSlider(SettingItem& item, float width) {
    float value = parseFloat(item.value, item.minimum);
    ImGui::SetNextItemWidth(width);
    if (ImGui::SliderFloat("##slider", &value, item.minimum, item.maximum, "%.2f")) {
        if (item.step > 0.0F) {
            value = item.minimum + (std::round((value - item.minimum) / item.step) * item.step);
        }
        item.value = std::format("{:.2f}", value);
    }
    editing_ = editing_ || ImGui::IsItemActive();
    // Reported once on release, not per frame of the drag.
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        push(SettingChanged{.key = item.key, .value = item.value});
    }
}

void HudInteraction::drawColour(SettingItem& item) {
    const Rgba current = parseHexColour(item.value).value_or(kDefaultAccent);
    std::array<float, 3> rgb{current.r, current.g, current.b};
    if (ImGui::ColorEdit3("##colour", rgb.data(), kColourFlags)) {
        item.value = toHex(Rgba{.r = rgb.at(0), .g = rgb.at(1), .b = rgb.at(2), .a = 1.0F});
        colourDirty_ = true;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(item.value.c_str());
    editing_ = editing_ || ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
    // Reported once the drag ends, like a slider, so a sweep across the hue
    // bar is one persisted value rather than one per frame.
    if (colourDirty_ && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        colourDirty_ = false;
        push(SettingChanged{.key = item.key, .value = item.value});
    }
}

void HudInteraction::drawAction(const SettingItem& item) {
    if (!ImGui::Button(item.label.c_str())) {
        return;
    }
    if (item.key == kOpenLogKey) {
        push(OpenLog{});
    } else if (item.key == kRestartKey) {
        push(RestartRequested{});
    } else {
        push(SettingChanged{.key = item.key, .value = std::string{kPressed}});
    }
}

void HudInteraction::drawSettingsFooter(const InteractFrame& in, const Theme& theme) {
    ImGui::Spacing();
    if (!settings_.notice.empty()) {
        ImGui::PushTextWrapPos(0.0F);
        coloured(settings_.notice, theme.mutedText);
        ImGui::PopTextWrapPos();
    }
    if (settings_.restartPending) {
        coloured("Some changes apply after a restart.", in.accent);
        if (ImGui::Button("Restart Jarvis now")) {
            push(RestartRequested{});
        }
    }
}

}  // namespace jarvis::overlay::hud

// Design notes:
// - The page edits a local copy of what Python pushed, so the widget shows
//   the new value at once; Python answers each SettingChanged with a fresh
//   page (validated value, notice, restart flag) that replaces the copy.
// - Actions are buttons whose key names the event: "open_log" and
//   "restart" map to their typed events, anything else is reported as
//   SettingChanged{key, "pressed"} so Python can add actions without a
//   native change.
// - Sliders report on release (IsItemDeactivatedAfterEdit) and quantise to
//   the item's step, so a drag produces one persisted value, not sixty.
// - The colour picker lives in a popup, where IsItemDeactivatedAfterEdit
//   does not see the picker's own widgets, so it reports when the mouse is
//   released after an edit instead. The HUD recolours when Python answers
//   (one bridge poll later); a native preview during the drag would need
//   the window to undo it whenever Python rejects or never answers, which
//   is more moving parts than a 30 ms delay is worth.
