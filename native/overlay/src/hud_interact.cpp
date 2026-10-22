#include "hud_interact.hpp"

#include "jarvis/overlay/hud_data.hpp"
#include "jarvis/overlay/hud_layout.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/ui_events.hpp"

#include <imgui.h>

#include "hud.hpp"
#include "hud_draw.hpp"
#include "text_entry.hpp"

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>

namespace jarvis::overlay::hud {

namespace {

// Typography of the corner HUD, 96-DPI pixels before the text scale; must
// match hud_panel.cpp.
constexpr float kBodySize = 14.5F;
constexpr float kCaptionSize = 15.0F;
constexpr float kIconStroke = 1.4F;
constexpr float kIconInset = 0.26F;
constexpr float kControlPlate = 0.62F;
constexpr float kIdleRing = 0.45F;
constexpr float kHoverRing = 0.95F;

// NOLINTBEGIN(bugprone-signed-bitwise,hicpp-signed-bitwise) - ImGui flags are plain ints
constexpr ImGuiWindowFlags kHudWindowFlags =
    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove |
    ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus |
    ImGuiWindowFlags_NoScrollWithMouse;
// NOLINTEND(bugprone-signed-bitwise,hicpp-signed-bitwise)

constexpr std::string_view kConfirmQuestion = "QUIT JARVIS?";

ImVec4 toVec4(Rgba colour) noexcept {
    return ImVec4{colour.r, colour.g, colour.b, colour.a};
}

}  // namespace

HudInteraction::HudInteraction(UiEventQueue& events) : events_(events) {}

InteractResult HudInteraction::frame(const InteractFrame& in, const HudFonts& fonts,
                                     const Theme& theme) {
    pending_ = {};
    editing_ = false;
    syncFromData(in.data);
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos({0.0F, 0.0F});
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0.0F, 0.0F});
    ImGui::Begin("##hud", nullptr, kHudWindowFlags);
    // Mid-morph nothing is interactive: both views are only pictures.
    if (in.expand <= 0.0F) {
        drawCollapsed(in, fonts);
    } else if (in.expand >= 1.0F) {
        drawExpanded(in, fonts, theme);
    }
    const bool escape = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
                        !editing_ && ImGui::IsKeyPressed(ImGuiKey_Escape, false);
    ImGui::End();
    ImGui::PopStyleVar();
    if (escape) {
        // Escape backs out one level: confirmation, then the panel.
        if (confirming_) {
            confirming_ = false;
        } else if (expanded_) {
            setExpanded(false, true);
        }
        pending_.releaseFocus = true;
    }
    return pending_;
}

void HudInteraction::setExpanded(bool expanded, bool report) {
    if (expanded == expanded_) {
        return;
    }
    expanded_ = expanded;
    confirming_ = false;
    if (!expanded) {
        page_ = Page::Conversation;
        inputOpen_ = false;
    }
    scrollToEnd_ = true;
    if (report) {
        push(SetExpanded{.expanded = expanded});
    }
}

void HudInteraction::openInput(bool focus) {
    inputOpen_ = true;
    if (focus) {
        focusField_ = true;
        pending_.takeFocus = true;
    }
}

void HudInteraction::closeInput() {
    inputOpen_ = false;
    focusField_ = false;
    entry_.clear();
    pending_.releaseFocus = true;
}

void HudInteraction::showDemoInput(std::string_view text) {
    inputOpen_ = true;
    entry_.set(text);
}

void HudInteraction::syncFromData(const HudData& data) {
    if (data.expandRequestId != seenExpandRequest_) {
        seenExpandRequest_ = data.expandRequestId;
        if (data.expandRequest.has_value()) {
            setExpanded(*data.expandRequest, false);
        }
    }
    if (data.settingsRevision != seenSettings_ && data.settings != nullptr) {
        seenSettings_ = data.settingsRevision;
        settings_ = *data.settings;
    }
}

void HudInteraction::push(UiEvent event) {
    events_.push(std::move(event));
}

// --- controls ------------------------------------------------------------------

void HudInteraction::drawControls(const InteractFrame& in, const ControlBar& bar, float alpha) {
    for (const HudControl control : kHudControls) {
        drawControl(in, control, bar.button(control), alpha);
    }
}

void HudInteraction::drawControl(const InteractFrame& in, HudControl control, RectF rect,
                                 float alpha) {
    ImGui::SetCursorScreenPos({rect.x, rect.y});
    ImGui::PushID(static_cast<int>(control));
    const bool clicked = ImGui::InvisibleButton("##control", {rect.width, rect.height});
    const bool hot = ImGui::IsItemHovered() || ImGui::IsItemFocused();
    const bool held = ImGui::IsItemActive();
    if (hot) {
        const std::string label{controlLabel(control)};
        ImGui::SetItemTooltip("%s", label.c_str());
    }
    ImGui::PopID();
    ImDrawList& draw = *ImGui::GetWindowDrawList();
    const ImVec2 centre{rect.x + (rect.width * 0.5F), rect.y + (rect.height * 0.5F)};
    const float radius = rect.width * 0.5F;
    const bool muted = control == HudControl::Mute && in.data.muted;
    const bool danger = control == HudControl::Quit || muted;
    const Rgba light = danger ? Rgba{.r = 1.0F, .g = 0.45F, .b = 0.35F, .a = 1.0F} : in.accent;
    draw.AddCircleFilled(
        centre, radius, toU32(Rgba{.r = 0.0F, .g = 0.03F, .b = 0.06F, .a = kControlPlate * alpha}));
    const float ring = (hot || held ? kHoverRing : kIdleRing) * alpha;
    draw.AddCircle(centre, radius - 0.5F, toU32(withAlpha(light, ring)), 0, in.scale);
    if (held) {
        draw.AddCircleFilled(centre, radius, toU32(withAlpha(light, 0.25F * alpha)));
    }
    const float inset = rect.width * kIconInset;
    const RectF glyph{.x = rect.x + inset,
                      .y = rect.y + inset,
                      .width = rect.width - (2.0F * inset),
                      .height = rect.height - (2.0F * inset)};
    drawControlIcon(draw, control, glyph, withAlpha(light, (hot ? 1.0F : 0.8F) * alpha),
                    kIconStroke * in.scale, muted, expanded_);
    if (clicked) {
        act(control);
    }
}

void HudInteraction::act(HudControl control) {
    switch (control) {
        case HudControl::Keyboard:
            openInput(true);
            return;
        case HudControl::Mute:
            push(ToggleMute{});
            return;
        case HudControl::Settings:
            if (!expanded_ || page_ != Page::Settings) {
                setExpanded(true, true);
                page_ = Page::Settings;
                push(SettingsOpened{});
            } else {
                page_ = Page::Conversation;
            }
            return;
        case HudControl::Expand:
            setExpanded(!expanded_, true);
            return;
        case HudControl::Quit:
            confirming_ = true;
            return;
        case HudControl::Minimize:
            confirming_ = false;
            closeInput();
            pending_.minimize = true;
            return;
        case HudControl::Close:
            break;
    }
    confirming_ = false;
    closeInput();
    push(CloseRequested{});
}

// --- corner HUD: field and confirmation --------------------------------------------

void HudInteraction::drawCollapsed(const InteractFrame& in, const HudFonts& fonts) {
    const float alpha = in.hover * in.opacity;
    if (alpha > 0.0F) {
        drawControls(in, in.collapsed.controls, alpha);
    }
    const float body = kBodySize * in.scale * in.textScale;
    if (confirming_) {
        drawConfirm(in, in.collapsed.confirmText, in.collapsed.confirmYes, in.collapsed.confirmNo,
                    fonts, body);
    }
    if (inputOpen_) {
        drawField(in.collapsed.input, fonts, body);
    }
}

void HudInteraction::drawField(RectF rect, const HudFonts& fonts, float fontSize) {
    ImGui::SetCursorScreenPos({rect.x, rect.y});
    ImGui::PushFont(fonts.body, fontSize);
    const float padY = std::max((rect.height - fontSize) * 0.5F, 0.0F);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {fontSize * 0.6F, padY});
    const TextEntry::Result result =
        entry_.draw("##request", fieldHint_.c_str(), rect.width, focusField_);
    focusField_ = false;
    ImGui::PopStyleVar();
    ImGui::PopFont();
    editing_ = editing_ || result.active;
    if (result.submitted.has_value()) {
        push(SubmitText{.text = *result.submitted});
        pending_.releaseFocus = true;
        inputOpen_ = expanded_;
    }
    if (result.cancelled) {
        entry_.clear();
        pending_.releaseFocus = true;
        inputOpen_ = expanded_;
    }
}

void HudInteraction::drawConfirm(const InteractFrame& in, RectF text, RectF yes, RectF no,
                                 const HudFonts& fonts, float fontSize) {
    ImDrawList& draw = *ImGui::GetWindowDrawList();
    const float caption = kCaptionSize * in.scale * in.textScale;
    spacedText(draw, *fonts.display, caption, {text.x, text.y}, kConfirmQuestion, 2.0F * in.scale,
               withAlpha(Rgba{.r = 1.0F, .g = 0.55F, .b = 0.45F}, in.opacity));
    ImGui::PushFont(fonts.display, fontSize);
    ImGui::SetCursorScreenPos({yes.x, yes.y});
    if (ImGui::Button("YES##quit", {yes.width, yes.height})) {
        confirming_ = false;
        push(QuitRequested{});
    }
    ImGui::SetCursorScreenPos({no.x, no.y});
    if (ImGui::Button("NO##quit", {no.width, no.height})) {
        confirming_ = false;
    }
    ImGui::PopFont();
}

// --- style ---------------------------------------------------------------------------

void applyHudStyle(const Theme& theme, float scale) {
    ImGuiStyle style;
    const Rgba plate{.r = theme.plate.r, .g = theme.plate.g, .b = theme.plate.b, .a = 0.78F};
    auto& colours = style.Colors;
    colours[ImGuiCol_Text] = toVec4(theme.text);
    colours[ImGuiCol_TextDisabled] = toVec4(theme.mutedText);
    colours[ImGuiCol_WindowBg] = toVec4(withAlpha(plate, 0.0F));
    colours[ImGuiCol_ChildBg] = toVec4(withAlpha(plate, 0.0F));
    colours[ImGuiCol_PopupBg] = toVec4(withAlpha(plate, 1.2F));
    colours[ImGuiCol_Border] = toVec4(withAlpha(theme.accent, 0.45F));
    colours[ImGuiCol_FrameBg] = toVec4(plate);
    colours[ImGuiCol_FrameBgHovered] = toVec4(mixColour(plate, theme.accent, 0.18F));
    colours[ImGuiCol_FrameBgActive] = toVec4(mixColour(plate, theme.accent, 0.28F));
    colours[ImGuiCol_Button] = toVec4(withAlpha(theme.accent, 0.14F));
    colours[ImGuiCol_ButtonHovered] = toVec4(withAlpha(theme.accent, 0.32F));
    colours[ImGuiCol_ButtonActive] = toVec4(withAlpha(theme.accent, 0.5F));
    colours[ImGuiCol_Header] = toVec4(withAlpha(theme.accent, 0.22F));
    colours[ImGuiCol_HeaderHovered] = toVec4(withAlpha(theme.accent, 0.34F));
    colours[ImGuiCol_HeaderActive] = toVec4(withAlpha(theme.accent, 0.5F));
    colours[ImGuiCol_SliderGrab] = toVec4(theme.accent);
    colours[ImGuiCol_SliderGrabActive] = toVec4(theme.wakeFlash);
    colours[ImGuiCol_CheckMark] = toVec4(theme.accent);
    colours[ImGuiCol_ScrollbarBg] = toVec4(withAlpha(plate, 0.0F));
    colours[ImGuiCol_ScrollbarGrab] = toVec4(withAlpha(theme.accent, 0.3F));
    colours[ImGuiCol_ScrollbarGrabHovered] = toVec4(withAlpha(theme.accent, 0.5F));
    colours[ImGuiCol_ScrollbarGrabActive] = toVec4(withAlpha(theme.accent, 0.7F));
    colours[ImGuiCol_Separator] = toVec4(withAlpha(theme.accent, 0.3F));
    colours[ImGuiCol_TextSelectedBg] = toVec4(withAlpha(theme.accent, 0.35F));
    colours[ImGuiCol_InputTextCursor] = toVec4(theme.wakeFlash);
    colours[ImGuiCol_NavCursor] = toVec4(theme.wakeFlash);
    style.FrameBorderSize = 1.0F;
    style.FrameRounding = 3.0F;
    style.PopupRounding = 3.0F;
    style.GrabRounding = 2.0F;
    style.ScrollbarRounding = 2.0F;
    style.WindowBorderSize = 0.0F;
    style.PopupBorderSize = 1.0F;
    style.ScaleAllSizes(scale);
    ImGui::GetStyle() = style;
}

}  // namespace jarvis::overlay::hud

// Design notes:
// - Real ImGui items (InvisibleButton, InputText, Button) rather than
//   hand-rolled hit testing: keyboard navigation (Tab, arrows, Enter,
//   Space), tooltips and the caret come for free and behave like every
//   other ImGui UI, while the glyphs are still drawn in the HUD's style.
// - Escape is handled once, after the window: inside the field it cancels
//   the field (TextEntry reports it), otherwise it backs out of the quit
//   confirmation, then out of the expanded panel, and gives focus back.
// - The field closes after a submit in the corner HUD (the reply shows in
//   the transcript rows it covered) but stays in the expanded view, where
//   it is the panel's permanent input row; focus is given back either way.
