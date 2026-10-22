#include "panel.hpp"

#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::overlay {

namespace {

/// Fraction of the waveform height that silence still shows, so the panel
/// never looks dead while Jarvis is listening to a quiet room.
constexpr float kSilenceFloor = 0.06F;
/// Oldest bars fade to this fraction of the accent alpha.
constexpr float kOldestBarAlpha = 0.30F;
/// How far the wake-word ring expands beyond the dot while it fades.
constexpr float kWakeRingSpread = 2.5F;
constexpr std::string_view kTranscriptPrefix = "You";
constexpr std::string_view kResponsePrefix = "Jarvis";
constexpr std::string_view kWakeWordCaption = "wake word";

ImVec4 toVec4(Rgba colour) noexcept {
    return ImVec4{colour.r, colour.g, colour.b, colour.a};
}

ImU32 toU32(Rgba colour) noexcept {
    return ImGui::ColorConvertFloat4ToU32(toVec4(colour));
}

ImVec2 textSize(std::string_view text) {
    return ImGui::CalcTextSize(text.data(), text.data() + text.size());
}

void textUnformatted(std::string_view text) {
    ImGui::TextUnformatted(text.data(), text.data() + text.size());
}

void textColoured(Rgba colour, std::string_view text) {
    ImGui::PushStyleColor(ImGuiCol_Text, toVec4(colour));
    textUnformatted(text);
    ImGui::PopStyleColor();
}

/// State dot, its label, and the wake-word ring that expands and fades.
void drawIndicator(const OverlaySnapshot& snapshot, const Theme& theme,
                   const PanelMetrics& metrics) {
    ImDrawList& draw = *ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float lineHeight = ImGui::GetTextLineHeight();
    const ImVec2 centre{origin.x + metrics.dotRadius, origin.y + (lineHeight * 0.5F)};
    const Rgba accent = stateColour(theme, snapshot.state);

    draw.AddCircleFilled(centre, metrics.dotRadius, toU32(accent));
    if (snapshot.wakeFlash > 0.0F) {
        const float spread = 1.0F + ((1.0F - snapshot.wakeFlash) * kWakeRingSpread);
        draw.AddCircle(centre, metrics.dotRadius * spread,
                       toU32(withAlpha(theme.wakeFlash, snapshot.wakeFlash)), 0,
                       metrics.ringThickness);
    }

    ImGui::SetCursorScreenPos(
        {origin.x + (metrics.dotRadius * 2.0F) + (metrics.padding * 0.5F), origin.y});
    textColoured(accent, stateLabel(snapshot.state));

    if (snapshot.wakeFlash > 0.0F) {
        const float captionWidth = textSize(kWakeWordCaption).x;
        const float right = origin.x + ImGui::GetContentRegionAvail().x;
        ImGui::SameLine();
        ImGui::SetCursorScreenPos({right - captionWidth, origin.y});
        textColoured(withAlpha(theme.mutedText, snapshot.wakeFlash), kWakeWordCaption);
    }
}

/// Vertical bars centred on a baseline, newest on the right and brightest.
void drawWaveform(const OverlaySnapshot& snapshot, const Theme& theme,
                  const PanelMetrics& metrics) {
    ImDrawList& draw = *ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    const float pitch = metrics.barWidth + metrics.barGap;
    const auto barCount =
        static_cast<std::size_t>(std::max(0.0F, (width + metrics.barGap) / pitch));
    const std::vector<float> bars = waveformBars(snapshot.levels, barCount, kSilenceFloor);
    const Rgba accent = stateColour(theme, snapshot.state);
    const float centreY = origin.y + (metrics.waveformHeight * 0.5F);

    std::size_t index = 0;
    for (const float bar : bars) {
        const float age = static_cast<float>(index + 1) / static_cast<float>(bars.size());
        const float alpha = kOldestBarAlpha + ((1.0F - kOldestBarAlpha) * age);
        const float half = std::max(bar * metrics.waveformHeight, metrics.barWidth) * 0.5F;
        const float x = origin.x + (static_cast<float>(index) * pitch);
        draw.AddRectFilled({x, centreY - half}, {x + metrics.barWidth, centreY + half},
                           toU32(withAlpha(accent, alpha)), metrics.barWidth * 0.5F);
        ++index;
    }
    ImGui::Dummy({width, metrics.waveformHeight});
}

/// "<prefix>  <text>" on one line, text cut from the front to fit.
void drawTextLine(std::string_view prefix, Rgba prefixColour, const std::string& text,
                  Rgba textColour, const PanelMetrics& metrics) {
    const float width = ImGui::GetContentRegionAvail().x;
    const float prefixWidth = textSize(kResponsePrefix).x + metrics.padding;
    const float charWidth = ImGui::CalcTextSize("n").x;
    const std::size_t fit = charactersPerLine(width - prefixWidth, charWidth);

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    textColoured(prefixColour, prefix);
    ImGui::SameLine();
    ImGui::SetCursorScreenPos({origin.x + prefixWidth, origin.y});
    textColoured(textColour, truncateUtf8Tail(text, fit));
}

}  // namespace

PanelMetrics PanelMetrics::forScale(float scale) noexcept {
    const PanelMetrics base;
    return PanelMetrics{
        .scale = scale,
        .padding = base.padding * scale,
        .dotRadius = base.dotRadius * scale,
        .barWidth = base.barWidth * scale,
        .barGap = base.barGap * scale,
        .waveformHeight = base.waveformHeight * scale,
        .rowGap = base.rowGap * scale,
        .ringThickness = base.ringThickness * scale,
    };
}

void drawPanel(const OverlaySnapshot& snapshot, const Theme& theme, const PanelMetrics& metrics) {
    // ImGui flags are plain ints by design.
    // NOLINTBEGIN(bugprone-signed-bitwise)
    constexpr ImGuiWindowFlags kFlags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove;
    // NOLINTEND(bugprone-signed-bitwise)
    ImGui::SetNextWindowPos({0.0F, 0.0F});
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {metrics.padding, metrics.padding});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0.0F, metrics.rowGap});
    ImGui::Begin("##jarvis-overlay", nullptr, kFlags);

    drawIndicator(snapshot, theme, metrics);
    drawWaveform(snapshot, theme, metrics);
    drawTextLine(kTranscriptPrefix, theme.mutedText, snapshot.transcript, theme.text, metrics);
    drawTextLine(kResponsePrefix, stateColour(theme, AssistantState::Speaking), snapshot.response,
                 theme.text, metrics);

    ImGui::End();
    ImGui::PopStyleVar(2);
}

}  // namespace jarvis::overlay

// Design notes:
// - The ImGui window is borderless and covers the whole swap chain: the
//   Win32 window *is* the panel, so no ImGui-level dragging or resizing.
// - Text is cut by the pure truncateUtf8Tail() rather than ImGui clipping so
//   the user sees an ellipsis and the newest words, not a hard edge.
// - Bars fade with age so the eye reads time left-to-right without an axis.
