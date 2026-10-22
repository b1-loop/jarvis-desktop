#include "jarvis/overlay/layout.hpp"

#include "jarvis/overlay/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::overlay {

namespace {

constexpr std::string_view kEllipsis = "…";
constexpr unsigned char kContinuationMask = 0xC0;
constexpr unsigned char kContinuationBits = 0x80;

bool isContinuationByte(char byte) noexcept {
    return (static_cast<unsigned char>(byte) & kContinuationMask) == kContinuationBits;
}

}  // namespace

Rgba stateColour(const Theme& theme, AssistantState state) noexcept {
    switch (state) {
        case AssistantState::Listening:
            return theme.listening;
        case AssistantState::Thinking:
            return theme.thinking;
        case AssistantState::Speaking:
            return theme.speaking;
        case AssistantState::Idle:
            break;
    }
    return theme.idle;
}

std::string_view stateLabel(AssistantState state) noexcept {
    switch (state) {
        case AssistantState::Listening:
            return "Listening";
        case AssistantState::Thinking:
            return "Thinking";
        case AssistantState::Speaking:
            return "Speaking";
        case AssistantState::Idle:
            break;
    }
    return "Idle";
}

Rgba withAlpha(Rgba colour, float factor) noexcept {
    colour.a = std::clamp(colour.a * factor, 0.0F, 1.0F);
    return colour;
}

std::size_t utf8Length(std::string_view utf8) noexcept {
    return static_cast<std::size_t>(
        std::ranges::count_if(utf8, [](char byte) { return !isContinuationByte(byte); }));
}

std::string truncateUtf8Tail(std::string_view utf8, std::size_t maxChars) {
    if (utf8Length(utf8) <= maxChars) {
        return std::string{utf8};
    }
    if (maxChars == 0) {
        return {};
    }
    // Walk backwards over whole code points until maxChars - 1 remain; the
    // last slot goes to the ellipsis so the result is exactly maxChars wide.
    std::size_t keep = maxChars - 1;
    auto start = utf8.end();
    while (keep > 0 && start != utf8.begin()) {
        --start;
        if (!isContinuationByte(*start)) {
            --keep;
        }
    }
    std::string result{kEllipsis};
    result.append(start, utf8.end());
    return result;
}

std::size_t charactersPerLine(float widthPx, float charWidthPx) noexcept {
    if (widthPx <= 0.0F || charWidthPx <= 0.0F) {
        return 0;
    }
    return static_cast<std::size_t>(std::floor(widthPx / charWidthPx));
}

int scaled(int logicalPx, float scale) noexcept {
    return static_cast<int>(std::lround(static_cast<float>(logicalPx) * scale));
}

Placement anchorToCorner(Corner corner, PixelRect workArea, int width, int height,
                         int margin) noexcept {
    const bool right = corner == Corner::TopRight || corner == Corner::BottomRight;
    const bool bottom = corner == Corner::BottomLeft || corner == Corner::BottomRight;
    return Placement{
        .x = right ? workArea.right - margin - width : workArea.left + margin,
        .y = bottom ? workArea.bottom - margin - height : workArea.top + margin,
        .width = width,
        .height = height,
    };
}

std::vector<float> waveformBars(std::span<const float> levels, std::size_t barCount, float floor) {
    std::vector<float> bars(barCount, std::clamp(floor, 0.0F, 1.0F));
    if (levels.empty() || barCount == 0) {
        return bars;
    }
    std::size_t index = 0;
    for (float& bar : bars) {
        // Integer window bounds; the last bar absorbs any rounding remainder.
        const std::size_t begin = (index * levels.size()) / barCount;
        const std::size_t end = std::max(begin + 1, ((index + 1) * levels.size()) / barCount);
        const auto window = levels.subspan(begin, std::min(end, levels.size()) - begin);
        bar = std::max(bar, std::clamp(std::ranges::max(window), 0.0F, 1.0F));
        ++index;
    }
    return bars;
}

}  // namespace jarvis::overlay

// Design notes:
// - Truncation counts code points, not bytes and not grapheme clusters: a
//   byte count would split characters, while grapheme segmentation needs
//   Unicode tables for a benefit nobody sees on a one-line panel.
// - Bars take the peak of their window rather than the mean so a single loud
//   syllable survives the downsampling; the waveform is feedback, not a meter.
