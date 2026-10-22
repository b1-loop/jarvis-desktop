#include "jarvis/overlay/layout.hpp"

#include "jarvis/overlay/model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::overlay {

namespace {

constexpr std::string_view kEllipsis = "…";
constexpr unsigned char kContinuationMask = 0xC0;
constexpr unsigned char kContinuationBits = 0x80;

// Boot canvas proportions, as fractions of its side.
constexpr float kBootRingCentre = 0.38F;
constexpr float kBootRingRadius = 0.19F;
constexpr float kBootTextTop = 0.66F;
constexpr float kBootTextHeight = 0.22F;
/// The panel ring leaves a little of the padding as room for its glow.
constexpr float kPanelRingInset = 0.8F;

constexpr PanelMetrics kCompactPanel{.width = 420, .height = 132, .textScale = 1.0F};
constexpr PanelMetrics kNormalPanel{.width = 600, .height = 216, .textScale = 1.25F};
constexpr PanelMetrics kLargePanel{.width = 760, .height = 300, .textScale = 1.5F};
constexpr std::size_t kClockChars = 16;
constexpr std::array<std::string_view, 7> kWeekdays{"SUN", "MON", "TUE", "WED",
                                                    "THU", "FRI", "SAT"};
constexpr std::array<std::string_view, 12> kMonths{"JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                                                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"};

bool isContinuationByte(char byte) noexcept {
    return (static_cast<unsigned char>(byte) & kContinuationMask) == kContinuationBits;
}

float mix(float from, float to, float t) noexcept {
    return from + ((to - from) * t);
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

std::string_view hudLabel(AssistantState state) noexcept {
    switch (state) {
        case AssistantState::Listening:
            return "LISTENING";
        case AssistantState::Thinking:
            return "PROCESSING";
        case AssistantState::Speaking:
            return "SPEAKING";
        case AssistantState::Idle:
            break;
    }
    return "STANDBY";
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

std::string truncateUtf8Head(std::string_view utf8, std::size_t maxChars) {
    if (utf8Length(utf8) <= maxChars) {
        return std::string{utf8};
    }
    if (maxChars == 0) {
        return {};
    }
    // Walk forwards over maxChars - 1 whole code points; the ellipsis takes
    // the last slot. A code point ends where the next lead byte starts.
    std::size_t keep = maxChars - 1;
    auto end = utf8.begin();
    while (end != utf8.end()) {
        if (!isContinuationByte(*end)) {
            if (keep == 0) {
                break;
            }
            --keep;
        }
        ++end;
    }
    std::string result{utf8.begin(), end};
    result.append(kEllipsis);
    return result;
}

std::string_view utf8Prefix(std::string_view utf8, std::size_t codePoints) {
    std::size_t seen = 0;
    std::size_t bytes = 0;
    for (const char byte : utf8) {
        if (!isContinuationByte(byte)) {
            if (seen == codePoints) {
                return utf8.substr(0, bytes);
            }
            ++seen;
        }
        ++bytes;
    }
    return utf8;
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

RectF toRectF(Placement placement) noexcept {
    return RectF{.x = static_cast<float>(placement.x),
                 .y = static_cast<float>(placement.y),
                 .width = static_cast<float>(placement.width),
                 .height = static_cast<float>(placement.height)};
}

RectF lerpRect(RectF from, RectF to, float t) noexcept {
    return RectF{.x = mix(from.x, to.x, t),
                 .y = mix(from.y, to.y, t),
                 .width = mix(from.width, to.width, t),
                 .height = mix(from.height, to.height, t)};
}

Placement centredIn(PixelRect workArea, int width, int height) noexcept {
    return Placement{.x = workArea.left + ((workArea.right - workArea.left - width) / 2),
                     .y = workArea.top + ((workArea.bottom - workArea.top - height) / 2),
                     .width = width,
                     .height = height};
}

RectF translated(RectF rect, float dx, float dy) noexcept {
    rect.x += dx;
    rect.y += dy;
    return rect;
}

Placement inflated(Placement placement, int amount) noexcept {
    return Placement{.x = placement.x - amount,
                     .y = placement.y - amount,
                     .width = placement.width + (2 * amount),
                     .height = placement.height + (2 * amount)};
}

Placement enclosingPlacement(RectF rect) noexcept {
    const auto left = static_cast<int>(std::floor(rect.x));
    const auto top = static_cast<int>(std::floor(rect.y));
    const auto right = static_cast<int>(std::ceil(rect.right()));
    const auto bottom = static_cast<int>(std::ceil(rect.bottom()));
    return Placement{.x = left, .y = top, .width = right - left, .height = bottom - top};
}

std::string_view toString(PanelSize size) noexcept {
    switch (size) {
        case PanelSize::Compact:
            return "compact";
        case PanelSize::Normal:
            return "normal";
        case PanelSize::Large:
            break;
    }
    return "large";
}

std::optional<PanelSize> parsePanelSize(std::string_view name) noexcept {
    for (const PanelSize size : {PanelSize::Compact, PanelSize::Normal, PanelSize::Large}) {
        if (toString(size) == name) {
            return size;
        }
    }
    return std::nullopt;
}

PanelMetrics panelMetrics(PanelSize size) noexcept {
    switch (size) {
        case PanelSize::Compact:
            return kCompactPanel;
        case PanelSize::Normal:
            return kNormalPanel;
        case PanelSize::Large:
            break;
    }
    return kLargePanel;
}

PanelMetrics withOverride(PanelMetrics metrics, int width, int height) noexcept {
    if (width > 0) {
        metrics.width = width;
    }
    if (height > 0) {
        metrics.height = height;
    }
    return metrics;
}

std::string formatClock(const std::tm& local) {
    std::array<char, kClockChars> buffer{};
    const std::size_t written = std::strftime(buffer.data(), buffer.size(), "%H:%M:%S", &local);
    return std::string{buffer.data(), written};
}

std::string formatUptime(float seconds) {
    constexpr long kMinute = 60;
    constexpr long kHour = 3600;
    const long total = std::max(0L, static_cast<long>(seconds));
    const auto twoDigits = [](long value) {
        return (value < 10 ? "0" : "") + std::to_string(value);
    };
    return twoDigits(total / kHour) + ":" + twoDigits((total % kHour) / kMinute) + ":" +
           twoDigits(total % kMinute);
}

std::string formatDate(const std::tm& local) {
    const auto day = static_cast<std::size_t>(std::clamp(local.tm_wday, 0, 6));
    const auto month = static_cast<std::size_t>(std::clamp(local.tm_mon, 0, 11));
    return std::string{kWeekdays.at(day)} + " " + std::to_string(local.tm_mday) + " " +
           std::string{kMonths.at(month)};
}

HudGeometry panelGeometry(RectF panel, float padding) noexcept {
    const float radius = std::max((panel.height * 0.5F) - (padding * kPanelRingInset), 0.0F);
    const float centreX = panel.x + padding + radius;
    const float textLeft = centreX + radius + padding;
    return HudGeometry{
        .centreX = centreX,
        .centreY = panel.y + (panel.height * 0.5F),
        .radius = radius,
        .text = RectF{.x = textLeft,
                      .y = panel.y + padding,
                      .width = std::max(panel.right() - padding - textLeft, 0.0F),
                      .height = std::max(panel.height - (2.0F * padding), 0.0F)},
    };
}

HudGeometry bootGeometry(RectF canvas) noexcept {
    const float side = std::min(canvas.width, canvas.height);
    return HudGeometry{
        .centreX = canvas.x + (canvas.width * 0.5F),
        .centreY = canvas.y + (side * kBootRingCentre),
        .radius = side * kBootRingRadius,
        .text = RectF{.x = canvas.x,
                      .y = canvas.y + (side * kBootTextTop),
                      .width = canvas.width,
                      .height = side * kBootTextHeight},
    };
}

HudGeometry translated(const HudGeometry& geometry, float dx, float dy) noexcept {
    return HudGeometry{.centreX = geometry.centreX + dx,
                       .centreY = geometry.centreY + dy,
                       .radius = geometry.radius,
                       .text = translated(geometry.text, dx, dy)};
}

HudGeometry lerpGeometry(const HudGeometry& from, const HudGeometry& to, float t) noexcept {
    return HudGeometry{.centreX = mix(from.centreX, to.centreX, t),
                       .centreY = mix(from.centreY, to.centreY, t),
                       .radius = mix(from.radius, to.radius, t),
                       .text = lerpRect(from.text, to.text, t)};
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
// - Geometry is float until the very end (enclosingPlacement) so the boot
//   handoff interpolates smoothly; only the Win32 window rectangle is
//   rounded, outwards, so nothing drawn is ever clipped by a pixel.
// - Truncation counts code points, not bytes and not grapheme clusters: a
//   byte count would split characters, while grapheme segmentation needs
//   Unicode tables for a benefit nobody sees on a one-line panel.
// - Bars take the peak of their window rather than the mean so a single loud
//   syllable survives the downsampling; the waveform is feedback, not a meter.
