/// @file layout.hpp
/// @brief What to draw for a snapshot, computed without any UI library.
///
/// Everything here is a pure function of its inputs so the panel's visual
/// decisions (colours, labels, how text is cut, how levels become bars) are
/// unit-tested without a window, a GPU or Dear ImGui. panel.cpp turns the
/// results into draw calls and nothing else.

#pragma once

#include "jarvis/overlay/model.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::overlay {

/// Straight-alpha colour with channels in [0, 1].
struct Rgba {
    float r{0.0F};
    float g{0.0F};
    float b{0.0F};
    float a{1.0F};

    [[nodiscard]] bool operator==(const Rgba&) const noexcept = default;
};

/// Palette of the panel. Defaults are a dark panel with one accent per state
/// so the state is readable from colour alone, before the label is.
struct Theme {
    Rgba background{0.07F, 0.08F, 0.11F, 1.0F};
    Rgba text{0.93F, 0.94F, 0.96F, 1.0F};
    Rgba mutedText{0.60F, 0.63F, 0.70F, 1.0F};
    Rgba idle{0.55F, 0.60F, 0.70F, 1.0F};
    Rgba listening{0.30F, 0.85F, 0.45F, 1.0F};
    Rgba thinking{0.98F, 0.75F, 0.25F, 1.0F};
    Rgba speaking{0.35F, 0.65F, 1.00F, 1.0F};
    Rgba wakeFlash{1.00F, 1.00F, 1.00F, 1.0F};
};

/// Accent colour for @p state.
[[nodiscard]] Rgba stateColour(const Theme& theme, AssistantState state) noexcept;

/// Human label shown next to the indicator, e.g. "Listening".
[[nodiscard]] std::string_view stateLabel(AssistantState state) noexcept;

/// @return @p colour with its alpha multiplied by @p factor (clamped to [0, 1]).
[[nodiscard]] Rgba withAlpha(Rgba colour, float factor) noexcept;

/// Number of Unicode code points in @p utf8. Continuation bytes are not
/// counted, so multi-byte characters count once.
[[nodiscard]] std::size_t utf8Length(std::string_view utf8) noexcept;

/// Keep the last @p maxChars code points of @p utf8, prefixed with an
/// ellipsis when anything was cut. The tail is kept because the newest words
/// of a transcript are the ones the user is waiting to see confirmed.
/// Never splits a multi-byte sequence; @p maxChars of 0 yields "".
[[nodiscard]] std::string truncateUtf8Tail(std::string_view utf8, std::size_t maxChars);

/// How many characters of average width @p charWidthPx fit in @p widthPx.
/// Zero or negative inputs yield 0 rather than dividing by zero.
[[nodiscard]] std::size_t charactersPerLine(float widthPx, float charWidthPx) noexcept;

/// Screen corner the panel is anchored to, inside the monitor's work area.
enum class Corner : std::uint8_t { TopLeft, TopRight, BottomLeft, BottomRight };

/// Axis-aligned rectangle in physical pixels, edges inclusive/exclusive as Win32.
struct PixelRect {
    int left{0};
    int top{0};
    int right{0};
    int bottom{0};

    [[nodiscard]] bool operator==(const PixelRect&) const noexcept = default;
};

/// Where to put a window of a given size.
struct Placement {
    int x{0};
    int y{0};
    int width{0};
    int height{0};

    [[nodiscard]] bool operator==(const Placement&) const noexcept = default;
};

/// Round a device-independent size to physical pixels at @p scale.
[[nodiscard]] int scaled(int logicalPx, float scale) noexcept;

/// Anchor a @p width x @p height window to @p corner of @p workArea, keeping
/// @p margin from both edges. Sizes and margin are already in physical pixels.
[[nodiscard]] Placement anchorToCorner(Corner corner, PixelRect workArea, int width, int height,
                                       int margin) noexcept;

/// Resample @p levels (oldest first) into @p barCount bar heights in [0, 1].
///
/// When there are more levels than bars, each bar takes the peak of its
/// window so short transients still show. Every bar is at least @p floor
/// tall so silence draws as a resting line, not as nothing.
[[nodiscard]] std::vector<float> waveformBars(std::span<const float> levels, std::size_t barCount,
                                              float floor);

}  // namespace jarvis::overlay
