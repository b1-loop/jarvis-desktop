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
#include <ctime>
#include <optional>
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

/// Palette of the HUD. One hue family per theme, with the state told apart
/// by brightness and a small hue shift (and, above all, by motion), so the
/// panel reads as one instrument rather than four traffic lights. The named
/// themes and the derivation from one accent are in palette.hpp.
struct Theme {
    /// Clear colour of the opaque fallback (whole-window alpha) path.
    Rgba background{0.02F, 0.05F, 0.08F, 1.0F};
    Rgba text{0.86F, 0.97F, 1.00F, 1.0F};
    Rgba mutedText{0.45F, 0.70F, 0.80F, 1.0F};
    Rgba idle{0.25F, 0.62F, 0.80F, 1.0F};
    Rgba listening{0.35F, 0.93F, 1.00F, 1.0F};
    Rgba thinking{0.55F, 0.78F, 1.00F, 1.0F};
    Rgba speaking{0.40F, 1.00F, 0.88F, 1.0F};
    Rgba wakeFlash{0.85F, 1.00F, 1.00F, 1.0F};
    /// The light the rings are drawn in.
    Rgba accent{0.30F, 0.85F, 1.00F, 1.0F};
    /// Glass plate behind the text so it stays readable over any desktop.
    Rgba plate{0.01F, 0.06F, 0.10F, 0.55F};
    /// Danger and alert light (quit control, muted microphone, the quit
    /// question); always a hue far from the accent so it never reads as a state.
    Rgba warning{1.00F, 0.45F, 0.35F, 1.0F};
    /// Multiplier on the bloom strength: pale accents carry light in all three
    /// channels and would otherwise bloom into a white smear.
    float bloom{1.0F};

    [[nodiscard]] bool operator==(const Theme&) const noexcept = default;
};

/// Accent colour for @p state.
[[nodiscard]] Rgba stateColour(const Theme& theme, AssistantState state) noexcept;

/// Human label shown next to the indicator, e.g. "Listening".
[[nodiscard]] std::string_view stateLabel(AssistantState state) noexcept;

/// Upper-case caption of the HUD, e.g. "STANDBY" for Idle.
[[nodiscard]] std::string_view hudLabel(AssistantState state) noexcept;

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

/// Keep the first @p maxChars code points of @p utf8, ending with an
/// ellipsis when anything was cut; for labels and values whose start is
/// the informative part (a model or device name). Never splits a
/// multi-byte sequence; @p maxChars of 0 yields "".
[[nodiscard]] std::string truncateUtf8Head(std::string_view utf8, std::size_t maxChars);

/// The first @p codePoints code points of @p utf8 (all of it if shorter),
/// as a view into it; used to type text on one character at a time.
[[nodiscard]] std::string_view utf8Prefix(std::string_view utf8, std::size_t codePoints);

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

/// Axis-aligned rectangle with float coordinates, for animated geometry.
struct RectF {
    float x{0.0F};
    float y{0.0F};
    float width{0.0F};
    float height{0.0F};

    [[nodiscard]] float right() const noexcept { return x + width; }
    [[nodiscard]] float bottom() const noexcept { return y + height; }
    [[nodiscard]] bool operator==(const RectF&) const noexcept = default;
};

/// @p placement as a RectF.
[[nodiscard]] RectF toRectF(Placement placement) noexcept;

/// Interpolate position and size from @p from (t = 0) to @p to (t = 1).
[[nodiscard]] RectF lerpRect(RectF from, RectF to, float t) noexcept;

/// A @p width x @p height window centred in @p workArea.
[[nodiscard]] Placement centredIn(PixelRect workArea, int width, int height) noexcept;

/// @p rect moved by (@p dx, @p dy).
[[nodiscard]] RectF translated(RectF rect, float dx, float dy) noexcept;

/// @p placement grown by @p amount on every side.
[[nodiscard]] Placement inflated(Placement placement, int amount) noexcept;

/// Smallest whole-pixel placement containing @p rect.
[[nodiscard]] Placement enclosingPlacement(RectF rect) noexcept;

/// Where the HUD's parts sit inside a canvas, in the same pixel space.
struct HudGeometry {
    /// Centre of the ring.
    float centreX{0.0F};
    float centreY{0.0F};
    /// Radius of the outermost ring.
    float radius{0.0F};
    /// Area for text: the panel's column, or the boot title block.
    RectF text;

    [[nodiscard]] bool operator==(const HudGeometry&) const noexcept = default;
};

/// How big the corner HUD is.
enum class PanelSize : std::uint8_t { Compact, Normal, Large };

/// Lower-case name as used in settings ("compact", "normal", "large").
[[nodiscard]] std::string_view toString(PanelSize size) noexcept;

/// Inverse of toString(PanelSize); std::nullopt for anything else.
[[nodiscard]] std::optional<PanelSize> parsePanelSize(std::string_view name) noexcept;

/// Dimensions of a panel size at 96 DPI and how much its text is scaled.
struct PanelMetrics {
    int width{0};
    int height{0};
    float textScale{1.0F};

    [[nodiscard]] bool operator==(const PanelMetrics&) const noexcept = default;
};

/// The metrics of @p size: Compact is the ADR-0011 strip (420 x 132),
/// Large a showpiece HUD with a ring of about 280 px (760 x 300).
[[nodiscard]] PanelMetrics panelMetrics(PanelSize size) noexcept;

/// @p metrics with an explicit @p width / @p height override applied;
/// 0 keeps the preset's value for that side.
[[nodiscard]] PanelMetrics withOverride(PanelMetrics metrics, int width, int height) noexcept;

/// "14:32:07" for @p local.
[[nodiscard]] std::string formatClock(const std::tm& local);

/// "02:14:07" for @p seconds of uptime; hours keep counting past 99.
[[nodiscard]] std::string formatUptime(float seconds);

/// "WED 22 OCT" for @p local; HUD dates are upper-case English on purpose
/// (they are part of the instrument, not of the user's locale).
[[nodiscard]] std::string formatDate(const std::tm& local);

/// Corner panel: ring on the left, text column on the right.
[[nodiscard]] HudGeometry panelGeometry(RectF panel, float padding) noexcept;

/// Boot canvas: ring in the upper half, title block underneath.
[[nodiscard]] HudGeometry bootGeometry(RectF canvas) noexcept;

/// @p geometry moved by (@p dx, @p dy), e.g. from screen to client coordinates.
[[nodiscard]] HudGeometry translated(const HudGeometry& geometry, float dx, float dy) noexcept;

/// Interpolate every part of the geometry; used for the boot handoff.
[[nodiscard]] HudGeometry lerpGeometry(const HudGeometry& from, const HudGeometry& to,
                                       float t) noexcept;

/// Resample @p levels (oldest first) into @p barCount bar heights in [0, 1].
///
/// When there are more levels than bars, each bar takes the peak of its
/// window so short transients still show. Every bar is at least @p floor
/// tall so silence draws as a resting line, not as nothing.
[[nodiscard]] std::vector<float> waveformBars(std::span<const float> levels, std::size_t barCount,
                                              float floor);

}  // namespace jarvis::overlay
