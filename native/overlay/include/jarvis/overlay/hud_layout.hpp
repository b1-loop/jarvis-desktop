/// @file hud_layout.hpp
/// @brief Where the interactive parts of the HUD are, and what is clickable.
///
/// The corner HUD stays click-through except over its own shapes (ADR-0024).
/// This header decides those shapes for both views, the collapsed corner
/// HUD and the expanded panel, as pure functions of the panel geometry:
/// the hover control bar, the text field, the quit confirmation, the
/// expanded panel's header, body, input row and resize grip, and the
/// HitMap the window procedure consults for WM_NCHITTEST, the cursor shape
/// and dragging. The renderer draws at exactly these rectangles, so what is
/// visible and what is clickable cannot drift apart.

#pragma once

#include "jarvis/overlay/layout.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace jarvis::overlay {

/// The hover controls (kHudControls gives their order, left to right).
enum class HudControl : std::uint8_t { Keyboard, Mute, Settings, Expand, Quit, Close, Minimize };

inline constexpr std::size_t kHudControlCount = 7;
inline constexpr std::array<HudControl, kHudControlCount> kHudControls{
    HudControl::Keyboard, HudControl::Mute,     HudControl::Quit, HudControl::Settings,
    HudControl::Expand,   HudControl::Minimize, HudControl::Close};

/// Short description shown as the tooltip, e.g. "Type a request".
[[nodiscard]] std::string_view controlLabel(HudControl control) noexcept;

/// Row of square buttons, in kHudControls order.
struct ControlBar {
    std::array<RectF, kHudControlCount> buttons{};
    /// Smallest rectangle around every button.
    RectF bounds;

    /// The rectangle of @p control.
    [[nodiscard]] RectF button(HudControl control) const noexcept;
    [[nodiscard]] bool operator==(const ControlBar&) const noexcept = default;
};

/// Buttons of side @p size with @p gap between them, right-aligned to
/// @p right with their top at @p top.
[[nodiscard]] ControlBar controlBar(float right, float top, float size, float gap) noexcept;

/// Interactive parts of the corner HUD, in the same space as its column.
struct CollapsedLayout {
    /// On the caption line, right-aligned; the brand fades out under it.
    ControlBar controls;
    /// Text field along the bottom of the column (where the telemetry is).
    RectF input;
    /// The quit confirmation replaces the transcript rows.
    RectF confirmText;
    RectF confirmYes;
    RectF confirmNo;

    [[nodiscard]] bool operator==(const CollapsedLayout&) const noexcept = default;
};

/// Layout for the text column @p column at DPI @p scale and the size
/// preset's @p textScale.
[[nodiscard]] CollapsedLayout collapsedLayout(RectF column, float scale, float textScale) noexcept;

/// Default and smallest size of the expanded panel, 96-DPI pixels.
inline constexpr int kExpandedWidth = 900;
inline constexpr int kExpandedHeight = 640;
inline constexpr int kExpandedMinWidth = 560;
inline constexpr int kExpandedMinHeight = 420;

/// Interactive parts of the expanded panel.
struct ExpandedLayout {
    RectF frame;
    /// The live ring in the header (top-left).
    HudGeometry ring;
    /// State caption and clock, right of the ring.
    RectF caption;
    ControlBar controls;
    RectF tabConversation;
    RectF tabSettings;
    /// Scrollable history or the settings page.
    RectF body;
    /// Text field at the bottom.
    RectF input;
    /// Bottom-right resize grip.
    RectF grip;

    [[nodiscard]] bool operator==(const ExpandedLayout&) const noexcept = default;
};

/// Layout of an expanded panel occupying @p frame at DPI @p scale.
[[nodiscard]] ExpandedLayout expandedLayout(RectF frame, float scale) noexcept;

/// What a point of the window does.
enum class HitKind : std::uint8_t {
    /// Not part of the HUD: clicks fall through to the window underneath.
    None,
    /// Part of the HUD's body: dragging moves it.
    Drag,
    /// A control, the text field or a widget: ImGui handles the click.
    Control,
    /// The resize grip of the expanded panel.
    Resize,
};

/// One clickable shape.
struct HitShape {
    enum class Form : std::uint8_t { Rect, Circle };
    Form form{Form::Rect};
    /// The rectangle, or the circle's bounding box.
    RectF bounds;
    HitKind kind{HitKind::None};
};

/// Ordered shapes; a later shape wins where shapes overlap, so controls
/// are added after the body they sit on.
class HitMap {
public:
    void clear() noexcept { shapes_.clear(); }
    void addRect(RectF rect, HitKind kind);
    void addCircle(float centreX, float centreY, float radius, HitKind kind);

    /// What the point (@p x, @p y) does; None outside every shape.
    [[nodiscard]] HitKind test(float x, float y) const noexcept;
    [[nodiscard]] std::span<const HitShape> shapes() const noexcept { return shapes_; }
    [[nodiscard]] bool empty() const noexcept { return shapes_.empty(); }

private:
    std::vector<HitShape> shapes_;
};

/// What is open in the corner HUD, which changes what is clickable.
struct CollapsedHits {
    bool inputOpen{false};
    bool confirming{false};
};

/// Fill @p map for the corner HUD: the ring disc and the column (grown by
/// @p feather, the scrim's soft edge) drag; the controls, and the input
/// or the confirmation when open, are controls.
void collapsedHitMap(HitMap& map, const HudGeometry& ring, RectF column,
                     const CollapsedLayout& layout, CollapsedHits open, float feather);

/// Fill @p map for the expanded panel: the frame drags, the tabs, body,
/// input and controls are controls, the grip resizes.
void expandedHitMap(HitMap& map, const ExpandedLayout& layout);

/// Fill @p map for the minimized orb: its disc of @p radius around
/// (@p centreX, @p centreY) drags (and a click on it restores the HUD);
/// everything around it, the halo included, stays click-through.
void orbHitMap(HitMap& map, float centreX, float centreY, float radius);

/// Seconds the expand/collapse morph takes.
inline constexpr float kMorphSeconds = 0.42F;
/// Seconds the hover controls take to fade in or out.
inline constexpr float kHoverFadeSeconds = 0.18F;
/// Seconds the controls stay after the cursor left the HUD.
inline constexpr float kHoverLingerSeconds = 0.6F;

/// @p value moved towards @p target (both in [0, 1]) at the rate that
/// covers the whole range in @p seconds, without overshooting; frame-rate
/// independent because the step is proportional to @p dtSeconds.
[[nodiscard]] float approach(float value, float target, float dtSeconds, float seconds) noexcept;

}  // namespace jarvis::overlay
