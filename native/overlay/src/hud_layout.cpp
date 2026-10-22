#include "jarvis/overlay/hud_layout.hpp"

#include "jarvis/overlay/layout.hpp"

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <string_view>

namespace jarvis::overlay {

namespace {

// Corner HUD, in 96-DPI pixels before the size preset's text scale.
constexpr float kCaptionSize = 15.0F;  // must match hud_panel.cpp
constexpr float kButtonShare = 1.3F;   // button side as a share of the caption size
constexpr float kButtonGap = 5.0F;
constexpr float kBodySize = 14.5F;  // must match hud_panel.cpp
constexpr float kInputShare = 1.75F;
constexpr float kConfirmRowDrop = 2.2F;
constexpr float kConfirmButtonWidth = 64.0F;
constexpr float kConfirmGap = 10.0F;

// Expanded panel, 96-DPI pixels.
constexpr float kPadding = 18.0F;
constexpr float kRingRadius = 48.0F;
constexpr float kExpandedButton = 26.0F;
constexpr float kExpandedGap = 6.0F;
constexpr float kTabHeight = 28.0F;
constexpr float kTabWidth = 150.0F;
constexpr float kSectionGap = 10.0F;
constexpr float kInputHeight = 36.0F;
constexpr float kGripSize = 18.0F;

bool contains(const HitShape& shape, float x, float y) noexcept {
    const RectF& box = shape.bounds;
    if (shape.form == HitShape::Form::Rect) {
        return x >= box.x && x < box.right() && y >= box.y && y < box.bottom();
    }
    const float radius = box.width * 0.5F;
    const float dx = x - (box.x + radius);
    const float dy = y - (box.y + radius);
    return (dx * dx) + (dy * dy) <= radius * radius;
}

RectF grown(RectF rect, float amount) noexcept {
    return RectF{.x = rect.x - amount,
                 .y = rect.y - amount,
                 .width = rect.width + (2.0F * amount),
                 .height = rect.height + (2.0F * amount)};
}

}  // namespace

std::string_view controlLabel(HudControl control) noexcept {
    switch (control) {
        case HudControl::Keyboard:
            return "Type a request";
        case HudControl::Mute:
            return "Mute or unmute the microphone";
        case HudControl::Settings:
            return "Settings";
        case HudControl::Expand:
            return "Expand or collapse";
        case HudControl::Quit:
            return "Quit Jarvis";
        case HudControl::Minimize:
            return "Minimize to the orb (click the orb to restore)";
        case HudControl::Close:
            break;
    }
    return "Hide until needed (waking Jarvis brings it back)";
}

RectF ControlBar::button(HudControl control) const noexcept {
    const auto found = std::ranges::find(kHudControls, control);
    const auto index = static_cast<std::size_t>(found - kHudControls.begin());
    return index < buttons.size() ? buttons.at(index) : RectF{};
}

ControlBar controlBar(float right, float top, float size, float gap) noexcept {
    ControlBar bar;
    const auto count = static_cast<float>(kHudControlCount);
    const float left = right - (count * size) - ((count - 1.0F) * gap);
    float x = left;
    for (RectF& button : bar.buttons) {
        button = RectF{.x = x, .y = top, .width = size, .height = size};
        x += size + gap;
    }
    bar.bounds = RectF{.x = left, .y = top, .width = right - left, .height = size};
    return bar;
}

CollapsedLayout collapsedLayout(RectF column, float scale, float textScale) noexcept {
    const float unit = scale * textScale;
    const float caption = kCaptionSize * unit;
    const float button = caption * kButtonShare;
    const float top = column.y + ((caption - button) * 0.5F);
    const float inputHeight = kBodySize * unit * kInputShare;
    const float body = kBodySize * unit;
    const float rowTop = column.y + (caption * kConfirmRowDrop);
    const float yesX = column.x;
    const float buttonWidth = kConfirmButtonWidth * unit;
    const float buttonTop = rowTop + (body * 1.8F);
    return CollapsedLayout{
        .controls = controlBar(column.right(), top, button, kButtonGap * unit),
        .input = RectF{.x = column.x,
                       .y = column.bottom() - inputHeight,
                       .width = column.width,
                       .height = inputHeight},
        .confirmText = RectF{.x = column.x, .y = rowTop, .width = column.width, .height = body},
        .confirmYes = RectF{.x = yesX, .y = buttonTop, .width = buttonWidth, .height = inputHeight},
        .confirmNo = RectF{.x = yesX + buttonWidth + (kConfirmGap * unit),
                           .y = buttonTop,
                           .width = buttonWidth,
                           .height = inputHeight},
    };
}

namespace {

/// The expanded panel's header: the ring, the caption beside it, the controls.
void layoutHeader(ExpandedLayout& layout, float scale) noexcept {
    const RectF& frame = layout.frame;
    const float pad = kPadding * scale;
    const float radius = kRingRadius * scale;
    layout.ring = HudGeometry{.centreX = frame.x + pad + radius,
                              .centreY = frame.y + pad + radius,
                              .radius = radius,
                              .text = {}};
    layout.controls = controlBar(frame.right() - pad, frame.y + pad, kExpandedButton * scale,
                                 kExpandedGap * scale);
    const float captionLeft = frame.x + (2.0F * (pad + radius));
    layout.caption = RectF{.x = captionLeft,
                           .y = frame.y + pad,
                           .width = std::max(layout.controls.bounds.x - captionLeft - pad, 0.0F),
                           .height = 2.0F * radius};
}

/// Below the header: the two tabs, the body, the input row and the grip.
void layoutContent(ExpandedLayout& layout, float scale) noexcept {
    const RectF& frame = layout.frame;
    const float pad = kPadding * scale;
    const float gap = kExpandedGap * scale;
    const float tabTop = frame.y + (2.0F * (pad + (kRingRadius * scale)));
    const float tabHeight = kTabHeight * scale;
    const float tabWidth = kTabWidth * scale;
    const float inputHeight = kInputHeight * scale;
    const float grip = kGripSize * scale;
    const float inputTop = frame.bottom() - pad - inputHeight;
    const float bodyTop = tabTop + tabHeight + (kSectionGap * scale);
    const float innerWidth = std::max(frame.width - (2.0F * pad), 0.0F);
    layout.tabConversation =
        RectF{.x = frame.x + pad, .y = tabTop, .width = tabWidth, .height = tabHeight};
    layout.tabSettings = RectF{
        .x = frame.x + pad + tabWidth + gap, .y = tabTop, .width = tabWidth, .height = tabHeight};
    layout.body = RectF{.x = frame.x + pad,
                        .y = bodyTop,
                        .width = innerWidth,
                        .height = std::max(inputTop - (kSectionGap * scale) - bodyTop, 0.0F)};
    layout.input = RectF{.x = frame.x + pad,
                         .y = inputTop,
                         .width = std::max(innerWidth - grip - gap, 0.0F),
                         .height = inputHeight};
    layout.grip =
        RectF{.x = frame.right() - grip, .y = frame.bottom() - grip, .width = grip, .height = grip};
}

}  // namespace

ExpandedLayout expandedLayout(RectF frame, float scale) noexcept {
    ExpandedLayout layout{.frame = frame};
    layoutHeader(layout, scale);
    layoutContent(layout, scale);
    return layout;
}

void HitMap::addRect(RectF rect, HitKind kind) {
    shapes_.push_back(HitShape{.form = HitShape::Form::Rect, .bounds = rect, .kind = kind});
}

void HitMap::addCircle(float centreX, float centreY, float radius, HitKind kind) {
    shapes_.push_back(HitShape{.form = HitShape::Form::Circle,
                               .bounds = RectF{.x = centreX - radius,
                                               .y = centreY - radius,
                                               .width = 2.0F * radius,
                                               .height = 2.0F * radius},
                               .kind = kind});
}

HitKind HitMap::test(float x, float y) const noexcept {
    for (const HitShape& shape : std::views::reverse(shapes_)) {
        if (contains(shape, x, y)) {
            return shape.kind;
        }
    }
    return HitKind::None;
}

void collapsedHitMap(HitMap& map, const HudGeometry& ring, RectF column,
                     const CollapsedLayout& layout, CollapsedHits open, float feather) {
    map.clear();
    map.addCircle(ring.centreX, ring.centreY, ring.radius, HitKind::Drag);
    map.addRect(grown(column, feather), HitKind::Drag);
    for (const RectF& button : layout.controls.buttons) {
        map.addRect(button, HitKind::Control);
    }
    if (open.inputOpen) {
        map.addRect(layout.input, HitKind::Control);
    }
    if (open.confirming) {
        map.addRect(layout.confirmYes, HitKind::Control);
        map.addRect(layout.confirmNo, HitKind::Control);
    }
}

void expandedHitMap(HitMap& map, const ExpandedLayout& layout) {
    map.clear();
    map.addRect(layout.frame, HitKind::Drag);
    map.addRect(layout.tabConversation, HitKind::Control);
    map.addRect(layout.tabSettings, HitKind::Control);
    map.addRect(layout.body, HitKind::Control);
    map.addRect(layout.input, HitKind::Control);
    for (const RectF& button : layout.controls.buttons) {
        map.addRect(button, HitKind::Control);
    }
    map.addRect(layout.grip, HitKind::Resize);
}

void orbHitMap(HitMap& map, float centreX, float centreY, float radius) {
    map.clear();
    map.addCircle(centreX, centreY, radius, HitKind::Drag);
}

float approach(float value, float target, float dtSeconds, float seconds) noexcept {
    if (seconds <= 0.0F) {
        return target;
    }
    const float step = std::max(dtSeconds, 0.0F) / seconds;
    return value < target ? std::min(value + step, target) : std::max(value - step, target);
}

}  // namespace jarvis::overlay

// Design notes:
// - The ring's whole disc drags, not only its stroke: a 1-pixel ring would
//   be unclickable, and the disc is where the dark scrim makes the HUD
//   look solid anyway. Gaps between buttons drag too, which is what a user
//   aiming between two icons expects.
// - Quit sits apart from Close (third, behind a confirmation) so a missed
//   click on "hide" can never end the session.
// - The expanded panel's header, margins and frame drag; everything with
//   content (tabs, the scrollable body, the input row) belongs to ImGui.
