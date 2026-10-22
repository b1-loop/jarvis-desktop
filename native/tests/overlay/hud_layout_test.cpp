#include "jarvis/overlay/hud_layout.hpp"

#include "jarvis/overlay/layout.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using jarvis::overlay::approach;
using jarvis::overlay::collapsedHitMap;
using jarvis::overlay::CollapsedHits;
using jarvis::overlay::collapsedLayout;
using jarvis::overlay::controlBar;
using jarvis::overlay::expandedHitMap;
using jarvis::overlay::expandedLayout;
using jarvis::overlay::HitKind;
using jarvis::overlay::HitMap;
using jarvis::overlay::HudControl;
using jarvis::overlay::kHudControlCount;
using jarvis::overlay::kHudControls;
using jarvis::overlay::panelGeometry;
using jarvis::overlay::RectF;

namespace {

bool inside(const RectF& inner, const RectF& outer) {
    return inner.x >= outer.x && inner.y >= outer.y && inner.right() <= outer.right() + 0.01F &&
           inner.bottom() <= outer.bottom() + 0.01F;
}

bool disjoint(const RectF& a, const RectF& b) {
    return a.right() <= b.x || b.right() <= a.x || a.bottom() <= b.y || b.bottom() <= a.y;
}

/// The large preset's panel at 96 DPI, as the window lays it out.
constexpr RectF kPanel{.x = 1000.0F, .y = 700.0F, .width = 760.0F, .height = 300.0F};
constexpr RectF kFrame{.x = 100.0F, .y = 100.0F, .width = 900.0F, .height = 640.0F};

}  // namespace

TEST_CASE("the control bar is right-aligned, in order, without overlaps", "[overlay][hud]") {
    const auto bar = controlBar(500.0F, 20.0F, 24.0F, 6.0F);
    REQUIRE(bar.buttons.back().right() == Approx(500.0F));
    REQUIRE(bar.bounds.width == Approx((6.0F * 24.0F) + (5.0F * 6.0F)));
    for (std::size_t i = 1; i < kHudControlCount; ++i) {
        REQUIRE(bar.buttons.at(i).x > bar.buttons.at(i - 1).right());
    }
    REQUIRE(bar.button(kHudControls.front()) == bar.buttons.front());
    REQUIRE(bar.button(HudControl::Close) == bar.buttons.back());
}

TEST_CASE("quit is kept away from close", "[overlay][hud]") {
    const auto bar = controlBar(500.0F, 20.0F, 24.0F, 6.0F);
    REQUIRE(bar.button(HudControl::Close).x - bar.button(HudControl::Quit).right() > 48.0F);
}

TEST_CASE("the corner HUD's controls and field sit inside its text column", "[overlay][hud]") {
    const auto geometry = panelGeometry(kPanel, 21.0F);
    const RectF& column = geometry.text;
    const auto layout = collapsedLayout(column, 1.0F, 1.5F);
    REQUIRE(layout.controls.bounds.right() == Approx(column.right()));
    REQUIRE(layout.controls.bounds.x > column.x);
    REQUIRE(inside(layout.input, column));
    REQUIRE(layout.input.bottom() == Approx(column.bottom()));
    REQUIRE(disjoint(layout.input, layout.controls.bounds));
    REQUIRE(disjoint(layout.confirmYes, layout.confirmNo));
    REQUIRE(inside(layout.confirmYes, column));
}

TEST_CASE("the expanded panel's parts tile its frame", "[overlay][hud]") {
    const auto layout = expandedLayout(kFrame, 1.0F);
    REQUIRE(inside(layout.body, kFrame));
    REQUIRE(inside(layout.input, kFrame));
    REQUIRE(inside(layout.controls.bounds, kFrame));
    REQUIRE(disjoint(layout.body, layout.input));
    REQUIRE(disjoint(layout.tabConversation, layout.tabSettings));
    REQUIRE(disjoint(layout.input, layout.grip));
    REQUIRE(layout.tabConversation.bottom() <= layout.body.y);
    REQUIRE(layout.ring.centreX - layout.ring.radius >= kFrame.x);
    REQUIRE(layout.grip.right() == Approx(kFrame.right()));
    REQUIRE(layout.grip.bottom() == Approx(kFrame.bottom()));
}

TEST_CASE("the expanded layout scales with the DPI", "[overlay][hud]") {
    const auto normal = expandedLayout(kFrame, 1.0F);
    const RectF big{
        .x = 0.0F, .y = 0.0F, .width = kFrame.width * 1.5F, .height = kFrame.height * 1.5F};
    const auto scaled = expandedLayout(big, 1.5F);
    REQUIRE(scaled.ring.radius == Approx(normal.ring.radius * 1.5F));
    REQUIRE(scaled.input.height == Approx(normal.input.height * 1.5F));
}

TEST_CASE("the corner HUD's hit map: body drags, controls click, the rest falls through",
          "[overlay][hud]") {
    const auto geometry = panelGeometry(kPanel, 21.0F);
    const auto layout = collapsedLayout(geometry.text, 1.0F, 1.5F);
    HitMap map;
    collapsedHitMap(map, geometry, geometry.text, layout, CollapsedHits{}, 28.0F);

    REQUIRE(map.test(geometry.centreX, geometry.centreY) == HitKind::Drag);
    const RectF mute = layout.controls.button(HudControl::Mute);
    REQUIRE(map.test(mute.x + 2.0F, mute.y + 2.0F) == HitKind::Control);
    REQUIRE(map.test(geometry.text.x + 5.0F, geometry.text.y + 60.0F) == HitKind::Drag);
    // The panel's empty corner outside the ring disc and the scrim.
    REQUIRE(map.test(kPanel.x + 2.0F, kPanel.y + 2.0F) == HitKind::None);
    REQUIRE(map.test(0.0F, 0.0F) == HitKind::None);
    // The text field is only clickable while it is open.
    const RectF& input = layout.input;
    REQUIRE(map.test(input.x + 4.0F, input.y + 4.0F) == HitKind::Drag);
    collapsedHitMap(map, geometry, geometry.text, layout, CollapsedHits{.inputOpen = true}, 28.0F);
    REQUIRE(map.test(input.x + 4.0F, input.y + 4.0F) == HitKind::Control);
}

TEST_CASE("the expanded hit map resizes at the grip and drags at the frame", "[overlay][hud]") {
    const auto layout = expandedLayout(kFrame, 1.0F);
    HitMap map;
    expandedHitMap(map, layout);
    REQUIRE(map.test(layout.grip.x + 2.0F, layout.grip.y + 2.0F) == HitKind::Resize);
    REQUIRE(map.test(layout.body.x + 10.0F, layout.body.y + 10.0F) == HitKind::Control);
    REQUIRE(map.test(kFrame.x + 2.0F, kFrame.y + 2.0F) == HitKind::Drag);
    REQUIRE(map.test(kFrame.x - 2.0F, kFrame.y) == HitKind::None);
}

TEST_CASE("approach moves at a frame-rate independent pace and never overshoots",
          "[overlay][hud]") {
    REQUIRE(approach(0.0F, 1.0F, 0.1F, 0.4F) == Approx(0.25F));
    REQUIRE(approach(0.9F, 1.0F, 0.5F, 0.4F) == Approx(1.0F));
    REQUIRE(approach(1.0F, 0.0F, 0.2F, 0.4F) == Approx(0.5F));
    REQUIRE(approach(0.3F, 0.7F, 0.0F, 0.4F) == Approx(0.3F));
    REQUIRE(approach(0.3F, 0.7F, 0.1F, 0.0F) == Approx(0.7F));
    // Four frames of 25 ms equal one of 100 ms.
    float stepped = 0.0F;
    for (int i = 0; i < 4; ++i) {
        stepped = approach(stepped, 1.0F, 0.025F, 0.4F);
    }
    REQUIRE(stepped == Approx(approach(0.0F, 1.0F, 0.1F, 0.4F)));
}
