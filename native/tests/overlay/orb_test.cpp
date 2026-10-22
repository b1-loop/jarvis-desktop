#include "jarvis/overlay/orb.hpp"

#include "jarvis/overlay/animator.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/placement.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <vector>

using Catch::Approx;
using jarvis::overlay::DockEdge;
using jarvis::overlay::dockOrb;
using jarvis::overlay::kOrbDiameter;
using jarvis::overlay::kOrbMargin;
using jarvis::overlay::MonitorArea;
using jarvis::overlay::monitorFor;
using jarvis::overlay::nearestEdge;
using jarvis::overlay::orbLook;
using jarvis::overlay::OrbMorph;
using jarvis::overlay::orbMorphAt;
using jarvis::overlay::orbSide;
using jarvis::overlay::PixelRect;
using jarvis::overlay::Placement;
using jarvis::overlay::restorePlacement;
using jarvis::overlay::SavedPlacement;
using jarvis::overlay::squareAt;
using jarvis::overlay::anim::AnimationFrame;

namespace {

constexpr PixelRect kWork{.left = 0, .top = 0, .right = 1920, .bottom = 1040};
MonitorArea primary() {
    return MonitorArea{.name = "DISPLAY1", .work = kWork, .scale = 1.0F, .primary = true};
}

/// A 760 x 300 HUD with its ring centre 150 px in from its left edge.
Placement hudAt(int x, int y) {
    return Placement{.x = x, .y = y, .width = 760, .height = 300};
}

float ringX(const Placement& hud) {
    return static_cast<float>(hud.x) + 150.0F;
}

float ringY(const Placement& hud) {
    return static_cast<float>(hud.y) + 150.0F;
}

Placement dock(const Placement& hud, const MonitorArea& monitor = primary()) {
    return dockOrb(hud, ringX(hud), ringY(hud), monitor);
}

AnimationFrame frameFor(int state, float level = 0.0F, float breathing = 0.0F) {
    AnimationFrame frame;
    frame.weights = {0.0F, 0.0F, 0.0F, 0.0F};
    frame.weights.at(static_cast<std::size_t>(state)) = 1.0F;
    frame.energy.level = level;
    frame.energy.peak = level;
    frame.breathing = breathing;
    return frame;
}

constexpr int kIdle = 0;
constexpr int kListening = 1;
constexpr int kThinking = 2;
constexpr int kSpeaking = 3;

}  // namespace

TEST_CASE("the nearest work-area edge is found for each side", "[overlay][orb]") {
    REQUIRE(nearestEdge(hudAt(30, 400), kWork) == DockEdge::Left);
    REQUIRE(nearestEdge(hudAt(1140, 380), kWork) == DockEdge::Right);
    REQUIRE(nearestEdge(hudAt(600, 20), kWork) == DockEdge::Top);
    REQUIRE(nearestEdge(hudAt(600, 720), kWork) == DockEdge::Bottom);
    // Equal distances: left wins over right, top over bottom.
    REQUIRE(nearestEdge(Placement{.x = 100, .y = 300, .width = 1720, .height = 300}, kWork) ==
            DockEdge::Left);
    REQUIRE(nearestEdge(Placement{.x = 300, .y = 100, .width = 760, .height = 840}, kWork) ==
            DockEdge::Top);
}

TEST_CASE("the orb docks at the nearest edge, centred on the ring", "[overlay][orb]") {
    const Placement left = dock(hudAt(60, 400));
    REQUIRE(left == Placement{.x = kOrbMargin,
                              .y = 550 - (kOrbDiameter / 2),
                              .width = kOrbDiameter,
                              .height = kOrbDiameter});
    const Placement bottom = dock(hudAt(600, 700));
    REQUIRE(bottom.y == 1040 - kOrbMargin - kOrbDiameter);
    REQUIRE(bottom.x == 750 - (kOrbDiameter / 2));
    const Placement top = dock(hudAt(600, 90));
    REQUIRE(top.y == kOrbMargin);
}

TEST_CASE("a HUD in a corner leaves its orb in that corner", "[overlay][orb]") {
    // The default bottom-right placement (24 px margin) counts as the corner.
    const Placement corner = dock(hudAt(1920 - 24 - 760, 1040 - 24 - 300));
    REQUIRE(corner == Placement{.x = 1920 - kOrbMargin - kOrbDiameter,
                                .y = 1040 - kOrbMargin - kOrbDiameter,
                                .width = kOrbDiameter,
                                .height = kOrbDiameter});
    const Placement snappedTopLeft = dock(hudAt(0, 0));
    REQUIRE(snappedTopLeft.x == kOrbMargin);
    REQUIRE(snappedTopLeft.y == kOrbMargin);
    // Close to one edge but far from the other: an edge, not a corner.
    const Placement edge = dock(hudAt(0, 300));
    REQUIRE(edge.x == kOrbMargin);
    REQUIRE(edge.y == 450 - (kOrbDiameter / 2));
}

TEST_CASE("the orb is DPI-scaled with its monitor", "[overlay][orb]") {
    REQUIRE(orbSide(1.0F) == kOrbDiameter);
    REQUIRE(orbSide(1.5F) == 72);
    REQUIRE(orbSide(2.0F) == 96);
    const MonitorArea hiDpi{.name = "D", .work = kWork, .scale = 1.5F, .primary = true};
    const Placement orb = dock(hudAt(0, 0), hiDpi);
    REQUIRE(orb.width == 72);
    REQUIRE(orb.height == 72);
    REQUIRE(orb.x == 18);  // the margin scales too
}

TEST_CASE("the orb stays fully on a monitor with negative coordinates", "[overlay][orb]") {
    // A monitor left of the primary: its work area starts at x = -2560.
    const MonitorArea leftOf{
        .name = "DISPLAY2",
        .work = PixelRect{.left = -2560, .top = -200, .right = 0, .bottom = 1240},
        .scale = 1.25F,
        .primary = false};
    // A HUD flush with the monitor's bottom edge docks its orb there, with
    // the monitor's own DPI and margin, in negative coordinates.
    const Placement hud{.x = -1500, .y = 1240 - 160, .width = 760, .height = 160};
    const Placement orb = dockOrb(hud, -1350.0F, 1235.0F, leftOf);
    REQUIRE(orb.width == 60);
    REQUIRE(orb.x >= leftOf.work.left);
    REQUIRE(orb.y + orb.height <= leftOf.work.bottom);
    REQUIRE(orb.y == 1240 - 15 - 60);
}

TEST_CASE("a remembered orb on an unplugged monitor lands on the primary", "[overlay][orb]") {
    const std::vector<MonitorArea> monitors{primary()};
    const SavedPlacement saved{.monitor = "DISPLAY9", .x = 5000.0F, .y = 20.0F};
    const Placement orb = restorePlacement(saved, monitors, kOrbDiameter, kOrbDiameter);
    REQUIRE(orb ==
            Placement{
                .x = 1920 - kOrbDiameter, .y = 20, .width = kOrbDiameter, .height = kOrbDiameter});
    REQUIRE(monitorFor(orb, monitors) == 0);
}

TEST_CASE("squareAt centres a square on a point", "[overlay][orb]") {
    REQUIRE(squareAt(100.0F, 50.0F, 40) == Placement{.x = 80, .y = 30, .width = 40, .height = 40});
}

TEST_CASE("the morph starts as the HUD and ends as the orb", "[overlay][orb][morph]") {
    REQUIRE(orbMorphAt(0.0F) ==
            OrbMorph{.travel = 0.0F, .panelAlpha = 1.0F, .sceneAlpha = 1.0F, .orbAlpha = 0.0F});
    REQUIRE(orbMorphAt(1.0F) ==
            OrbMorph{.travel = 1.0F, .panelAlpha = 0.0F, .sceneAlpha = 0.0F, .orbAlpha = 1.0F});
    REQUIRE(orbMorphAt(-1.0F) == orbMorphAt(0.0F));
    REQUIRE(orbMorphAt(2.0F) == orbMorphAt(1.0F));
}

TEST_CASE("the panel is gone before the ring has travelled far", "[overlay][orb][morph]") {
    const OrbMorph third = orbMorphAt(0.35F);
    REQUIRE(third.panelAlpha == Approx(0.0F));
    REQUIRE(third.travel < 0.25F);
    REQUIRE(third.sceneAlpha == Approx(1.0F));
    REQUIRE(third.orbAlpha == Approx(0.0F));
}

TEST_CASE("something glows on every frame of the morph", "[overlay][orb][morph]") {
    float previousTravel = -1.0F;
    for (int step = 0; step <= 100; ++step) {
        const OrbMorph morph = orbMorphAt(static_cast<float>(step) / 100.0F);
        // The ring or the orb (or both) is always visible: no empty frame,
        // and no frame where only the panel's rectangle would show.
        REQUIRE(morph.sceneAlpha + morph.orbAlpha >= 0.99F);
        REQUIRE(morph.travel >= previousTravel);
        previousTravel = morph.travel;
    }
}

TEST_CASE("the idle orb breathes slowly and dimly", "[overlay][orb][look]") {
    const auto exhaled = orbLook(frameFor(kIdle, 0.0F, 0.0F));
    const auto inhaled = orbLook(frameFor(kIdle, 0.0F, 1.0F));
    REQUIRE(inhaled.core > exhaled.core);
    REQUIRE(inhaled.pulse > exhaled.pulse);
    REQUIRE(inhaled.core < 0.7F);
    REQUIRE(exhaled.orbit == Approx(0.0F));
}

TEST_CASE("listening and speaking pulse with the audio level", "[overlay][orb][look]") {
    for (const int state : {kListening, kSpeaking}) {
        const auto quiet = orbLook(frameFor(state, 0.0F));
        const auto loud = orbLook(frameFor(state, 1.0F));
        REQUIRE(loud.pulse > quiet.pulse + 0.25F);
        REQUIRE(loud.core >= quiet.core);
    }
    REQUIRE(orbLook(frameFor(kListening, 0.5F)).core > orbLook(frameFor(kIdle, 0.0F, 1.0F)).core);
}

TEST_CASE("thinking orbits, and the orbit advances with time", "[overlay][orb][look]") {
    AnimationFrame early = frameFor(kThinking);
    early.time = 0.1F;
    AnimationFrame later = early;
    later.time = 0.4F;
    REQUIRE(orbLook(early).orbit == Approx(1.0F));
    REQUIRE(orbLook(later).orbitAngle > orbLook(early).orbitAngle);
}

TEST_CASE("the wake ripple passes through to the orb", "[overlay][orb][look]") {
    AnimationFrame frame = frameFor(kListening);
    frame.ripple = {.radius = 1.3F, .alpha = 0.6F};
    const auto look = orbLook(frame);
    REQUIRE(look.rippleRadius == Approx(1.3F));
    REQUIRE(look.rippleAlpha == Approx(0.6F));
}

TEST_CASE("a state cross-fade blends the two looks", "[overlay][orb][look]") {
    AnimationFrame frame = frameFor(kIdle);
    frame.weights = {0.5F, 0.5F, 0.0F, 0.0F};
    const float blended = orbLook(frame).core;
    REQUIRE(blended > orbLook(frameFor(kIdle)).core);
    REQUIRE(blended < orbLook(frameFor(kListening)).core);
}
