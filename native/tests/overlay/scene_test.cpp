#include "jarvis/overlay/scene.hpp"

#include "jarvis/overlay/animation.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <vector>

using Catch::Matchers::WithinAbs;
namespace anim = jarvis::overlay::anim;

namespace {

constexpr float kTolerance = 1e-4F;
constexpr float kPi = std::numbers::pi_v<float>;

std::size_t countKind(const std::vector<anim::GlobeLine>& lines, anim::GlobeLineKind kind) {
    std::size_t count = 0;
    for (const anim::GlobeLine& line : lines) {
        count += line.kind == kind ? 1 : 0;
    }
    return count;
}

/// Every line is undrawn at 0 and complete at 1.
bool revealEnds(std::size_t lines) {
    for (std::size_t i = 0; i < lines; ++i) {
        if (anim::lineReveal(0.0F, i, lines) != 0.0F || anim::lineReveal(1.0F, i, lines) != 1.0F) {
            return false;
        }
    }
    return true;
}

/// Earlier lines are never behind later ones, and no line ever goes backwards.
bool revealOrdered(std::size_t lines) {
    for (int step = 0; step <= 100; ++step) {
        const float p = static_cast<float>(step) / 100.0F;
        for (std::size_t i = 0; i + 1 < lines; ++i) {
            const float now = anim::lineReveal(p, i, lines);
            if (now < anim::lineReveal(p, i + 1, lines) ||
                anim::lineReveal(p + 0.01F, i, lines) < now) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

// --- projection ------------------------------------------------------------------------

TEST_CASE("the origin projects onto the viewport centre", "[overlay][scene]") {
    const anim::Camera camera{.yaw = 0.7F, .pitch = -0.3F, .distance = 4.0F, .nearPlane = 0.1F};
    const anim::Viewport viewport{.centreX = 200.0F, .centreY = 120.0F, .focal = 300.0F};
    const anim::Projected origin = anim::project(camera, viewport, {});
    REQUIRE(origin.visible);
    REQUIRE_THAT(origin.x, WithinAbs(200.0, kTolerance));
    REQUIRE_THAT(origin.y, WithinAbs(120.0, kTolerance));
    REQUIRE_THAT(origin.depth, WithinAbs(4.0, kTolerance));
    REQUIRE_THAT(origin.scale, WithinAbs(75.0, kTolerance));
}

TEST_CASE("known points land where a pinhole camera puts them", "[overlay][scene]") {
    const anim::Camera camera{.yaw = 0.0F, .pitch = 0.0F, .distance = 4.0F, .nearPlane = 0.1F};
    const anim::Viewport viewport = anim::viewportFor(camera, 100.0F, 100.0F, 50.0F);
    REQUIRE(viewport.focal == 200.0F);

    // One unit right at the origin's depth lands exactly one radius right.
    const anim::Projected right =
        anim::project(camera, viewport, {.x = 1.0F, .y = 0.0F, .z = 0.0F});
    REQUIRE_THAT(right.x, WithinAbs(150.0, kTolerance));
    REQUIRE_THAT(right.y, WithinAbs(100.0, kTolerance));
    // World up is screen up (smaller y).
    const anim::Projected up = anim::project(camera, viewport, {.x = 0.0F, .y = 1.0F, .z = 0.0F});
    REQUIRE_THAT(up.y, WithinAbs(50.0, kTolerance));
    // Nearer points are bigger: z = -2 is at depth 2, twice the scale.
    const anim::Projected near =
        anim::project(camera, viewport, {.x = 1.0F, .y = 0.0F, .z = -2.0F});
    REQUIRE_THAT(near.depth, WithinAbs(2.0, kTolerance));
    REQUIRE_THAT(near.x, WithinAbs(200.0, kTolerance));
    REQUIRE(near.scale > right.scale);
}

TEST_CASE("points behind the eye are culled", "[overlay][scene]") {
    const anim::Camera camera{.yaw = 0.0F, .pitch = 0.0F, .distance = 4.0F, .nearPlane = 0.1F};
    const anim::Viewport viewport{.centreX = 0.0F, .centreY = 0.0F, .focal = 100.0F};
    REQUIRE_FALSE(anim::project(camera, viewport, {.x = 0.0F, .y = 0.0F, .z = -5.0F}).visible);
    REQUIRE_FALSE(anim::project(camera, viewport, {.x = 0.0F, .y = 0.0F, .z = -3.95F}).visible);
}

TEST_CASE("a camera above the equator sees the front of the globe below centre",
          "[overlay][scene]") {
    const anim::Camera camera{.yaw = 0.0F, .pitch = 0.4F, .distance = 4.0F, .nearPlane = 0.1F};
    const anim::Viewport viewport{.centreX = 0.0F, .centreY = 0.0F, .focal = 100.0F};
    const anim::Projected front =
        anim::project(camera, viewport, {.x = 0.0F, .y = 0.0F, .z = -1.0F});
    REQUIRE(front.y > 0.0F);
    // Turning the camera by yaw is the same as turning the world the other way.
    const anim::Camera turned{.yaw = 0.5F, .pitch = 0.0F, .distance = 4.0F, .nearPlane = 0.1F};
    const anim::Camera straight{.yaw = 0.0F, .pitch = 0.0F, .distance = 4.0F, .nearPlane = 0.1F};
    const anim::Vec3 point{.x = 0.3F, .y = 0.2F, .z = -0.6F};
    const anim::Projected a = anim::project(turned, viewport, point);
    const anim::Projected b = anim::project(straight, viewport, anim::rotateY(point, -0.5F));
    REQUIRE_THAT(a.x, WithinAbs(b.x, kTolerance));
    REQUIRE_THAT(a.y, WithinAbs(b.y, kTolerance));
}

TEST_CASE("rotations keep lengths and turn the axes the right way", "[overlay][scene]") {
    const anim::Vec3 x{.x = 1.0F, .y = 0.0F, .z = 0.0F};
    const anim::Vec3 turned = anim::rotateZ(x, kPi / 2.0F);
    REQUIRE_THAT(turned.y, WithinAbs(1.0, kTolerance));
    REQUIRE_THAT(anim::rotateY(x, kPi / 2.0F).z, WithinAbs(-1.0, kTolerance));
    REQUIRE_THAT(anim::rotateX({.x = 0.0F, .y = 1.0F, .z = 0.0F}, kPi / 2.0F).z,
                 WithinAbs(1.0, kTolerance));
    const anim::Vec3 any{.x = 0.3F, .y = -0.7F, .z = 0.2F};
    REQUIRE_THAT(anim::length(anim::rotateX(anim::rotateY(anim::rotateZ(any, 1.0F), 2.0F), 3.0F)),
                 WithinAbs(anim::length(any), kTolerance));
}

TEST_CASE("the drifting camera stays near its rest pose and holds still without motion",
          "[overlay][scene]") {
    const anim::Camera rest = anim::driftingCamera(0.0F, 0.0F);
    for (int i = 0; i < 200; ++i) {
        const float time = static_cast<float>(i) * 0.37F;
        const anim::Camera camera = anim::driftingCamera(time, 1.0F);
        REQUIRE(std::abs(camera.yaw - rest.yaw) <= 0.23F);
        REQUIRE(std::abs(camera.pitch - rest.pitch) <= 0.19F);  // two swings
        REQUIRE(anim::driftingCamera(time, 0.0F).yaw == rest.yaw);
    }
}

TEST_CASE("parallax is zero at rest and grows with depth", "[overlay][scene]") {
    const anim::Camera rest = anim::driftingCamera(0.0F, 0.0F);
    REQUIRE(anim::parallax(rest, 1.0F).x == 0.0F);
    anim::Camera drifted = rest;
    drifted.yaw += 0.2F;
    REQUIRE(anim::parallax(drifted, 0.0F).x == 0.0F);
    const float near = anim::parallax(drifted, 1.0F).x;
    const float far = anim::parallax(drifted, -1.0F).x;
    REQUIRE(near > 0.0F);
    REQUIRE_THAT(far, WithinAbs(-near, kTolerance));
    REQUIRE(std::abs(anim::parallax(drifted, 0.5F).x) < std::abs(near));
}

// --- globe ---------------------------------------------------------------------------

TEST_CASE("the globe has the line counts its spec asks for", "[overlay][scene][globe]") {
    const anim::GlobeSpec spec{
        .latitudes = 7, .meridians = 12, .segments = 48, .orbits = 2, .orbitRadius = 1.3F};
    const std::vector<anim::GlobeLine> lines = anim::buildGlobe(spec);
    REQUIRE(lines.size() == 21);
    REQUIRE(countKind(lines, anim::GlobeLineKind::Latitude) == 7);
    REQUIRE(countKind(lines, anim::GlobeLineKind::Meridian) == 12);
    REQUIRE(countKind(lines, anim::GlobeLineKind::Orbit) == 2);
    REQUIRE(lines.front().kind == anim::GlobeLineKind::Latitude);
    REQUIRE(lines.back().kind == anim::GlobeLineKind::Orbit);
}

TEST_CASE("meridians are open half circles, the rest closed circles", "[overlay][scene][globe]") {
    const auto shaped = [](const anim::GlobeLine& line) {
        const bool meridian = line.kind == anim::GlobeLineKind::Meridian;
        return line.points.size() == (meridian ? 25U : 48U) && line.closed == !meridian;
    };
    REQUIRE(std::ranges::all_of(anim::buildGlobe({}), shaped));
}

TEST_CASE("an empty globe spec builds no lines", "[overlay][scene][globe]") {
    REQUIRE(anim::buildGlobe({.latitudes = 0, .meridians = 0, .segments = 8, .orbits = 0}).empty());
}

TEST_CASE("globe points lie on the unit sphere and orbits on their radius",
          "[overlay][scene][globe]") {
    const anim::GlobeSpec spec{};
    REQUIRE(std::ranges::all_of(anim::buildGlobe(spec), [&spec](const anim::GlobeLine& line) {
        const float expected = line.kind == anim::GlobeLineKind::Orbit ? spec.orbitRadius : 1.0F;
        return std::ranges::all_of(line.points, [expected](const anim::Vec3& point) {
            return std::abs(anim::length(point) - expected) <= kTolerance;
        });
    }));
}

TEST_CASE("meridians run from pole to pole", "[overlay][scene][globe]") {
    REQUIRE(std::ranges::all_of(anim::buildGlobe({}), [](const anim::GlobeLine& line) {
        return line.kind != anim::GlobeLineKind::Meridian ||
               (std::abs(line.points.front().y + 1.0F) <= kTolerance &&
                std::abs(line.points.back().y - 1.0F) <= kTolerance);
    }));
}

TEST_CASE("the globe draws itself line by line", "[overlay][scene][globe]") {
    constexpr std::size_t kLines = 21;
    REQUIRE(revealEnds(kLines));
    REQUIRE(revealOrdered(kLines));
    // Half way through, the first line is done and the last has not started.
    REQUIRE(anim::lineReveal(0.5F, 0, kLines) == 1.0F);
    REQUIRE(anim::lineReveal(0.5F, kLines - 1, kLines) == 0.0F);
    REQUIRE(anim::lineReveal(0.3F, 0, 1) == 0.3F);
}

// --- orbits, rings, rotors -------------------------------------------------------------------

TEST_CASE("satellite orbits are circles of their radius", "[overlay][scene]") {
    REQUIRE(anim::satelliteOrbits().size() == 4);
    REQUIRE(std::ranges::all_of(anim::satelliteOrbits(), [](const anim::Orbit& orbit) {
        return std::ranges::all_of(std::array{0.0F, 1.0F, 2.5F, 4.0F}, [&orbit](float angle) {
            return std::abs(anim::length(anim::orbitPosition(orbit, angle)) - orbit.radius) <=
                   kTolerance;
        });
    }));
}

TEST_CASE("ring layers go from the outside in and turn both ways", "[overlay][scene][rings]") {
    const auto layers = anim::ringLayers();
    REQUIRE(layers.size() == anim::kRingCount);
    REQUIRE(std::ranges::is_sorted(layers, std::ranges::greater{}, &anim::RingLayer::radius));
    REQUIRE(std::ranges::all_of(layers, [](const anim::RingLayer& layer) {
        return layer.alpha > 0.0F && layer.alpha <= 1.0F;
    }));
    REQUIRE(std::ranges::any_of(layers,
                                [](const anim::RingLayer& layer) { return layer.speed > 0.0F; }));
    REQUIRE(std::ranges::any_of(layers,
                                [](const anim::RingLayer& layer) { return layer.speed < 0.0F; }));
}

TEST_CASE("tick labels count up to their degrees", "[overlay][scene][rings]") {
    REQUIRE(anim::tickLabel(0, 12, 1.0F) == 0);
    REQUIRE(anim::tickLabel(3, 12, 1.0F) == 90);
    REQUIRE(anim::tickLabel(11, 12, 1.0F) == 330);
    REQUIRE(anim::tickLabel(11, 12, 0.0F) == 0);
    REQUIRE(anim::tickLabel(6, 12, 0.5F) == 90);
    REQUIRE(anim::tickLabel(1, 0, 1.0F) == 0);
}

TEST_CASE("thinking spins faster than idle", "[overlay][scene][rings]") {
    const std::array<float, 4> idle{1.0F, 0.0F, 0.0F, 0.0F};
    const std::array<float, 4> thinking{0.0F, 0.0F, 1.0F, 0.0F};
    const std::array<float, 4> half{0.5F, 0.0F, 0.5F, 0.0F};
    REQUIRE(anim::stateSpeedFactor(thinking) > anim::stateSpeedFactor(idle));
    REQUIRE_THAT(anim::stateSpeedFactor(half),
                 WithinAbs((anim::stateSpeedFactor(idle) + anim::stateSpeedFactor(thinking)) / 2.0F,
                           kTolerance));
}

TEST_CASE("a rotor eases its speed and is frame-rate independent", "[overlay][scene][rings]") {
    anim::Rotor once{0.5F};
    anim::Rotor many{0.5F};
    once.advance(2.0F, 1.0F);
    for (int i = 0; i < 60; ++i) {
        many.advance(2.0F, 1.0F / 60.0F);
    }
    REQUIRE_THAT(once.angle(), WithinAbs(many.angle(), 1e-3));
    REQUIRE_THAT(once.speed(), WithinAbs(many.speed(), 1e-4));
    REQUIRE(once.speed() < 2.0F);  // still easing in
    REQUIRE(once.speed() > 1.5F);

    anim::Rotor instant{0.0F};
    instant.advance(1.0F, 0.5F);
    REQUIRE_THAT(instant.angle(), WithinAbs(0.5, kTolerance));
    instant.advance(3.0F, 0.0F);  // zero dt changes nothing
    REQUIRE(instant.speed() == 1.0F);
}

TEST_CASE("rotor angles stay wrapped after a day of uptime", "[overlay][scene][rings]") {
    anim::Rotor rotor{0.6F, 0.0F, 3.0F};
    for (int i = 0; i < 24 * 60; ++i) {
        rotor.advance(3.0F, 60.0F);
    }
    REQUIRE(rotor.angle() >= 0.0F);
    REQUIRE(rotor.angle() < anim::kTau);
}

TEST_CASE("the projector matches rotating then projecting point by point", "[overlay][scene]") {
    const anim::Camera camera = anim::driftingCamera(3.7F, 1.0F);
    const anim::Viewport viewport = anim::viewportFor(camera, 320.0F, 200.0F, 90.0F);
    const anim::Projector projector{camera, viewport, 1.3F, 0.38F};
    const auto same = [&](const anim::Vec3& point) {
        const anim::Projected fast = projector(point);
        const anim::Projected slow =
            anim::project(camera, viewport, anim::rotateZ(anim::rotateY(point, 1.3F), 0.38F));
        return fast.visible == slow.visible && std::abs(fast.x - slow.x) <= 1e-3F &&
               std::abs(fast.y - slow.y) <= 1e-3F && std::abs(fast.depth - slow.depth) <= 1e-4F;
    };
    REQUIRE(std::ranges::all_of(anim::buildGlobe({}), [&same](const anim::GlobeLine& line) {
        return std::ranges::all_of(line.points, same);
    }));
}
