#include "jarvis/overlay/pacing.hpp"

#include "jarvis/overlay/particles.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <optional>

using Catch::Matchers::WithinAbs;
namespace pacing = jarvis::overlay::pacing;
namespace anim = jarvis::overlay::anim;

namespace {

constexpr double kRefresh240 = 1.0 / 240.0;
constexpr double kTolerance = 1e-9;

/// Whole periods between @p a and @p b on a grid of @p period.
double periodsBetween(double a, double b, double period) {
    return (b - a) / period;
}

}  // namespace

TEST_CASE("frames get a whole number of refreshes", "[overlay][pacing]") {
    REQUIRE(pacing::refreshesPerFrame(60.0, 60) == 1);
    REQUIRE(pacing::refreshesPerFrame(144.0, 60) == 2);
    REQUIRE(pacing::refreshesPerFrame(240.0, 60) == 4);
    REQUIRE(pacing::refreshesPerFrame(60.0, 30) == 2);
    REQUIRE(pacing::refreshesPerFrame(144.0, 30) == 5);
    REQUIRE(pacing::refreshesPerFrame(240.0, 30) == 8);
    REQUIRE(pacing::refreshesPerFrame(59.94, 60) == 1);
    REQUIRE(pacing::refreshesPerFrame(30.0, 60) == 1);  // never less than one
    REQUIRE(pacing::refreshesPerFrame(0.0, 60) == 1);
    REQUIRE(pacing::refreshesPerFrame(60.0, 0) == 1);
}

TEST_CASE("slots follow the previous one by whole refreshes", "[overlay][pacing]") {
    const pacing::Grid grid{.vblank = 10.0, .period = kRefresh240};
    const double slot = pacing::nextSlot(grid, 10.0, 4, 10.001);
    REQUIRE_THAT(periodsBetween(10.0, slot, kRefresh240), WithinAbs(4.0, 1e-6));
    // Wake a little after the vblank, never before it.
    const double wake = pacing::wakeTime(grid, slot);
    REQUIRE(wake > slot);
    REQUIRE(wake < slot + (kRefresh240 * 0.5));
}

TEST_CASE("slots are re-snapped onto a grid that drifted", "[overlay][pacing]") {
    // The previous slot was planned on a grid measured 0.3 ms earlier.
    const pacing::Grid grid{.vblank = 20.0003, .period = kRefresh240};
    const double slot = pacing::nextSlot(grid, 20.0, 4, 20.001);
    const double offGrid = std::remainder(slot - grid.vblank, kRefresh240);
    REQUIRE_THAT(offGrid, WithinAbs(0.0, kTolerance));
    REQUIRE_THAT(periodsBetween(grid.vblank, slot, kRefresh240), WithinAbs(4.0, 1e-6));
}

TEST_CASE("a late loop resynchronises to the next slot instead of catching up",
          "[overlay][pacing]") {
    const pacing::Grid grid{.vblank = 5.0, .period = 1.0 / 60.0};
    // A 200 ms stall: the planned slot is long gone.
    const double now = 5.2;
    const double slot = pacing::nextSlot(grid, 5.0, 1, now);
    REQUIRE(pacing::wakeTime(grid, slot) >= now);
    REQUIRE(pacing::wakeTime(grid, slot) < now + grid.period);
    const double offGrid = std::remainder(slot - grid.vblank, grid.period);
    REQUIRE_THAT(offGrid, WithinAbs(0.0, 1e-9));
}

TEST_CASE("without a previous slot the first slot ahead is taken", "[overlay][pacing]") {
    const pacing::Grid grid{.vblank = 1.0, .period = 0.01};
    const double slot = pacing::nextSlot(grid, std::nullopt, 3, 1.0505);
    REQUIRE_THAT(slot, WithinAbs(1.05, kTolerance));  // its wake time is 1.0515
    const double later = pacing::nextSlot(grid, std::nullopt, 3, 1.0517);
    REQUIRE_THAT(later, WithinAbs(1.06, kTolerance));
    // No grid: draw now.
    REQUIRE(pacing::nextSlot(pacing::Grid{}, std::nullopt, 1, 3.0) == 3.0);
}

TEST_CASE("particles are drawn between their last two fixed steps", "[overlay][particles]") {
    anim::ParticleSystem field{7, 16};
    const anim::FieldForces forces{};
    field.advance(1.0F / 60.0F, forces);
    REQUIRE_THAT(field.blend(), WithinAbs(0.0, 1e-3));
    // Half a step later nothing is simulated, but the drawing moves on.
    field.advance(0.5F / 60.0F, forces);
    REQUIRE_THAT(field.blend(), WithinAbs(0.5, 1e-3));
    const anim::SimParticle& p = field.particles().front();
    const anim::Vec2f drawn = field.position(p);
    REQUIRE_THAT(drawn.x, WithinAbs((p.prevX + p.x) * 0.5F, 1e-5));
    REQUIRE_THAT(drawn.y, WithinAbs((p.prevY + p.y) * 0.5F, 1e-5));
}
