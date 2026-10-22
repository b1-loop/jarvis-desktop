#include "jarvis/overlay/particles.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

using Catch::Matchers::WithinAbs;
namespace anim = jarvis::overlay::anim;

namespace {

constexpr std::uint64_t kSeed = 0xC0FFEEULL;
constexpr std::size_t kCount = 300;

float meanRadius(std::span<const anim::SimParticle> particles) {
    float sum = 0.0F;
    for (const anim::SimParticle& p : particles) {
        sum += std::hypot(p.x, p.y);
    }
    return sum / static_cast<float>(particles.size());
}

/// Run @p system for @p seconds in frames of @p dt under @p forces.
void run(anim::ParticleSystem& system, float seconds, float dt, const anim::FieldForces& forces) {
    const int frames = static_cast<int>(std::lround(seconds / dt));
    for (int i = 0; i < frames; ++i) {
        system.advance(dt, forces);
    }
}

bool sameField(const anim::ParticleSystem& a, const anim::ParticleSystem& b, float tolerance) {
    return std::ranges::equal(
        a.particles(), b.particles(), [tolerance](const auto& p, const auto& q) {
            return std::abs(p.x - q.x) <= tolerance && std::abs(p.y - q.y) <= tolerance;
        });
}

}  // namespace

TEST_CASE("curl noise is divergence-free and bounded", "[overlay][particles]") {
    constexpr float kStep = 1e-3F;
    float worst = 0.0F;
    for (int i = 0; i < 40; ++i) {
        const float x = -2.0F + (0.1F * static_cast<float>(i));
        const float y = 1.5F - (0.07F * static_cast<float>(i));
        const float t = 0.3F * static_cast<float>(i);
        const anim::Vec2f here = anim::curlNoise(x, y, t);
        REQUIRE(std::hypot(here.x, here.y) <= 1.0F + 1e-5F);
        const float dudx =
            (anim::curlNoise(x + kStep, y, t).x - anim::curlNoise(x - kStep, y, t).x) /
            (2.0F * kStep);
        const float dvdy =
            (anim::curlNoise(x, y + kStep, t).y - anim::curlNoise(x, y - kStep, t).y) /
            (2.0F * kStep);
        worst = std::max(worst, std::abs(dudx + dvdy));
    }
    REQUIRE(worst < 5e-3F);  // zero up to the finite-difference error
}

TEST_CASE("curl noise changes over space and time", "[overlay][particles]") {
    const anim::Vec2f a = anim::curlNoise(0.2F, 0.4F, 0.0F);
    const anim::Vec2f b = anim::curlNoise(0.9F, 0.4F, 0.0F);
    const anim::Vec2f c = anim::curlNoise(0.2F, 0.4F, 3.0F);
    REQUIRE((a.x != b.x || a.y != b.y));
    REQUIRE((a.x != c.x || a.y != c.y));
}

TEST_CASE("a particle system is deterministic for a fixed seed", "[overlay][particles]") {
    anim::ParticleSystem first{kSeed, kCount};
    anim::ParticleSystem second{kSeed, kCount};
    anim::ParticleSystem other{kSeed + 1, kCount};
    REQUIRE(first.particles().size() == kCount);
    const anim::FieldForces forces{.breathe = 0.3F, .swirl = 0.5F};
    run(first, 1.5F, 1.0F / 60.0F, forces);
    run(second, 1.5F, 1.0F / 60.0F, forces);
    run(other, 1.5F, 1.0F / 60.0F, forces);
    REQUIRE(sameField(first, second, 0.0F));
    REQUIRE_FALSE(sameField(first, other, 1e-3F));
}

TEST_CASE("the field does not depend on the frame rate", "[overlay][particles]") {
    const anim::FieldForces forces{.breathe = 0.5F, .pull = 0.2F, .noise = 1.0F};
    anim::ParticleSystem at60{kSeed, kCount};
    anim::ParticleSystem at15{kSeed, kCount};
    anim::ParticleSystem uneven{kSeed, kCount};
    run(at60, 2.0F, 1.0F / 60.0F, forces);
    run(at15, 2.0F, 1.0F / 15.0F, forces);
    // Every interval is within maxStepsPerAdvance (0.25 s), so nothing is skipped.
    for (const float dt :
         {0.2F, 0.011F, 0.25F, 0.1F, 0.039F, 0.2F, 0.2F, 0.25F, 0.25F, 0.25F, 0.25F}) {
        uneven.advance(dt, forces);
    }
    REQUIRE(at60.steps() == 120);
    REQUIRE(at15.steps() == at60.steps());
    REQUIRE(uneven.steps() == at60.steps());
    REQUIRE(sameField(at60, at15, 1e-4F));
    REQUIRE(sameField(at60, uneven, 1e-4F));
}

TEST_CASE("a long stall skips time instead of catching up", "[overlay][particles]") {
    anim::ParticleSystem system{kSeed, kCount};
    system.advance(10.0F, {});
    REQUIRE(system.steps() == 600);  // 10 s at 60 steps per second
    // Only maxStepsPerAdvance steps actually ran; the field is still calm.
    REQUIRE(meanRadius(system.particles()) < system.config().maxRadius);
}

TEST_CASE("listening pulls the field in and speaking bursts it out", "[overlay][particles]") {
    anim::ParticleSystem calm{kSeed, kCount};
    anim::ParticleSystem listening{kSeed, kCount};
    anim::ParticleSystem speaking{kSeed, kCount};
    const float dt = 1.0F / 60.0F;
    run(calm, 1.0F, dt, {});
    run(listening, 1.0F, dt, {.pull = 1.0F});
    run(speaking, 0.4F, dt, {.burst = 1.0F});
    const float rest = meanRadius(calm.particles());
    REQUIRE(meanRadius(listening.particles()) < rest * 0.8F);
    REQUIRE(meanRadius(speaking.particles()) > rest * 1.1F);
}

TEST_CASE("particles stay bounded and alphas stay in range under any forces",
          "[overlay][particles]") {
    anim::ParticleSystem system{kSeed, kCount};
    const anim::FieldForces wild{
        .breathe = 1.0F, .pull = 1.0F, .burst = 1.0F, .swirl = 1.0F, .noise = 2.0F, .motion = 2.0F};
    for (int i = 0; i < 600; ++i) {
        system.advance(1.0F / 60.0F, wild);
    }
    const float limit = system.config().maxRadius * 1.5F;
    REQUIRE(std::ranges::all_of(system.particles(), [&system, limit](const anim::SimParticle& p) {
        const float alpha = system.alpha(p);
        return std::isfinite(p.x) && std::hypot(p.x, p.y) < limit && alpha >= 0.0F && alpha <= 1.0F;
    }));
}

TEST_CASE("a zero or negative interval changes nothing", "[overlay][particles]") {
    anim::ParticleSystem system{kSeed, 10};
    const float x = system.particles().front().x;
    system.advance(0.0F, {});
    system.advance(-1.0F, {});
    REQUIRE(system.steps() == 0);
    REQUIRE_THAT(system.particles().front().x, WithinAbs(x, 0.0));
}

TEST_CASE("only the active share of the field moves", "[overlay][particles]") {
    anim::ParticleSystem system{kSeed, kCount};
    const float lastX = system.particles().back().x;
    const float firstX = system.particles().front().x;
    for (int i = 0; i < 30; ++i) {
        system.advance(1.0F / 60.0F, {}, kCount / 2);
    }
    REQUIRE(system.particles().back().x == lastX);
    REQUIRE(system.particles().front().x != firstX);
}
