#include "jarvis/overlay/animation.hpp"

#include "jarvis/overlay/animator.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/scene.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

using Catch::Matchers::WithinAbs;
using jarvis::overlay::AssistantState;
using jarvis::overlay::OverlaySnapshot;
namespace anim = jarvis::overlay::anim;

namespace {

constexpr float kTolerance = 1e-5F;
constexpr int kSamples = 1000;

using Clock = anim::OverlayAnimator::Clock;

Clock::time_point at(double seconds) {
    return Clock::time_point{
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds))};
}

float sum(const anim::StateWeights& weights) {
    return std::accumulate(weights.begin(), weights.end(), 0.0F);
}

float radiusOf(const anim::ParticlePoint& point) {
    return std::hypot(point.x, point.y);
}

bool near(float actual, float expected, float tolerance) {
    return std::abs(actual - expected) <= tolerance;
}

bool samePositions(const std::vector<anim::ParticlePoint>& a,
                   const std::vector<anim::ParticlePoint>& b) {
    return std::ranges::equal(a, b, [](const auto& p, const auto& q) {
        return p.x == q.x && p.y == q.y && p.alpha == q.alpha;
    });
}

/// True when every particle lies between the ring and the farthest start
/// and has a valid alpha.
bool inFlight(const std::vector<anim::ParticlePoint>& points, const anim::ParticleConfig& config) {
    constexpr float kSlack = 1e-4F;
    return std::ranges::all_of(points, [&config](const anim::ParticlePoint& point) {
        const float radius = radiusOf(point);
        return radius >= config.targetRadius - kSlack && radius <= config.maxStartRadius + kSlack &&
               point.alpha >= 0.0F && point.alpha <= 1.0F;
    });
}

/// True when every progress of @p frame is at least that of @p previous.
bool movesForward(const anim::BootFrame& previous, const anim::BootFrame& frame) {
    bool rings = true;
    for (std::size_t i = 0; i < frame.ringArrival.size(); ++i) {
        rings = rings && frame.ringArrival.at(i) >= previous.ringArrival.at(i);
    }
    return rings && frame.ignite >= previous.ignite && frame.handoff >= previous.handoff &&
           frame.spin >= previous.spin && frame.titleChars >= previous.titleChars &&
           frame.subtitle >= previous.subtitle && frame.bootOnly <= previous.bootOnly &&
           frame.globe >= previous.globe && frame.labels >= previous.labels &&
           frame.act >= previous.act && near(frame.handoff + frame.bootOnly, 1.0F, kTolerance);
}

/// True when the blend's weights sum to 1 at every sample over [from, from + span].
bool sumsToOne(const anim::StateBlend& blend, float from, float span) {
    constexpr int kSteps = 20;
    for (int i = 0; i <= kSteps; ++i) {
        const float now = from + (span * static_cast<float>(i) / kSteps);
        if (!near(sum(blend.weights(now)), 1.0F, kTolerance)) {
            return false;
        }
    }
    return true;
}

void requireWeights(const anim::StateWeights& actual, const anim::StateWeights& expected) {
    for (std::size_t i = 0; i < actual.size(); ++i) {
        REQUIRE_THAT(actual.at(i), WithinAbs(expected.at(i), kTolerance));
    }
}

}  // namespace

// --- easing -------------------------------------------------------------------------

TEST_CASE("every easing curve starts at 0, ends at 1 and clamps its input", "[overlay][anim]") {
    const auto easing =
        GENERATE(anim::Easing::Linear, anim::Easing::InCubic, anim::Easing::OutCubic,
                 anim::Easing::InOutCubic, anim::Easing::OutExpo);
    REQUIRE(anim::ease(easing, 0.0F) == 0.0F);
    REQUIRE(anim::ease(easing, 1.0F) == 1.0F);
    REQUIRE(anim::ease(easing, -3.0F) == 0.0F);
    REQUIRE(anim::ease(easing, 7.0F) == 1.0F);
}

TEST_CASE("every easing curve is monotonic", "[overlay][anim]") {
    const auto easing =
        GENERATE(anim::Easing::Linear, anim::Easing::InCubic, anim::Easing::OutCubic,
                 anim::Easing::InOutCubic, anim::Easing::OutExpo);
    float previous = 0.0F;
    for (int i = 1; i <= kSamples; ++i) {
        const float value = anim::ease(easing, static_cast<float>(i) / kSamples);
        REQUIRE(value >= previous);
        previous = value;
    }
}

TEST_CASE("cubic easings have their textbook values", "[overlay][anim]") {
    REQUIRE_THAT(anim::easeInCubic(0.5F), WithinAbs(0.125, kTolerance));
    REQUIRE_THAT(anim::easeOutCubic(0.5F), WithinAbs(0.875, kTolerance));
    REQUIRE_THAT(anim::easeInOutCubic(0.5F), WithinAbs(0.5, kTolerance));
    REQUIRE_THAT(anim::easeInOutCubic(0.25F), WithinAbs(0.0625, kTolerance));
    // In-out is point-symmetric around the middle.
    REQUIRE_THAT(anim::easeInOutCubic(0.2F) + anim::easeInOutCubic(0.8F),
                 WithinAbs(1.0, kTolerance));
}

TEST_CASE("progress maps a beat onto the unit interval and treats zero duration as a step",
          "[overlay][anim]") {
    REQUIRE(anim::progress(0.5F, 1.0F, 2.0F) == 0.0F);
    REQUIRE_THAT(anim::progress(2.0F, 1.0F, 2.0F), WithinAbs(0.5, kTolerance));
    REQUIRE(anim::progress(9.0F, 1.0F, 2.0F) == 1.0F);
    REQUIRE(anim::progress(0.99F, 1.0F, 0.0F) == 0.0F);
    REQUIRE(anim::progress(1.0F, 1.0F, 0.0F) == 1.0F);
}

TEST_CASE("keyframe tracks hold their ends and ease between keys", "[overlay][anim]") {
    const std::array<anim::Keyframe, 3> track{{
        {.time = 1.0F, .value = 2.0F, .easing = anim::Easing::Linear},
        {.time = 2.0F, .value = 4.0F, .easing = anim::Easing::Linear},
        {.time = 3.0F, .value = 0.0F, .easing = anim::Easing::InOutCubic},
    }};
    REQUIRE(anim::sampleTrack(track, 0.0F) == 2.0F);
    REQUIRE(anim::sampleTrack(track, 1.0F) == 2.0F);
    REQUIRE_THAT(anim::sampleTrack(track, 1.5F), WithinAbs(3.0, kTolerance));
    REQUIRE(anim::sampleTrack(track, 2.0F) == 4.0F);
    REQUIRE_THAT(anim::sampleTrack(track, 2.5F), WithinAbs(2.0, kTolerance));
    REQUIRE(anim::sampleTrack(track, 3.0F) == 0.0F);
    REQUIRE(anim::sampleTrack(track, 99.0F) == 0.0F);
    REQUIRE(anim::sampleTrack({}, 1.0F) == 0.0F);
}

// --- randomness and particles ---------------------------------------------------------

TEST_CASE("SplitMix64 reproduces the reference sequence", "[overlay][anim]") {
    // First outputs for seed 0 from the reference implementation (Vigna).
    anim::SplitMix64 generator{0};
    REQUIRE(generator.next() == 0xE220A8397B1DCDAFULL);
    REQUIRE(generator.next() == 0x6E789E6AA1B965F4ULL);
    REQUIRE(generator.next() == 0x06C45D188009454FULL);
}

TEST_CASE("unit floats stay in the unit interval and hashes are stable", "[overlay][anim]") {
    anim::SplitMix64 generator{42};
    for (int i = 0; i < kSamples; ++i) {
        const float value = generator.nextUnit();
        REQUIRE(value >= 0.0F);
        REQUIRE(value < 1.0F);
    }
    REQUIRE(anim::hashUnit(7, 3) == anim::hashUnit(7, 3));
    REQUIRE(anim::hashUnit(7, 3) != anim::hashUnit(7, 4));
}

TEST_CASE("a particle field is deterministic for a seed", "[overlay][anim]") {
    const anim::ParticleField first{1234};
    const anim::ParticleField second{1234};
    const anim::ParticleField other{4321};
    REQUIRE(first.particles().size() == first.config().count);

    std::vector<anim::ParticlePoint> a;
    std::vector<anim::ParticlePoint> b;
    std::vector<anim::ParticlePoint> c;
    first.evaluate(0.4F, a);
    second.evaluate(0.4F, b);
    other.evaluate(0.4F, c);
    REQUIRE(samePositions(a, b));
    REQUIRE(a.front().x != c.front().x);
}

TEST_CASE("particles stream from outside the ring and settle on it", "[overlay][anim]") {
    const anim::ParticleField field{99};
    const anim::ParticleConfig& config = field.config();
    std::vector<anim::ParticlePoint> points;

    field.evaluate(0.0F, points);
    REQUIRE(std::ranges::equal(points, field.particles(), [](const auto& point, const auto& seed) {
        return near(radiusOf(point), seed.startRadius, 1e-4F);
    }));

    for (const float time : {0.1F, 0.3F, 0.5F, 0.7F}) {
        field.evaluate(time, points);
        REQUIRE(inFlight(points, config));
    }

    field.evaluate(field.settledAfter(), points);
    REQUIRE(std::ranges::all_of(points, [&config](const anim::ParticlePoint& point) {
        return near(radiusOf(point), config.targetRadius, 1e-4F) &&
               near(point.alpha, 0.0F, kTolerance);
    }));
}

TEST_CASE("particle positions depend on time only, not on evaluation history", "[overlay][anim]") {
    const anim::ParticleField field{5};
    std::vector<anim::ParticlePoint> direct;
    std::vector<anim::ParticlePoint> stepped;
    field.evaluate(0.37F, direct);
    for (const float time : {0.05F, 0.2F, 0.31F, 0.37F}) {
        field.evaluate(time, stepped);
    }
    REQUIRE(samePositions(direct, stepped));
}

// --- boot timeline -----------------------------------------------------------------------

TEST_CASE("the boot timeline hits its keyframes exactly", "[overlay][anim][boot]") {
    const anim::BootTimeline timeline;
    const anim::BootBeats& beats = timeline.beats();
    REQUIRE(timeline.title() == "J.A.R.V.I.S.");

    const anim::BootFrame start = timeline.evaluate(0.0F);
    REQUIRE(start.ignite == 0.0F);
    REQUIRE(start.flash == 0.0F);
    REQUIRE(start.spin == 0.0F);
    REQUIRE(start.titleChars == 0);
    REQUIRE(start.handoff == 0.0F);
    REQUIRE(start.bootOnly == 1.0F);
    REQUIRE(start.scanAlpha == 0.0F);
    REQUIRE_FALSE(start.finished);

    REQUIRE(timeline.evaluate(beats.pointGrow).ignite == 1.0F);
    REQUIRE(timeline.evaluate(beats.titleStart).titleChars == 1);
    const float typedAt = beats.titleStart + (11.0F * beats.charInterval);
    REQUIRE(timeline.evaluate(typedAt + 1e-4F).titleChars == 12);
    REQUIRE(timeline.evaluate(beats.handoffStart).handoff == 0.0F);
    REQUIRE_THAT(timeline.evaluate(beats.handoffStart + (beats.handoffDuration * 0.5F)).handoff,
                 WithinAbs(0.5, kTolerance));
    REQUIRE_THAT(timeline.evaluate(beats.systemsStart + (beats.scanDuration * 0.5F)).scanAlpha,
                 WithinAbs(1.0, kTolerance));

    const anim::BootFrame end = timeline.evaluate(timeline.duration());
    REQUIRE(end.finished);
    REQUIRE(end.handoff == 1.0F);
    REQUIRE(end.bootOnly == 0.0F);
    REQUIRE(end.spin == 1.0F);
    REQUIRE(end.titleChars == 12);
    REQUIRE_FALSE(end.cursorVisible);
}

TEST_CASE("the boot lasts about four seconds", "[overlay][anim][boot]") {
    const anim::BootTimeline timeline;
    REQUIRE(timeline.duration() > 3.6F);
    REQUIRE(timeline.duration() < 4.4F);
}

TEST_CASE("boot progressions only ever move forward", "[overlay][anim][boot]") {
    const anim::BootTimeline timeline;
    anim::BootFrame previous = timeline.evaluate(0.0F);
    int backwards = 0;
    for (int i = 1; i <= kSamples; ++i) {
        const float time = timeline.duration() * static_cast<float>(i) / kSamples;
        const anim::BootFrame frame = timeline.evaluate(time);
        backwards += movesForward(previous, frame) ? 0 : 1;
        previous = frame;
    }
    REQUIRE(backwards == 0);
}

TEST_CASE("the cursor shows while typing and is gone once the handoff starts",
          "[overlay][anim][boot]") {
    const anim::BootTimeline timeline;
    const anim::BootBeats& beats = timeline.beats();
    REQUIRE(timeline.evaluate(beats.titleStart + (2.5F * beats.charInterval)).cursorVisible);
    REQUIRE_FALSE(timeline.evaluate(beats.titleStart - 0.01F).cursorVisible);
    REQUIRE_FALSE(timeline.evaluate(beats.handoffStart + 0.01F).cursorVisible);
}

TEST_CASE("skipping jumps to the handoff, which still plays out", "[overlay][anim][boot]") {
    const anim::BootTimeline timeline;
    const float handoff = timeline.beats().handoffStart;
    const float glide = timeline.beats().handoffDuration;

    REQUIRE(timeline.timelineTime(0.8F, std::nullopt) == 0.8F);
    REQUIRE(timeline.timelineTime(-1.0F, std::nullopt) == 0.0F);
    // Before the skip nothing changes, at the skip the handoff begins.
    REQUIRE(timeline.timelineTime(0.3F, 0.5F) == 0.3F);
    REQUIRE(timeline.timelineTime(0.5F, 0.5F) == handoff);
    REQUIRE_THAT(timeline.timelineTime(0.5F + glide, 0.5F), WithinAbs(handoff + glide, kTolerance));
    REQUIRE(timeline.evaluate(timeline.timelineTime(0.5F + glide, 0.5F)).finished);
    // A skip during the handoff is ignored.
    REQUIRE(timeline.timelineTime(handoff + 0.2F, handoff + 0.1F) == handoff + 0.2F);
}

// --- state cross-fade -------------------------------------------------------------------------

TEST_CASE("a state cross-fade sums to one and lands on the target", "[overlay][anim][state]") {
    anim::StateBlend blend{0.2F};
    REQUIRE(blend.weights(0.0F) == anim::StateWeights{1.0F, 0.0F, 0.0F, 0.0F});
    REQUIRE_FALSE(blend.transitioning(0.0F));

    blend.setTarget(AssistantState::Listening, 1.0F);
    REQUIRE(blend.transitioning(1.05F));
    REQUIRE(sumsToOne(blend, 1.0F, 0.6F));
    // A spring from rest: 99 % of the way at the settle time, at rest soon after.
    REQUIRE(blend.weights(1.21F).at(1) >= 0.99F);
    requireWeights(blend.weights(1.6F), {0.0F, 1.0F, 0.0F, 0.0F});
    REQUIRE_FALSE(blend.transitioning(1.6F));
}

TEST_CASE("a state cross-fade eases in and out without a jolt", "[overlay][anim][state]") {
    anim::StateBlend blend{0.3F};
    blend.setTarget(AssistantState::Speaking, 0.0F);
    // Starts from rest (the first millisecond barely moves), never overshoots.
    REQUIRE(blend.weights(0.001F).at(3) < 0.001F);
    float previous = 0.0F;
    for (int step = 1; step <= 100; ++step) {
        const float weight = blend.weights(static_cast<float>(step) * 0.006F).at(3);
        REQUIRE(weight >= previous);
        REQUIRE(weight <= 1.0F);
        previous = weight;
    }
}

TEST_CASE("re-targeting mid-fade starts from what is on screen", "[overlay][anim][state]") {
    anim::StateBlend blend{0.2F};
    blend.setTarget(AssistantState::Listening, 0.0F);
    const anim::StateWeights before = blend.weights(0.05F);
    blend.setTarget(AssistantState::Thinking, 0.05F);
    REQUIRE(blend.weights(0.05F) == before);  // no jump
    // The listening weight keeps the momentum it had (it still rises for an
    // instant) instead of reversing on the spot: no kink in the motion.
    REQUIRE(blend.weights(0.055F).at(1) > before.at(1));
    REQUIRE_THAT(sum(blend.weights(0.2F)), WithinAbs(1.0, kTolerance));
    requireWeights(blend.weights(0.7F), {0.0F, 0.0F, 1.0F, 0.0F});

    blend.setTarget(AssistantState::Thinking, 0.8F);  // same target: no restart
    REQUIRE_FALSE(blend.transitioning(0.8F));
}

// --- springs and frame time -------------------------------------------------------

TEST_CASE("a critically damped spring lands on its target without overshooting",
          "[overlay][anim][spring]") {
    anim::Spring spring{0.3F};
    float previous = 0.0F;
    for (int frame = 0; frame < 60; ++frame) {
        const float value = spring.advance(1.0F, 1.0F / 60.0F);
        REQUIRE(value >= previous);
        REQUIRE(value <= 1.0F);
        previous = value;
    }
    REQUIRE(spring.restingAt(1.0F));  // snapped exactly, so "done" is an equality
    REQUIRE(spring.velocity() == 0.0F);
}

TEST_CASE("a spring covers 99 percent of the way in its settle time", "[overlay][anim][spring]") {
    const float omega = anim::springOmega(0.4F);
    const anim::SpringState settled = anim::springStep({}, 1.0F, omega, 0.4F);
    REQUIRE_THAT(settled.value, WithinAbs(0.99, 1e-4));
    const anim::SpringState half = anim::springStep({}, 1.0F, omega, 0.1F);
    REQUIRE(half.value > 0.4F);  // fast start...
    REQUIRE(half.velocity > 0.0F);
}

TEST_CASE("spring steps do not depend on the frame rate", "[overlay][anim][spring]") {
    const float omega = anim::springOmega(0.25F);
    const anim::SpringState once =
        anim::springStep({.value = 0.2F, .velocity = 3.0F}, 1.0F, omega, 0.1F);
    anim::SpringState stepped{.value = 0.2F, .velocity = 3.0F};
    for (int frame = 0; frame < 24; ++frame) {
        stepped = anim::springStep(stepped, 1.0F, omega, 0.1F / 24.0F);
    }
    REQUIRE_THAT(stepped.value, WithinAbs(once.value, 1e-5));
    REQUIRE_THAT(stepped.velocity, WithinAbs(once.velocity, 1e-3));
    // Zero time changes nothing.
    const anim::SpringState still = anim::springStep(once, 0.0F, omega, 0.0F);
    REQUIRE(still.value == once.value);
}

TEST_CASE("a re-targeted spring keeps its velocity", "[overlay][anim][spring]") {
    anim::Spring spring{0.3F};
    spring.advance(1.0F, 0.05F);
    const float velocity = spring.velocity();
    REQUIRE(velocity > 0.0F);
    spring.advance(0.0F, 1e-4F);  // turned around: the motion bends, it does not flip
    REQUIRE(spring.velocity() > 0.0F);
    REQUIRE_THAT(spring.velocity(), WithinAbs(velocity, velocity * 0.1F));
    spring.snap(0.5F);
    REQUIRE(spring.restingAt(0.5F));
}

TEST_CASE("frame deltas are clamped after a stall", "[overlay][anim]") {
    REQUIRE(anim::clampFrameDelta(0.016F) == 0.016F);
    REQUIRE(anim::clampFrameDelta(3.0F) == anim::kMaxFrameDelta);
    REQUIRE(anim::clampFrameDelta(-1.0F) == 0.0F);
}

TEST_CASE("a level history is followed with a fast attack and a slow release",
          "[overlay][anim][level]") {
    std::array<float, 3> shown{0.0F, 1.0F, 0.5F};
    const std::array<float, 3> target{1.0F, 0.0F, 0.5F};
    anim::followLevels(shown, target, 0.02F, 0.03F, 0.15F);
    const float rise = shown.at(0);
    const float fall = 1.0F - shown.at(1);
    REQUIRE(rise > 0.4F);
    REQUIRE(fall < 0.15F);
    REQUIRE(rise > fall);
    REQUIRE(shown.at(2) == 0.5F);
    // Only the common length is followed.
    std::array<float, 1> shorter{0.0F};
    anim::followLevels(shorter, target, 1.0F, 0.03F, 0.15F);
    REQUIRE(shorter.at(0) > 0.99F);
}

// --- level smoothing and periodic motion
// ----------------------------------------------------------

TEST_CASE("level smoothing depends on elapsed time, not frame count", "[overlay][anim][level]") {
    const auto settleOver = [](int frames) {
        anim::LevelSmoother smoother{0.05F, 0.3F};
        for (int i = 0; i < frames; ++i) {
            smoother.update(0.8F, 0.1F / static_cast<float>(frames));
        }
        return smoother.value();
    };
    const float once = settleOver(1);
    REQUIRE_THAT(settleOver(6), WithinAbs(once, 1e-5));
    REQUIRE_THAT(settleOver(60), WithinAbs(once, 1e-5));
    REQUIRE_THAT(once, WithinAbs(0.8F * (1.0F - std::exp(-0.1F / 0.05F)), 1e-5));
}

TEST_CASE("levels attack fast and release slowly", "[overlay][anim][level]") {
    anim::LevelSmoother rising{0.05F, 0.3F};
    anim::LevelSmoother falling{0.05F, 0.3F};
    falling.reset(1.0F);
    const float up = rising.update(1.0F, 0.05F);
    const float down = 1.0F - falling.update(0.0F, 0.05F);
    REQUIRE(up > down);
    REQUIRE(rising.update(0.3F, 0.0F) == up);  // zero dt changes nothing
}

TEST_CASE("idle breathing is a slow cosine between 0 and 1", "[overlay][anim]") {
    REQUIRE_THAT(anim::breathing(0.0F, 4.0F), WithinAbs(0.0, kTolerance));
    REQUIRE_THAT(anim::breathing(2.0F, 4.0F), WithinAbs(1.0, kTolerance));
    REQUIRE_THAT(anim::breathing(4.0F, 4.0F), WithinAbs(0.0, 1e-4));
    REQUIRE(anim::breathing(1.0F, 0.0F) == 0.0F);
}

TEST_CASE("the wake ripple spreads out while it fades", "[overlay][anim]") {
    const anim::Ripple fresh = anim::wakeRipple(1.0F, 1.4F);
    REQUIRE(fresh.radius == 1.0F);
    REQUIRE(fresh.alpha == 1.0F);
    const anim::Ripple later = anim::wakeRipple(0.5F, 1.4F);
    REQUIRE(later.radius > fresh.radius);
    REQUIRE(later.alpha < fresh.alpha);
    const anim::Ripple done = anim::wakeRipple(0.0F, 1.4F);
    REQUIRE_THAT(done.radius, WithinAbs(2.4, kTolerance));
    REQUIRE(done.alpha == 0.0F);
}

TEST_CASE("the radial waveform is symmetric, bounded and flat in silence",
          "[overlay][anim][level]") {
    std::array<float, 16> spokes{};
    const std::vector<float> silence(32, 0.0F);
    anim::radialWaveform(silence, 0.0F, 1.0F, 0.05F, spokes);
    REQUIRE(
        std::ranges::all_of(spokes, [](float spoke) { return near(spoke, 0.05F, kTolerance); }));

    std::vector<float> rising(32);
    float loudness = 0.0F;
    for (float& level : rising) {
        level = loudness / 31.0F;  // newest is loudest
        loudness += 1.0F;
    }
    anim::radialWaveform(rising, 0.7F, 2.5F, 0.05F, spokes);
    bool symmetric = true;
    for (std::size_t i = 1; i < spokes.size(); ++i) {
        symmetric = symmetric && spokes.at(i) == spokes.at(spokes.size() - i);
    }
    REQUIRE(symmetric);
    REQUIRE(
        std::ranges::all_of(spokes, [](float spoke) { return spoke >= 0.05F && spoke <= 1.0F; }));
    REQUIRE(spokes.front() > spokes.at(spokes.size() / 2));  // newest at the top

    anim::radialWaveform(rising, 1.0F, 0.0F, 0.05F, std::span<float>{});  // no-op, no crash
}

// --- the animator ---------------------------------------------------------------------------

TEST_CASE("the animator plays the boot only after it was requested", "[overlay][anim][boot]") {
    anim::OverlayAnimator animator{at(0.0)};
    OverlaySnapshot snapshot;
    REQUIRE_FALSE(animator.advance(snapshot, at(1.0)).boot.has_value());
    REQUIRE(animator.pace(snapshot) == anim::Pace::Idle);

    snapshot.bootStartedAt = at(1.0);
    REQUIRE(animator.advance(snapshot, at(1.5)).boot.has_value());
    REQUIRE(animator.pace(snapshot) == anim::Pace::Active);

    const double end = 1.0 + animator.timeline().duration() + 0.01;
    REQUIRE_FALSE(animator.advance(snapshot, at(end)).boot.has_value());
    REQUIRE(animator.pace(snapshot) == anim::Pace::Idle);
}

TEST_CASE("a skipped boot finishes one handoff after the skip", "[overlay][anim][boot]") {
    anim::OverlayAnimator animator{at(0.0)};
    OverlaySnapshot snapshot;
    snapshot.bootStartedAt = at(0.0);
    snapshot.bootSkippedAt = at(0.3);
    const float glide = animator.timeline().beats().handoffDuration;

    const auto& gliding = animator.advance(snapshot, at(0.3 + (glide * 0.5)));
    REQUIRE(gliding.boot.has_value());
    REQUIRE_THAT(gliding.boot.value_or(anim::BootFrame{}).handoff, WithinAbs(0.5, 1e-4));
    REQUIRE_FALSE(animator.advance(snapshot, at(0.31 + glide)).boot.has_value());
}

TEST_CASE("animator frames do not depend on the frame rate", "[overlay][anim]") {
    OverlaySnapshot snapshot;
    snapshot.bootStartedAt = at(0.0);
    snapshot.state = AssistantState::Listening;
    snapshot.levels.assign(64, 0.4F);
    anim::OverlayAnimator fast{at(0.0)};
    anim::OverlayAnimator slow{at(0.0)};
    fast.advance(snapshot, at(0.0));
    slow.advance(snapshot, at(0.0));
    for (int frame = 1; frame <= 120; ++frame) {
        fast.advance(snapshot, at(frame / 60.0));
    }
    for (int frame = 1; frame <= 20; ++frame) {
        slow.advance(snapshot, at(frame / 10.0));
    }
    const anim::AnimationFrame& a = fast.frame();
    const anim::AnimationFrame& b = slow.frame();
    REQUIRE_THAT(a.time, WithinAbs(b.time, 1e-5));
    REQUIRE_THAT(a.level, WithinAbs(b.level, 1e-4));
    REQUIRE_THAT(a.breathing, WithinAbs(b.breathing, 1e-5));
    REQUIRE((a.boot.has_value() && b.boot.has_value()));
    const anim::BootFrame bootA = a.boot.value_or(anim::BootFrame{});
    const anim::BootFrame bootB = b.boot.value_or(anim::BootFrame{});
    REQUIRE_THAT(bootA.ignite, WithinAbs(bootB.ignite, 1e-5));
    REQUIRE(bootA.titleChars == bootB.titleChars);
    requireWeights(a.weights, b.weights);
}

TEST_CASE("the animator stays active while a state fades or a level decays", "[overlay][anim]") {
    anim::OverlayAnimator animator{at(0.0)};
    OverlaySnapshot snapshot;
    snapshot.state = AssistantState::Speaking;
    snapshot.levels.assign(8, 0.9F);
    animator.advance(snapshot, at(0.0));
    animator.advance(snapshot, at(0.5));
    REQUIRE(animator.pace(snapshot) == anim::Pace::Active);

    snapshot.state = AssistantState::Idle;
    snapshot.levels.assign(8, 0.0F);
    animator.advance(snapshot, at(0.55));
    REQUIRE(animator.pace(snapshot) == anim::Pace::Active);  // cross-fade and release
    // Frames, not one leap: a single 4.5 s step is a stall and is clamped.
    for (int frame = 1; frame <= 45; ++frame) {
        animator.advance(snapshot, at(0.55 + (frame * 0.1)));
    }
    REQUIRE(animator.pace(snapshot) == anim::Pace::Idle);
    requireWeights(animator.frame().weights, {1.0F, 0.0F, 0.0F, 0.0F});
}

// --- frame statistics -------------------------------------------------------------------------

TEST_CASE("frame statistics average render time and frame interval", "[overlay][anim]") {
    anim::FrameStatsAccumulator accumulator{0.5};
    accumulator.record(2.0, 999.0);  // first interval has no predecessor
    REQUIRE(accumulator.stats().frames == 1);
    REQUIRE(accumulator.stats().renderMs == 2.0);
    REQUIRE(accumulator.stats().intervalMs == 0.0);

    accumulator.record(4.0, 16.0);
    REQUIRE(accumulator.stats().renderMs == 3.0);
    REQUIRE(accumulator.stats().intervalMs == 16.0);
    accumulator.record(4.0, 18.0);
    REQUIRE(accumulator.stats().intervalMs == 17.0);

    accumulator.reset();
    REQUIRE(accumulator.stats().frames == 0);
}

namespace {

/// 100 intervals: 98 on time, one missed refresh (40 ms, a slow frame)
/// and one late one under 1.5x the period (20 ms, not a hitch).
anim::FrameStatsAccumulator mostlySmoothFrames() {
    anim::FrameStatsAccumulator accumulator;
    accumulator.record(1.0, 0.0, 16.7);  // the first frame has no interval
    for (int i = 0; i < 98; ++i) {
        accumulator.record(1.0, 16.7, 16.7);
    }
    accumulator.record(9.0, 40.0, 16.7);
    accumulator.record(1.0, 20.0, 16.7);
    return accumulator;
}

}  // namespace

TEST_CASE("frame statistics report interval percentiles", "[overlay][anim]") {
    const anim::FrameStats stats = mostlySmoothFrames().stats();
    REQUIRE(stats.frames == 101);
    REQUIRE(stats.intervalP50Ms == 16.7);
    REQUIRE(stats.intervalP99Ms == 20.0);
    REQUIRE(stats.intervalMaxMs == 40.0);
    REQUIRE(stats.renderP99Ms == 1.0);
}

TEST_CASE("frame statistics count hitches until reset", "[overlay][anim]") {
    anim::FrameStatsAccumulator accumulator = mostlySmoothFrames();
    REQUIRE(accumulator.stats().hitches == 1);
    accumulator.reset();
    REQUIRE(accumulator.stats().intervalMaxMs == 0.0);
    REQUIRE(accumulator.stats().hitches == 0);
}

TEST_CASE("frame statistics count refreshes off the frame cadence", "[overlay][anim]") {
    anim::FrameStatsAccumulator accumulator;
    accumulator.recordPresentation(1, 4, 4, 240.0);  // on cadence
    accumulator.recordPresentation(1, 5, 4, 240.0);  // held one refresh too long
    accumulator.recordPresentation(1, 3, 4, 240.0);  // and one too short
    accumulator.recordPresentation(2, 2, 1, 60.0);   // two frames, one refresh each
    const anim::FrameStats stats = accumulator.stats();
    REQUIRE(stats.presentedFrames == 5);
    REQUIRE(stats.offCadenceRefreshes == 2);
    REQUIRE(stats.refreshHz == 60.0);
}

TEST_CASE("percentiles use the nearest rank of the values", "[overlay][anim]") {
    std::array<double, 5> values{5.0, 1.0, 4.0, 2.0, 3.0};
    REQUIRE(anim::percentile(values, 0.5) == 3.0);
    REQUIRE(anim::percentile(values, 1.0) == 5.0);
    REQUIRE(anim::percentile(values, 0.0) == 1.0);
    REQUIRE(anim::percentile(std::span<double>{}, 0.5) == 0.0);
}
