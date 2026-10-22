#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/animator.hpp"
#include "jarvis/overlay/model.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <optional>

using Catch::Matchers::WithinAbs;
using jarvis::overlay::AssistantState;
using jarvis::overlay::OverlaySnapshot;
namespace anim = jarvis::overlay::anim;

namespace {

constexpr float kTolerance = 1e-5F;
constexpr float kJustBefore = 1e-3F;

using Clock = anim::OverlayAnimator::Clock;

Clock::time_point at(double seconds) {
    return Clock::time_point{
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds))};
}

/// True when every ring of @p frame has arrival @p value.
bool allArrived(const anim::BootFrame& frame, float value) {
    return std::ranges::all_of(frame.ringArrival, [value](float a) { return a == value; });
}

/// True when a flying ring only ever shrinks towards its locked size.
bool scaleNeverGrows() {
    float previous = anim::ringPose(0.0F).scale;
    for (int i = 1; i <= 100; ++i) {
        const float scale = anim::ringPose(static_cast<float>(i) / 100.0F).scale;
        if (scale > previous) {
            return false;
        }
        previous = scale;
    }
    return true;
}

}  // namespace

TEST_CASE("the boot has four acts in order with exact boundaries", "[overlay][boot][acts]") {
    const anim::BootTimeline timeline;
    const anim::BootBeats& b = timeline.beats();
    REQUIRE(timeline.actAt(0.0F) == anim::BootAct::Ignition);
    REQUIRE(timeline.actAt(b.assemblyStart - kJustBefore) == anim::BootAct::Ignition);
    REQUIRE(timeline.actAt(b.assemblyStart) == anim::BootAct::Assembly);
    REQUIRE(timeline.actAt(b.systemsStart - kJustBefore) == anim::BootAct::Assembly);
    REQUIRE(timeline.actAt(b.systemsStart) == anim::BootAct::SystemsCheck);
    REQUIRE(timeline.actAt(b.handoffStart - kJustBefore) == anim::BootAct::SystemsCheck);
    REQUIRE(timeline.actAt(b.handoffStart) == anim::BootAct::Handoff);
    REQUIRE(timeline.actAt(timeline.duration() + 5.0F) == anim::BootAct::Handoff);
    REQUIRE(timeline.evaluate(b.systemsStart).act == anim::BootAct::SystemsCheck);
    // Each act is long enough to read.
    REQUIRE(b.assemblyStart >= 0.5F);
    REQUIRE(b.systemsStart - b.assemblyStart >= 1.0F);
    REQUIRE(b.handoffStart - b.systemsStart >= 1.0F);
    REQUIRE(b.handoffDuration >= 0.4F);
}

TEST_CASE("act 1 ignites a point, flashes and sends out a shock ring", "[overlay][boot][acts]") {
    const anim::BootTimeline timeline;
    const anim::BootBeats& b = timeline.beats();
    REQUIRE(timeline.evaluate(b.pointGrow).ignite == 1.0F);
    REQUIRE(timeline.evaluate(b.flashPeak).flash == 1.0F);
    REQUIRE(timeline.evaluate(b.shockStart - kJustBefore).shockAlpha == 0.0F);
    const anim::BootFrame early = timeline.evaluate(b.shockStart + 0.1F);
    const anim::BootFrame late = timeline.evaluate(b.shockStart + 0.5F);
    REQUIRE(early.shockAlpha > late.shockAlpha);
    REQUIRE(late.shockRadius > early.shockRadius);
    REQUIRE(timeline.evaluate(b.shockStart + b.shockDuration).shockAlpha == 0.0F);
}

TEST_CASE("act 2 flies the rings in one after another and draws the globe",
          "[overlay][boot][acts]") {
    const anim::BootTimeline timeline;
    const anim::BootBeats& b = timeline.beats();
    const anim::BootFrame before = timeline.evaluate(b.assemblyStart - kJustBefore);
    REQUIRE(allArrived(before, 0.0F));
    const anim::BootFrame during = timeline.evaluate(b.assemblyStart + (b.ringStagger * 1.5F));
    REQUIRE(during.ringArrival.front() > during.ringArrival.back());
    REQUIRE(allArrived(timeline.evaluate(b.systemsStart), 1.0F));
    REQUIRE(timeline.evaluate(b.globeStart).globe == 0.0F);
    REQUIRE_THAT(timeline.evaluate(b.globeStart + (b.globeDuration * 0.5F)).globe,
                 WithinAbs(0.5, kTolerance));
    REQUIRE(timeline.evaluate(b.globeStart + b.globeDuration + kJustBefore).globe == 1.0F);
}

TEST_CASE("a flying ring starts big, tilted and invisible and locks flat", "[overlay][boot]") {
    const anim::RingPose far = anim::ringPose(0.0F);
    const anim::RingPose locked = anim::ringPose(1.0F);
    REQUIRE(far.scale > 2.0F);
    REQUIRE(far.tilt > 0.5F);
    REQUIRE(far.alpha == 0.0F);
    REQUIRE(locked.scale == 1.0F);
    REQUIRE(locked.tilt == 0.0F);
    REQUIRE(locked.alpha == 1.0F);
    REQUIRE(scaleNeverGrows());
}

TEST_CASE("act 3 counts labels up and types the readout line by line", "[overlay][boot][acts]") {
    const anim::BootTimeline timeline;
    const anim::BootBeats& b = timeline.beats();
    REQUIRE(timeline.evaluate(b.systemsStart).labels == 0.0F);
    REQUIRE(timeline.evaluate(b.systemsStart + b.labelCount).labels == 1.0F);

    constexpr std::size_t kLength = 20;
    REQUIRE(timeline.readoutChars(0, kLength, b.readoutStart - kJustBefore) == 0);
    REQUIRE(timeline.readoutChars(0, kLength, b.readoutStart) == 1);
    const float second = b.readoutStart + b.readoutLineInterval;
    REQUIRE(timeline.readoutChars(1, kLength, second - kJustBefore) == 0);
    REQUIRE(timeline.readoutChars(1, kLength, second) == 1);
    REQUIRE(timeline.readoutChars(0, kLength, second) > timeline.readoutChars(1, kLength, second));
    REQUIRE(timeline.readoutChars(0, kLength, b.handoffStart) == kLength);
    // Six lines of 30 characters are fully typed before the hand-off.
    REQUIRE(timeline.readoutChars(5, 30, b.handoffStart) == 30);
}

TEST_CASE("the title types on with glitch spikes", "[overlay][boot][acts]") {
    const anim::BootTimeline timeline;
    const anim::BootBeats& b = timeline.beats();
    REQUIRE(timeline.evaluate(b.titleStart - kJustBefore).titleChars == 0);
    REQUIRE(timeline.evaluate(b.titleStart).titleChars == 1);
    REQUIRE(timeline.evaluate(b.subtitleStart).titleChars == timeline.title().size());
    REQUIRE(timeline.evaluate(b.titleStart + 0.015F).glitch > 0.4F);
    REQUIRE(timeline.evaluate(b.flashPeak).glitch > 0.8F);
}

TEST_CASE("act 4 whips into the corner", "[overlay][boot][acts]") {
    const anim::BootTimeline timeline;
    const anim::BootBeats& b = timeline.beats();
    REQUIRE(timeline.evaluate(b.handoffStart).whip == 0.0F);
    REQUIRE_THAT(timeline.evaluate(b.handoffStart + (b.handoffDuration * 0.5F)).whip,
                 WithinAbs(1.0, kTolerance));
    REQUIRE(timeline.evaluate(timeline.duration()).whip < 1e-6F);
    REQUIRE(timeline.evaluate(b.handoffStart - 0.5F).whip == 0.0F);
}

TEST_CASE("any skip lands on the hand-off, never on the end", "[overlay][boot][skip]") {
    const anim::BootTimeline timeline;
    const anim::BootBeats& b = timeline.beats();
    for (const float skipAt : {0.05F, 1.0F, 2.5F, b.handoffStart - 0.01F}) {
        const float time = timeline.timelineTime(skipAt, skipAt);
        REQUIRE(time == b.handoffStart);
        const anim::BootFrame frame = timeline.evaluate(time);
        REQUIRE(frame.act == anim::BootAct::Handoff);
        REQUIRE_FALSE(frame.finished);
    }
}

TEST_CASE("leaving Idle during the boot skips to the hand-off", "[overlay][boot][skip]") {
    jarvis::overlay::OverlayModel model;
    anim::OverlayAnimator animator{at(0.0)};
    REQUIRE(model.playBoot(at(0.0)));
    model.setState(AssistantState::Listening, at(1.0));
    const OverlaySnapshot snapshot = model.snapshot(at(1.0));
    const auto& frame = animator.advance(snapshot, at(1.0));
    REQUIRE(frame.boot.has_value());
    REQUIRE(frame.boot.value_or(anim::BootFrame{}).act == anim::BootAct::Handoff);
    const float glide = animator.timeline().beats().handoffDuration;
    REQUIRE_FALSE(animator.advance(snapshot, at(1.01 + glide)).boot.has_value());
}

TEST_CASE("the boot is the brightest, glitchiest moment and idle the calmest", "[overlay][boot]") {
    anim::OverlayAnimator animator{at(0.0)};
    OverlaySnapshot snapshot;
    snapshot.bootStartedAt = at(0.0);
    const float flash = animator.timeline().beats().flashPeak;
    const anim::AnimationFrame booting = animator.advance(snapshot, at(flash));
    REQUIRE(booting.glitch > 0.5F);
    REQUIRE(booting.bloom > 1.5F);

    snapshot.bootStartedAt.reset();
    animator.advance(snapshot, at(20.0));
    const anim::AnimationFrame& idle = animator.advance(snapshot, at(20.1));
    REQUIRE(idle.glitch == 0.0F);
    REQUIRE(idle.bloom < booting.bloom * 0.5F);
    REQUIRE(idle.particleShare < 1.0F);
    REQUIRE(animator.pace(snapshot) == anim::Pace::Idle);
}

TEST_CASE("the animator's intensity scales particles and bloom", "[overlay][effects]") {
    const anim::OverlayAnimator calm{at(0.0),
                                     anim::AnimatorConfig{.intensity = anim::Intensity::Calm}};
    const anim::OverlayAnimator max{at(0.0),
                                    anim::AnimatorConfig{.intensity = anim::Intensity::Max}};
    REQUIRE(calm.field().particles().size() < max.field().particles().size());
    REQUIRE(calm.scale().bloom < max.scale().bloom);
}

TEST_CASE("transcript rows show while busy and linger briefly after", "[overlay][anim]") {
    REQUIRE_FALSE(anim::transcriptWanted(false, AssistantState::Speaking, 0.0F));
    REQUIRE(anim::transcriptWanted(true, AssistantState::Listening, 100.0F));
    REQUIRE(anim::transcriptWanted(true, AssistantState::Idle, anim::kTranscriptLinger - 0.1F));
    REQUIRE_FALSE(anim::transcriptWanted(true, AssistantState::Idle, anim::kTranscriptLinger));
}

TEST_CASE("the transcript slides in with text and out after going idle", "[overlay][anim]") {
    anim::OverlayAnimator animator{at(0.0)};
    OverlaySnapshot snapshot;
    animator.advance(snapshot, at(0.0));
    REQUIRE(animator.advance(snapshot, at(1.0)).transcript == 0.0F);
    snapshot.state = AssistantState::Listening;
    snapshot.transcript = "open the terminal";
    animator.advance(snapshot, at(1.05));
    REQUIRE(animator.frame().transcript > 0.2F);
    REQUIRE(animator.advance(snapshot, at(2.0)).transcript > 0.99F);
    snapshot.state = AssistantState::Idle;
    animator.advance(snapshot, at(2.1));
    REQUIRE(animator.advance(snapshot, at(2.0 + anim::kTranscriptLinger - 0.5)).transcript > 0.99F);
    animator.advance(snapshot, at(2.2 + anim::kTranscriptLinger));
    REQUIRE(animator.pace(snapshot) == anim::Pace::Active);  // still fading out
    REQUIRE(animator.advance(snapshot, at(5.0 + anim::kTranscriptLinger)).transcript < 0.01F);
}
