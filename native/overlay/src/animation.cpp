#include "jarvis/overlay/animation.hpp"

#include "jarvis/overlay/effects.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace jarvis::overlay::anim {

namespace {

constexpr float kHalf = 0.5F;
constexpr std::size_t kStateCount = 4;
constexpr float kPi = std::numbers::pi_v<float>;
/// Exponent of easeOutExpo; 10 makes f(1) within 0.1% of 1 before the snap.
constexpr float kExpoSteepness = 10.0F;

// SplitMix64 constants (Steele, Lea, Flood 2014).
constexpr std::uint64_t kGoldenGamma = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t kMixA = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t kMixB = 0x94D049BB133111EBULL;
constexpr unsigned kShiftA = 30;
constexpr unsigned kShiftB = 27;
constexpr unsigned kShiftC = 31;
/// A float has a 24-bit significand; the top 24 bits map exactly onto [0, 1).
constexpr unsigned kUnitBits = 24;
constexpr unsigned kUnitShift = 64 - kUnitBits;
constexpr float kUnitScale = 1.0F / static_cast<float>(1U << kUnitBits);

// Ignition: the centre flashes white-hot, then settles to an ember behind the rings.
constexpr float kFlashSettle = 0.75F;
constexpr float kFlashEmber = 0.18F;
constexpr float kShockStartRadius = 0.1F;
constexpr float kShockSpread = 2.3F;
/// One revolution of the radial sweep every this many seconds.
constexpr float kSweepPeriod = 1.25F;

// Ring fly-in: from this far in front of the screen (in camera distances
// of 4 units), tilted by up to this much, fading in over the first part.
constexpr float kFlyCamera = 4.0F;
constexpr float kFlyDepth = 2.8F;
constexpr float kFlyTilt = 1.1F;
constexpr float kFlyFadeIn = 0.35F;
/// easeOutExpo passes 95% here: the moment a ring visibly "locks".
constexpr float kLockShare = 0.43F;

// Glitch spikes of the boot (strength, decay seconds).
constexpr float kFlashGlitch = 0.9F;
constexpr float kFlashGlitchDecay = 0.10F;
constexpr float kLockGlitch = 0.32F;
constexpr float kLockGlitchDecay = 0.06F;
constexpr float kTitleGlitch = 0.55F;
constexpr float kTitleGlitchDecay = 0.12F;
constexpr float kTypeGlitch = 0.25F;
constexpr float kTypeGlitchDecay = 0.04F;
constexpr std::size_t kTypeGlitchEvery = 3;
constexpr float kHandoffGlitch = 0.6F;
constexpr float kHandoffGlitchDecay = 0.10F;
/// A spike starts this much before its beat so it peaks on the beat.
constexpr float kSpikeLead = 0.015F;

// Particles fade in over the first 20% of their trip and dissolve into the
// ring over the last 20%.
constexpr float kParticleFadeIn = 0.2F;
constexpr float kParticleFadeOutStart = 0.8F;
constexpr float kParticleMinSize = 0.5F;
constexpr float kParticleMaxSize = 1.5F;
constexpr float kParticleMinSwirlShare = 0.2F;

// Radial waveform: share of a spoke driven by history versus live energy.
constexpr float kHistoryShare = 0.55F;
constexpr float kEnergyShare = 0.45F;
constexpr float kShimmerTurns = 1.5F;
constexpr float kShimmerSpeed = 2.3F;

/// omega x t at which a critically damped spring from rest is 99 % there:
/// the root of (1 + x) e^-x = 0.01.
constexpr float kSpringSettle = 6.6384F;

// Frame statistics percentiles.
constexpr double kMedian = 0.50;
constexpr double kP95 = 0.95;
constexpr double kP99 = 0.99;

StateWeights oneHot(AssistantState state) noexcept {
    StateWeights weights{};
    weights.at(stateIndex(state)) = 1.0F;
    return weights;
}

std::vector<GlitchSpike> bootSpikes(const BootBeats& b, std::size_t titleLength) {
    std::vector<GlitchSpike> spikes;
    spikes.push_back(
        {.at = b.flashPeak - kSpikeLead, .strength = kFlashGlitch, .decay = kFlashGlitchDecay});
    for (std::size_t i = 0; i < kRingCount; ++i) {
        const float start = b.assemblyStart + (b.ringStagger * static_cast<float>(i));
        spikes.push_back({.at = start + (b.ringFlight * kLockShare),
                          .strength = kLockGlitch,
                          .decay = kLockGlitchDecay});
    }
    spikes.push_back({.at = b.titleStart, .strength = kTitleGlitch, .decay = kTitleGlitchDecay});
    for (std::size_t i = kTypeGlitchEvery; i < titleLength; i += kTypeGlitchEvery) {
        spikes.push_back({.at = b.titleStart + (b.charInterval * static_cast<float>(i)),
                          .strength = kTypeGlitch,
                          .decay = kTypeGlitchDecay});
    }
    spikes.push_back(
        {.at = b.handoffStart, .strength = kHandoffGlitch, .decay = kHandoffGlitchDecay});
    return spikes;
}

/// Characters typed at @p interval per character since @p start, at most @p length.
std::size_t typedChars(float time, float start, float interval, std::size_t length) noexcept {
    if (time < start || interval <= 0.0F) {
        return 0;
    }
    const auto typed = static_cast<std::size_t>((time - start) / interval) + 1;
    return std::min(typed, length);
}

}  // namespace

// --- scalar helpers ------------------------------------------------------------

float clamp01(float value) noexcept {
    return std::clamp(value, 0.0F, 1.0F);
}

float lerp(float from, float to, float t) noexcept {
    return from + ((to - from) * t);
}

float progress(float time, float start, float duration) noexcept {
    if (duration <= 0.0F) {
        return time >= start ? 1.0F : 0.0F;
    }
    return clamp01((time - start) / duration);
}

float smoothstep(float edge0, float edge1, float value) noexcept {
    const float t = progress(value, edge0, edge1 - edge0);
    return t * t * (3.0F - (2.0F * t));
}

// --- easing ----------------------------------------------------------------------

float easeInCubic(float t) noexcept {
    const float x = clamp01(t);
    return x * x * x;
}

float easeOutCubic(float t) noexcept {
    const float inverse = 1.0F - clamp01(t);
    return 1.0F - (inverse * inverse * inverse);
}

float easeInOutCubic(float t) noexcept {
    const float x = clamp01(t);
    if (x < kHalf) {
        return 4.0F * x * x * x;
    }
    const float tail = (-2.0F * x) + 2.0F;
    return 1.0F - ((tail * tail * tail) * kHalf);
}

float easeOutExpo(float t) noexcept {
    const float x = clamp01(t);
    return x >= 1.0F ? 1.0F : 1.0F - std::exp2(-kExpoSteepness * x);
}

float ease(Easing easing, float t) noexcept {
    switch (easing) {
        case Easing::InCubic:
            return easeInCubic(t);
        case Easing::OutCubic:
            return easeOutCubic(t);
        case Easing::InOutCubic:
            return easeInOutCubic(t);
        case Easing::OutExpo:
            return easeOutExpo(t);
        case Easing::Linear:
            break;
    }
    return clamp01(t);
}

// --- frame time and springs ------------------------------------------------------

float clampFrameDelta(float dtSeconds) noexcept {
    return std::clamp(dtSeconds, 0.0F, kMaxFrameDelta);
}

float springOmega(float settleSeconds) noexcept {
    // A zero settle time is an (almost) instant spring, never a division by zero.
    constexpr float kShortest = 1e-3F;
    return kSpringSettle / std::max(settleSeconds, kShortest);
}

SpringState springStep(SpringState state, float target, float omega, float dtSeconds) noexcept {
    if (dtSeconds <= 0.0F || omega <= 0.0F) {
        return state;
    }
    // x(t) = target + (c1 + c2 t) e^(-omega t), the critically damped solution.
    const float c1 = state.value - target;
    const float c2 = state.velocity + (omega * c1);
    const float decay = std::exp(-omega * dtSeconds);
    const float offset = (c1 + (c2 * dtSeconds)) * decay;
    return SpringState{.value = target + offset, .velocity = (c2 * decay) - (omega * offset)};
}

Spring::Spring(float settleSeconds, float value) noexcept
    : omega_(springOmega(settleSeconds)), state_{.value = value, .velocity = 0.0F} {}

float Spring::advance(float target, float dtSeconds) noexcept {
    state_ = springStep(state_, target, omega_, dtSeconds);
    if (std::abs(state_.value - target) <= kSpringRest &&
        std::abs(state_.velocity) <= kSpringRest * omega_) {
        state_ = SpringState{.value = target, .velocity = 0.0F};
    }
    return state_.value;
}

void Spring::snap(float value) noexcept {
    state_ = SpringState{.value = value, .velocity = 0.0F};
}

bool Spring::restingAt(float target) const noexcept {
    return state_.value == target && state_.velocity == 0.0F;
}

// --- keyframe tracks ---------------------------------------------------------------

float sampleTrack(std::span<const Keyframe> keys, float time) noexcept {
    if (keys.empty()) {
        return 0.0F;
    }
    if (time <= keys.front().time) {
        return keys.front().value;
    }
    const auto next =
        std::ranges::find_if(keys, [time](const Keyframe& key) { return key.time > time; });
    if (next == keys.end()) {
        return keys.back().value;
    }
    const Keyframe& to = *next;
    const Keyframe& from = *std::prev(next);
    const float t = ease(to.easing, progress(time, from.time, to.time - from.time));
    return lerp(from.value, to.value, t);
}

// --- deterministic randomness ---------------------------------------------------------

std::uint64_t SplitMix64::next() noexcept {
    state_ += kGoldenGamma;
    std::uint64_t mixed = state_;
    mixed = (mixed ^ (mixed >> kShiftA)) * kMixA;
    mixed = (mixed ^ (mixed >> kShiftB)) * kMixB;
    return mixed ^ (mixed >> kShiftC);
}

float SplitMix64::nextUnit() noexcept {
    return static_cast<float>(next() >> kUnitShift) * kUnitScale;
}

float SplitMix64::uniform(float low, float high) noexcept {
    return lerp(low, high, nextUnit());
}

float hashUnit(std::uint64_t index, std::uint64_t step) noexcept {
    SplitMix64 generator{(index * kGoldenGamma) ^ step};
    return generator.nextUnit();
}

// --- boot particle stream -------------------------------------------------------------------

ParticleField::ParticleField(std::uint64_t seed, ParticleConfig config) : config_(config) {
    SplitMix64 random{seed};
    particles_.reserve(config_.count);
    for (std::size_t i = 0; i < config_.count; ++i) {
        // Each field is drawn in a fixed order so the sequence for a seed
        // never changes when the loop body is reformatted.
        Particle particle;
        particle.angle = random.uniform(0.0F, kTau);
        particle.startRadius = random.uniform(config_.minStartRadius, config_.maxStartRadius);
        particle.delay = random.uniform(0.0F, config_.maxDelay);
        particle.travel = random.uniform(config_.minTravel, config_.maxTravel);
        particle.swirl =
            random.uniform(config_.maxSwirl * kParticleMinSwirlShare, config_.maxSwirl);
        particle.size = random.uniform(kParticleMinSize, kParticleMaxSize);
        particles_.push_back(particle);
    }
}

void ParticleField::evaluate(float time, std::vector<ParticlePoint>& out) const {
    out.resize(particles_.size());
    std::size_t index = 0;
    for (const Particle& particle : particles_) {
        const float p = progress(time, particle.delay, particle.travel);
        const float radius = lerp(particle.startRadius, config_.targetRadius, easeInOutCubic(p));
        const float angle = particle.angle - (particle.swirl * (1.0F - easeOutCubic(p)));
        const bool started = time >= particle.delay;
        const float alpha = started ? smoothstep(0.0F, kParticleFadeIn, p) *
                                          (1.0F - smoothstep(kParticleFadeOutStart, 1.0F, p))
                                    : 0.0F;
        out.at(index) = ParticlePoint{.x = radius * std::cos(angle),
                                      .y = radius * std::sin(angle),
                                      .alpha = alpha,
                                      .size = particle.size};
        ++index;
    }
}

float ParticleField::settledAfter() const noexcept {
    float latest = 0.0F;
    for (const Particle& particle : particles_) {
        latest = std::max(latest, particle.delay + particle.travel);
    }
    return latest;
}

// --- boot sequence -----------------------------------------------------------------------

RingPose ringPose(float arrival) noexcept {
    const float a = clamp01(arrival);
    const float depth = kFlyDepth * (1.0F - a);
    return RingPose{.scale = kFlyCamera / (kFlyCamera - depth),
                    .tilt = kFlyTilt * (1.0F - a),
                    .alpha = smoothstep(0.0F, kFlyFadeIn, a)};
}

BootTimeline::BootTimeline(BootBeats beats, std::string title)
    : beats_(beats), title_(std::move(title)), spikes_(bootSpikes(beats_, title_.size())) {}

float BootTimeline::duration() const noexcept {
    return beats_.handoffStart + beats_.handoffDuration;
}

float BootTimeline::timelineTime(float sinceStart,
                                 std::optional<float> skippedAfter) const noexcept {
    const float time = std::max(sinceStart, 0.0F);
    if (!skippedAfter.has_value()) {
        return time;
    }
    const float skipAt = std::max(*skippedAfter, 0.0F);
    if (skipAt >= beats_.handoffStart || time < skipAt) {
        return time;
    }
    return beats_.handoffStart + (time - skipAt);
}

BootAct BootTimeline::actAt(float time) const noexcept {
    if (time < beats_.assemblyStart) {
        return BootAct::Ignition;
    }
    if (time < beats_.systemsStart) {
        return BootAct::Assembly;
    }
    if (time < beats_.handoffStart) {
        return BootAct::SystemsCheck;
    }
    return BootAct::Handoff;
}

std::size_t BootTimeline::readoutChars(std::size_t line, std::size_t length,
                                       float time) const noexcept {
    const float start =
        beats_.readoutStart + (beats_.readoutLineInterval * static_cast<float>(line));
    return typedChars(time, start, beats_.readoutCharInterval, length);
}

BootFrame BootTimeline::evaluate(float time) const noexcept {
    BootFrame frame;
    frame.time = time;
    frame.act = actAt(time);
    evaluateIgnition(time, frame);
    evaluateAssembly(time, frame);
    evaluateSystems(time, frame);
    evaluateHandoff(time, frame);
    frame.glitch = glitchAmount(spikes_, time);
    return frame;
}

void BootTimeline::evaluateIgnition(float time, BootFrame& frame) const noexcept {
    const BootBeats& b = beats_;
    const std::array<Keyframe, 3> flash{{
        {.time = 0.0F, .value = 0.0F, .easing = Easing::Linear},
        {.time = b.flashPeak, .value = 1.0F, .easing = Easing::OutCubic},
        {.time = kFlashSettle, .value = kFlashEmber, .easing = Easing::InOutCubic},
    }};
    frame.ignite = easeOutCubic(progress(time, 0.0F, b.pointGrow));
    frame.flash = sampleTrack(flash, time);
    const float shock = progress(time, b.shockStart, b.shockDuration);
    const bool shocking = time >= b.shockStart && shock < 1.0F;
    frame.shockRadius = kShockStartRadius + (kShockSpread * easeOutCubic(shock));
    frame.shockAlpha = shocking ? (1.0F - shock) * (1.0F - shock) : 0.0F;
}

void BootTimeline::evaluateAssembly(float time, BootFrame& frame) const noexcept {
    const BootBeats& b = beats_;
    for (std::size_t i = 0; i < frame.ringArrival.size(); ++i) {
        const float start = b.assemblyStart + (b.ringStagger * static_cast<float>(i));
        frame.ringArrival.at(i) = easeOutExpo(progress(time, start, b.ringFlight));
    }
    frame.globe = progress(time, b.globeStart, b.globeDuration);
    frame.particleTime = time - b.particlesStart;
    frame.sweepAngle = (kTau * time / kSweepPeriod) - (kTau / 4.0F);
    frame.spin = easeOutCubic(progress(time, 0.0F, duration()));
}

void BootTimeline::evaluateSystems(float time, BootFrame& frame) const noexcept {
    const BootBeats& b = beats_;
    frame.labels = easeOutCubic(progress(time, b.systemsStart, b.labelCount));
    const float scan = progress(time, b.systemsStart, b.scanDuration);
    frame.scan = easeInOutCubic(scan);
    const bool scanning = time >= b.systemsStart && time <= b.systemsStart + b.scanDuration;
    frame.scanAlpha = scanning ? std::sin(kPi * scan) : 0.0F;

    const std::size_t length = title_.size();
    frame.titleChars = typedChars(time, b.titleStart, b.charInterval, length);
    const float typedAt =
        b.titleStart + (static_cast<float>(length > 0 ? length - 1 : 0) * b.charInterval);
    const bool typing = time >= b.titleStart && frame.titleChars < length;
    const bool blinkOn =
        b.cursorBlink > 0.0F && std::fmod(time - typedAt, 2.0F * b.cursorBlink) < b.cursorBlink;
    frame.cursorVisible = time < b.handoffStart && (typing || (time >= typedAt && blinkOn));
    frame.subtitle = easeOutCubic(progress(time, b.subtitleStart, b.subtitleDuration));
}

void BootTimeline::evaluateHandoff(float time, BootFrame& frame) const noexcept {
    const BootBeats& b = beats_;
    const float p = progress(time, b.handoffStart, b.handoffDuration);
    frame.handoff = easeInOutCubic(p);
    const float speed = std::sin(kPi * p);
    frame.whip = speed * speed;
    frame.bootOnly = 1.0F - frame.handoff;
    frame.finished = time >= duration();
}

// --- state animations ---------------------------------------------------------------------

std::size_t stateIndex(AssistantState state) noexcept {
    return static_cast<std::size_t>(state) % kStateCount;
}

StateBlend::StateBlend(float settleSeconds, AssistantState initial) noexcept
    : omega_(springOmega(settleSeconds)), from_(oneHot(initial)), target_(initial) {}

SpringState StateBlend::sample(std::size_t index, float now) const noexcept {
    const float goal = index == stateIndex(target_) ? 1.0F : 0.0F;
    const SpringState start{.value = from_.at(index), .velocity = velocity_.at(index)};
    return springStep(start, goal, omega_, std::max(now - since_, 0.0F));
}

void StateBlend::setTarget(AssistantState state, float now) noexcept {
    if (state == target_) {
        return;
    }
    // Value and velocity of every weight at this instant, so the new fade
    // continues the motion on screen instead of restarting from rest.
    StateWeights value{};
    StateWeights velocity{};
    for (std::size_t i = 0; i < value.size(); ++i) {
        const SpringState current = sample(i, now);
        value.at(i) = current.value;
        velocity.at(i) = current.velocity;
    }
    from_ = value;
    velocity_ = velocity;
    target_ = state;
    since_ = now;
}

StateWeights StateBlend::weights(float now) const noexcept {
    StateWeights out{};
    float total = 0.0F;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out.at(i) = clamp01(sample(i, now).value);
        total += out.at(i);
    }
    // The springs keep the sum at 1; the clamp can only nudge it by a
    // rounding error, which this removes.
    for (float& weight : out) {
        weight /= std::max(total, kSpringRest);
    }
    return out;
}

bool StateBlend::transitioning(float now) const noexcept {
    const std::size_t target = stateIndex(target_);
    for (std::size_t i = 0; i < from_.size(); ++i) {
        const float goal = i == target ? 1.0F : 0.0F;
        if (std::abs(sample(i, now).value - goal) > kSpringRest) {
            return true;
        }
    }
    return false;
}

LevelSmoother::LevelSmoother(float attackSeconds, float releaseSeconds) noexcept
    : attack_(attackSeconds), release_(releaseSeconds) {}

float LevelSmoother::update(float target, float dtSeconds) noexcept {
    value_ = approach(value_, target, dtSeconds, attack_, release_);
    return value_;
}

void followLevels(std::span<float> shown, std::span<const float> target, float dtSeconds,
                  float attackSeconds, float releaseSeconds) noexcept {
    const std::size_t common = std::min(shown.size(), target.size());
    for (std::size_t i = 0; i < common; ++i) {
        shown[i] = approach(shown[i], target[i], dtSeconds, attackSeconds, releaseSeconds);
    }
}

float breathing(float time, float periodSeconds) noexcept {
    if (periodSeconds <= 0.0F) {
        return 0.0F;
    }
    return kHalf - (kHalf * std::cos(kTau * time / periodSeconds));
}

Ripple wakeRipple(float flash, float spread) noexcept {
    const float remaining = clamp01(flash);
    return Ripple{.radius = 1.0F + (spread * easeOutCubic(1.0F - remaining)),
                  .alpha = remaining * remaining};
}

void radialWaveform(std::span<const float> levels, float energy, float time, float floor,
                    std::span<float> out) noexcept {
    const std::size_t spokes = out.size();
    if (spokes == 0) {
        return;
    }
    const float half = static_cast<float>(spokes) * kHalf;
    const float base = clamp01(floor);
    std::size_t i = 0;
    for (float& spoke : out) {
        // 0 at the top (newest sample), 1 at the bottom (oldest).
        const float distance = static_cast<float>(std::min(i, spokes - i)) / half;
        ++i;
        float history = 0.0F;
        if (!levels.empty()) {
            // Fractional age between two samples, newest first.
            const float age = distance * static_cast<float>(levels.size() - 1);
            const auto older = static_cast<std::size_t>(age);
            const std::size_t newer = levels.size() - 1;
            const float a = levels[newer - older];
            const float b = levels[newer - std::min(older + 1, newer)];
            history = perceptualLevel(lerp(a, b, age - static_cast<float>(older)));
        }
        const float shimmer =
            kHalf + (kHalf * std::sin((distance * kTau * kShimmerTurns) + (time * kShimmerSpeed)));
        const float drive = (kHistoryShare * history) + (kEnergyShare * clamp01(energy) * shimmer);
        spoke = base + ((1.0F - base) * clamp01(drive));
    }
}

// --- frame statistics -------------------------------------------------------------------------

void FrameStatsAccumulator::record(double renderMs, double intervalMs, double expectedMs) noexcept {
    ++stats_.frames;
    renders_.at(renderCount_++ % kStatsWindow) = renderMs;
    if (stats_.frames == 1) {
        stats_.renderMs = renderMs;
        return;
    }
    intervals_.at(intervalCount_++ % kStatsWindow) = intervalMs;
    if (expectedMs > 0.0 && intervalMs > expectedMs * kHitchFactor) {
        ++stats_.hitches;
    }
    stats_.renderMs += (renderMs - stats_.renderMs) * smoothing_;
    // The first interval is the first sample, not a blend with zero.
    stats_.intervalMs = stats_.frames == 2
                            ? intervalMs
                            : stats_.intervalMs + ((intervalMs - stats_.intervalMs) * smoothing_);
}

void FrameStatsAccumulator::recordPresentation(std::uint32_t presents, std::uint32_t refreshes,
                                               std::uint32_t refreshesPerFrame,
                                               double refreshHz) noexcept {
    const std::int64_t expected =
        static_cast<std::int64_t>(presents) * std::max<std::int64_t>(refreshesPerFrame, 1);
    stats_.presentedFrames += presents;
    stats_.offCadenceRefreshes +=
        static_cast<std::uint64_t>(std::abs(static_cast<std::int64_t>(refreshes) - expected));
    stats_.refreshHz = refreshHz;
}

void FrameStatsAccumulator::reset() noexcept {
    stats_ = {};
    intervalCount_ = 0;
    renderCount_ = 0;
}

FrameStats FrameStatsAccumulator::stats() const {
    FrameStats out = stats_;
    std::array<double, kStatsWindow> scratch = intervals_;
    const std::span<double> intervals{scratch.data(), std::min(intervalCount_, kStatsWindow)};
    out.intervalP50Ms = percentile(intervals, kMedian);
    out.intervalP95Ms = percentile(intervals, kP95);
    out.intervalP99Ms = percentile(intervals, kP99);
    out.intervalMaxMs = percentile(intervals, 1.0);
    scratch = renders_;
    out.renderP99Ms = percentile({scratch.data(), std::min(renderCount_, kStatsWindow)}, kP99);
    return out;
}

double percentile(std::span<double> values, double share) noexcept {
    if (values.empty()) {
        return 0.0;
    }
    std::ranges::sort(values);
    const double rank = std::ceil(std::clamp(share, 0.0, 1.0) * static_cast<double>(values.size()));
    const auto index = static_cast<std::size_t>(std::max(rank, 1.0)) - 1;
    return values[index];
}

void FrameStatsAccumulator::recordGpu(double gpuMs) noexcept {
    ++stats_.gpuFrames;
    stats_.gpuMs =
        stats_.gpuFrames == 1 ? gpuMs : stats_.gpuMs + ((gpuMs - stats_.gpuMs) * smoothing_);
}

}  // namespace jarvis::overlay::anim

// Design notes:
// - Time, not frames: every public function takes seconds. State carried
//   between frames lives in animator.cpp, and each piece of it composes
//   exactly across frame intervals.
// - The boot particle stream is closed-form (start, delay, travel, swirl)
//   rather than integrated, so a dropped frame cannot change where it lands
//   and a test can ask for t = 0.37 s directly.
// - A skipped boot jumps to the handoff instead of to the end: the panel
//   still glides into its corner, which is what makes a skip feel deliberate
//   rather than broken. Any state change other than Idle does the same.
// - The glitch spikes are derived from the beats in the constructor, so a
//   retimed beat moves its glitch with it and the envelope cannot drift.
// - SplitMix64 instead of <random>: the distributions in <random> are
//   implementation-defined, so "deterministic with a fixed seed" would only
//   hold per standard library.
