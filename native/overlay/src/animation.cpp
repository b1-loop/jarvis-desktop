#include "jarvis/overlay/animation.hpp"

#include "jarvis/overlay/model.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace jarvis::overlay::anim {

namespace {

constexpr float kHalf = 0.5F;
constexpr std::size_t kStateCount = 4;
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

// Boot: the centre flashes white-hot, then settles to an ember behind the rings.
constexpr float kFlashPeak = 0.12F;
constexpr float kFlashSettle = 0.7F;
constexpr float kFlashEmber = 0.18F;
constexpr std::array<Keyframe, 3> kFlashTrack{{
    {.time = 0.0F, .value = 0.0F, .easing = Easing::Linear},
    {.time = kFlashPeak, .value = 1.0F, .easing = Easing::OutCubic},
    {.time = kFlashSettle, .value = kFlashEmber, .easing = Easing::InOutCubic},
}};
/// One revolution of the radial sweep every this many seconds.
constexpr float kSweepPeriod = 1.25F;
/// The grid fades in and out over these fractions of a second.
constexpr float kGridFadeIn = 0.15F;
constexpr float kGridFadeOut = 0.35F;

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

/// Smoothed levels below this count as silence for frame pacing.
constexpr float kSilentLevel = 0.002F;

float secondsSince(OverlayModel::Clock::time_point then,
                   OverlayModel::Clock::time_point now) noexcept {
    return std::chrono::duration<float>(now - then).count();
}

StateWeights oneHot(AssistantState state) noexcept {
    StateWeights weights{};
    weights.at(stateIndex(state)) = 1.0F;
    return weights;
}

/// Perceptual curve for an RMS level: quiet speech is ~0.05 RMS, which
/// should already move the ring visibly.
float perceptual(float level) noexcept {
    return std::sqrt(clamp01(level));
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

// --- particles --------------------------------------------------------------------------

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

BootTimeline::BootTimeline(BootBeats beats, std::string title)
    : beats_(beats), title_(std::move(title)) {}

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

BootFrame BootTimeline::evaluate(float time) const noexcept {
    const BootBeats& b = beats_;
    BootFrame frame;
    frame.time = time;
    frame.ignite = easeOutCubic(progress(time, 0.0F, b.igniteEnd));
    frame.flash = sampleTrack(kFlashTrack, time);
    frame.sweepAngle = (kTau * time / kSweepPeriod) - (kTau / 4.0F);
    frame.spin = easeOutCubic(progress(time, 0.0F, duration()));
    frame.particleTime = time - b.particlesStart;

    const float scan = progress(time, b.scanStart, b.scanDuration);
    frame.scan = easeInOutCubic(scan);
    const bool scanning = time >= b.scanStart && time <= b.scanStart + b.scanDuration;
    frame.scanAlpha = scanning ? std::sin(std::numbers::pi_v<float> * scan) : 0.0F;

    const float gridEnd = b.gridStart + b.gridDuration;
    frame.grid = progress(time, b.gridStart, kGridFadeIn) *
                 (1.0F - progress(time, gridEnd - kGridFadeOut, kGridFadeOut));

    const std::size_t length = title_.size();
    if (time >= b.titleStart && b.charInterval > 0.0F) {
        const auto typed = static_cast<std::size_t>((time - b.titleStart) / b.charInterval) + 1;
        frame.titleChars = std::min(typed, length);
    }
    const float typedAt = b.titleStart + (static_cast<float>(length - 1) * b.charInterval);
    const bool typing = time >= b.titleStart && frame.titleChars < length;
    const bool blinkOn =
        b.cursorBlink > 0.0F && std::fmod(time - typedAt, 2.0F * b.cursorBlink) < b.cursorBlink;
    frame.cursorVisible = time < b.handoffStart && (typing || (time >= typedAt && blinkOn));

    frame.subtitle = easeOutCubic(progress(time, b.subtitleStart, b.subtitleDuration));
    frame.handoff = easeInOutCubic(progress(time, b.handoffStart, b.handoffDuration));
    frame.bootOnly = 1.0F - frame.handoff;
    frame.finished = time >= duration();
    return frame;
}

// --- state animations ---------------------------------------------------------------------

std::size_t stateIndex(AssistantState state) noexcept {
    return static_cast<std::size_t>(state) % kStateCount;
}

StateBlend::StateBlend(float durationSeconds, AssistantState initial) noexcept
    : duration_(std::max(durationSeconds, 0.0F)),
      from_(oneHot(initial)),
      target_(initial),
      since_(-duration_) {}

void StateBlend::setTarget(AssistantState state, float now) noexcept {
    if (state == target_) {
        return;
    }
    from_ = weights(now);
    target_ = state;
    since_ = now;
}

StateWeights StateBlend::weights(float now) const noexcept {
    const float p = easeInOutCubic(progress(now, since_, duration_));
    StateWeights out{};
    const std::size_t target = stateIndex(target_);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const float goal = i == target ? 1.0F : 0.0F;
        out.at(i) = lerp(from_.at(i), goal, p);
    }
    return out;
}

bool StateBlend::transitioning(float now) const noexcept {
    return now < since_ + duration_;
}

LevelSmoother::LevelSmoother(float attackSeconds, float releaseSeconds) noexcept
    : attack_(attackSeconds), release_(releaseSeconds) {}

float LevelSmoother::update(float target, float dtSeconds) noexcept {
    if (dtSeconds <= 0.0F) {
        return value_;
    }
    const float tau = target > value_ ? attack_ : release_;
    if (tau <= 0.0F) {
        value_ = target;
        return value_;
    }
    value_ += (target - value_) * (1.0F - std::exp(-dtSeconds / tau));
    return value_;
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
            const auto age = std::lround(distance * static_cast<float>(levels.size() - 1));
            history = perceptual(*std::next(levels.rbegin(), age));
        }
        const float shimmer =
            kHalf + (kHalf * std::sin((distance * kTau * kShimmerTurns) + (time * kShimmerSpeed)));
        const float drive = (kHistoryShare * history) + (kEnergyShare * clamp01(energy) * shimmer);
        spoke = base + ((1.0F - base) * clamp01(drive));
    }
}

// --- the animator ---------------------------------------------------------------------------

OverlayAnimator::OverlayAnimator(Clock::time_point epoch, AnimatorConfig config,
                                 BootTimeline timeline)
    : epoch_(epoch),
      config_(config),
      timeline_(std::move(timeline)),
      particles_(config.particleSeed),
      blend_(config.crossFadeSeconds),
      smoother_(config.attackSeconds, config.releaseSeconds) {}

std::optional<BootFrame> OverlayAnimator::bootFrame(const OverlaySnapshot& snapshot,
                                                    Clock::time_point now) const noexcept {
    if (!snapshot.bootStartedAt.has_value()) {
        return std::nullopt;
    }
    const Clock::time_point start = *snapshot.bootStartedAt;
    std::optional<float> skippedAfter;
    if (snapshot.bootSkippedAt.has_value()) {
        skippedAfter = secondsSince(start, *snapshot.bootSkippedAt);
    }
    const float time = timeline_.timelineTime(secondsSince(start, now), skippedAfter);
    BootFrame frame = timeline_.evaluate(time);
    if (frame.finished) {
        return std::nullopt;
    }
    return frame;
}

const AnimationFrame& OverlayAnimator::advance(const OverlaySnapshot& snapshot,
                                               Clock::time_point now) {
    const float time = secondsSince(epoch_, now);
    const float dt = lastTime_.has_value() ? time - *lastTime_ : 0.0F;
    lastTime_ = time;

    blend_.setTarget(snapshot.state, time);
    const bool showsLevel =
        snapshot.state == AssistantState::Listening || snapshot.state == AssistantState::Speaking;
    const float latest = snapshot.levels.empty() ? 0.0F : snapshot.levels.back();

    frame_.time = time;
    frame_.boot = bootFrame(snapshot, now);
    frame_.weights = blend_.weights(time);
    frame_.level = smoother_.update(showsLevel ? perceptual(latest) : 0.0F, dt);
    frame_.breathing = breathing(time, config_.breathingPeriod);
    frame_.ripple = wakeRipple(snapshot.wakeFlash, config_.rippleSpread);
    return frame_;
}

Pace OverlayAnimator::pace(const OverlaySnapshot& snapshot) const noexcept {
    const bool busy = frame_.boot.has_value() || blend_.transitioning(frame_.time) ||
                      snapshot.wakeFlash > 0.0F || snapshot.state != AssistantState::Idle ||
                      smoother_.value() > kSilentLevel;
    return busy ? Pace::Active : Pace::Idle;
}

// --- frame statistics -------------------------------------------------------------------------

void FrameStatsAccumulator::record(double renderMs, double intervalMs) noexcept {
    ++stats_.frames;
    if (stats_.frames == 1) {
        stats_.renderMs = renderMs;
        return;
    }
    stats_.renderMs += (renderMs - stats_.renderMs) * smoothing_;
    // The first interval is the first sample, not a blend with zero.
    stats_.intervalMs = stats_.frames == 2
                            ? intervalMs
                            : stats_.intervalMs + ((intervalMs - stats_.intervalMs) * smoothing_);
}

}  // namespace jarvis::overlay::anim

// Design notes:
// - Time, not frames: every public function takes seconds. The only state
//   carried between frames is the level smoother, and its coefficient is
//   1 - exp(-dt/tau), which composes exactly: two half-steps equal one step.
// - Particles are closed-form (start, delay, travel, swirl) rather than
//   integrated, so a dropped frame cannot change where they land and a test
//   can ask for t = 0.37 s directly.
// - A skipped boot jumps to the handoff instead of to the end: the panel
//   still glides into its corner, which is what makes a skip feel deliberate
//   rather than broken.
// - SplitMix64 instead of <random>: the distributions in <random> are
//   implementation-defined, so "deterministic with a fixed seed" would only
//   hold per standard library.
