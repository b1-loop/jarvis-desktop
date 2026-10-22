#include "jarvis/overlay/animator.hpp"

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/effects.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/particles.hpp"
#include "jarvis/overlay/scene.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

namespace jarvis::overlay::anim {

namespace {

/// Index of each state in StateWeights.
constexpr std::size_t kIdle = 0;
constexpr std::size_t kListening = 1;
constexpr std::size_t kThinking = 2;
constexpr std::size_t kSpeaking = 3;

/// The particle field must not reuse the boot stream's sequence.
constexpr std::uint64_t kFieldSalt = 0x5041525449434C45ULL;

/// Rings and satellites change speed over this long (Rotor time constant).
constexpr float kRotorTau = 0.6F;
constexpr float kGlobeSpeed = 0.25F;
constexpr float kStreamSpeed = 0.9F;
/// Speed boost from the audio: level and onsets spin everything up a little.
constexpr float kLevelSpin = 0.9F;
constexpr float kOnsetSpin = 0.6F;

// Bloom per state: idle dim and breathing, speech brightening with level.
constexpr float kIdleBloom = 0.80F;
constexpr float kIdleBloomBreath = 0.12F;
constexpr float kListeningBloom = 1.0F;
constexpr float kListeningLevelBloom = 0.6F;
constexpr float kThinkingBloom = 1.1F;
constexpr float kSpeakingBloom = 1.0F;
constexpr float kSpeakingLevelBloom = 0.7F;
constexpr float kSpeakingOnsetBloom = 0.4F;
/// The boot is brighter than any state, and the flash brighter still.
constexpr float kBootBloom = 1.3F;
constexpr float kBootFlashBloom = 0.8F;
constexpr float kMaxBloom = 2.0F;

// Particle forces.
constexpr float kPullFloor = 0.25F;
constexpr float kPullGain = 1.1F;
constexpr float kBurstLevel = 0.9F;
constexpr float kBurstOnset = 0.8F;
constexpr float kNoiseIdle = 0.6F;
constexpr float kNoiseBusy = 0.4F;
constexpr float kNoiseSpeech = 0.5F;
constexpr float kIdleMotion = 0.6F;
/// Idle shows only this share of the particles: sparse, but alive.
constexpr float kIdleParticleShare = 0.7F;
/// During the boot the field fades in over this long from the assembly.
constexpr float kBootFieldFade = 1.0F;

/// Transcript rows slide in quickly and fade out a little slower.
constexpr float kTranscriptIn = 0.12F;
constexpr float kTranscriptOut = 0.25F;
/// A transcript reveal this close to its target counts as settled.
constexpr float kSettled = 0.01F;

/// Smoothed levels below this count as silence for frame pacing.
constexpr float kSilentLevel = 0.002F;

float secondsSince(OverlayModel::Clock::time_point then,
                   OverlayModel::Clock::time_point now) noexcept {
    return std::chrono::duration<float>(now - then).count();
}

}  // namespace

float bloomStrength(const StateWeights& weights, const AudioEnergy& energy,
                    float breathing) noexcept {
    const float level = clamp01(energy.level);
    const std::array<float, 4> perState{
        kIdleBloom + (kIdleBloomBreath * clamp01(breathing)),
        kListeningBloom + (kListeningLevelBloom * level),
        kThinkingBloom,
        kSpeakingBloom + (kSpeakingLevelBloom * level) +
            (kSpeakingOnsetBloom * clamp01(energy.onset)),
    };
    float bloom = 0.0F;
    for (std::size_t i = 0; i < perState.size(); ++i) {
        bloom += clamp01(weights.at(i)) * perState.at(i);
    }
    return std::clamp(bloom, 0.0F, kMaxBloom);
}

FieldForces fieldForces(const StateWeights& weights, const AudioEnergy& energy, float breathing,
                        float motion) noexcept {
    const float idle = clamp01(weights.at(kIdle));
    const float speaking = clamp01(weights.at(kSpeaking));
    const float level = clamp01(energy.level);
    return FieldForces{
        .breathe = clamp01(breathing),
        .pull = clamp01(weights.at(kListening)) * clamp01(kPullFloor + (kPullGain * level)),
        .burst = speaking * clamp01((kBurstLevel * level) + (kBurstOnset * clamp01(energy.onset))),
        .swirl = clamp01(weights.at(kThinking)),
        .noise = kNoiseIdle + (kNoiseBusy * (1.0F - idle)) + (kNoiseSpeech * speaking * level),
        .motion = motion * lerp(1.0F, kIdleMotion, idle),
    };
}

bool transcriptWanted(bool hasText, AssistantState state, float secondsIdle) noexcept {
    if (!hasText) {
        return false;
    }
    return state != AssistantState::Idle || secondsIdle < kTranscriptLinger;
}

void OverlayAnimator::setIntensity(Intensity intensity) {
    if (intensity == config_.intensity) {
        return;
    }
    config_.intensity = intensity;
    scale_ = intensityScale(intensity);
    field_ = ParticleSystem(config_.particleSeed ^ kFieldSalt,
                            particleBudget(intensity, config_.particleCount));
}

OverlayAnimator::OverlayAnimator(Clock::time_point epoch, AnimatorConfig config,
                                 BootTimeline timeline)
    : epoch_(epoch),
      config_(config),
      scale_(intensityScale(config.intensity)),
      timeline_(std::move(timeline)),
      bootParticles_(config.particleSeed),
      field_(config.particleSeed ^ kFieldSalt,
             particleBudget(config.intensity, config.particleCount)),
      blend_(config.crossFadeSeconds),
      globe_(kRotorTau),
      stream_(kRotorTau) {
    // Start every layer at a different angle so nothing lines up at t = 0.
    SplitMix64 phases{config.particleSeed};
    for (Rotor& ring : rings_) {
        ring = Rotor{kRotorTau, phases.uniform(0.0F, kTau)};
    }
    auto rotor = satellites_.begin();
    for (const Orbit& orbit : satelliteOrbits().first(kSatelliteCount)) {
        *rotor++ = Rotor{kRotorTau, orbit.phase};
    }
}

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
    const float dt = lastTime_.has_value() ? std::max(time - *lastTime_, 0.0F) : 0.0F;
    lastTime_ = time;

    blend_.setTarget(snapshot.state, time);
    const bool showsLevel =
        snapshot.state == AssistantState::Listening || snapshot.state == AssistantState::Speaking;
    const float latest = showsLevel && !snapshot.levels.empty() ? snapshot.levels.back() : 0.0F;

    frame_.time = time;
    frame_.boot = bootFrame(snapshot, now);
    frame_.weights = blend_.weights(time);
    frame_.energy = reactor_.update(latest, dt);
    frame_.level = frame_.energy.level;
    frame_.breathing = breathing(time, config_.breathingPeriod);
    frame_.ripple = wakeRipple(snapshot.wakeFlash, config_.rippleSpread);
    frame_.motion = scale_.motion;
    advanceRotors(dt);
    advanceEffects(snapshot, dt);
    advanceTranscript(snapshot, dt);
    return frame_;
}

void OverlayAnimator::advanceTranscript(const OverlaySnapshot& snapshot, float dt) {
    const bool idle = snapshot.state == AssistantState::Idle;
    if (idle && !wasIdle_) {
        idleSince_ = frame_.time;
    }
    wasIdle_ = idle;
    const bool hasText = !snapshot.transcript.empty() || !snapshot.response.empty();
    transcriptWanted_ = transcriptWanted(hasText, snapshot.state, frame_.time - idleSince_);
    frame_.transcript = approach(frame_.transcript, transcriptWanted_ ? 1.0F : 0.0F, dt,
                                 kTranscriptIn, kTranscriptOut);
}

void OverlayAnimator::advanceRotors(float dt) {
    const AudioEnergy& energy = frame_.energy;
    const float audio = 1.0F + (kLevelSpin * energy.level) + (kOnsetSpin * energy.onset);
    const float speed = stateSpeedFactor(frame_.weights) * audio * scale_.motion;
    std::size_t ring = 0;
    for (const RingLayer& layer : ringLayers()) {
        rings_.at(ring).advance(layer.speed * speed, dt);
        frame_.ringAngles.at(ring) = rings_.at(ring).angle();
        ++ring;
    }
    globe_.advance(kGlobeSpeed * speed, dt);
    frame_.globeAngle = globe_.angle();
    std::size_t satellite = 0;
    for (const Orbit& orbit : satelliteOrbits().first(kSatelliteCount)) {
        satellites_.at(satellite).advance(orbit.speed * speed, dt);
        frame_.satelliteAngles.at(satellite) = satellites_.at(satellite).angle();
        ++satellite;
    }
    stream_.advance(kStreamSpeed * speed, dt);
    frame_.streamPhase = stream_.angle();
    frame_.camera = driftingCamera(frame_.time, scale_.motion);
}

void OverlayAnimator::advanceEffects(const OverlaySnapshot& snapshot, float dt) {
    const float idle = frame_.weights.at(kIdle);
    float bloom = bloomStrength(frame_.weights, frame_.energy, frame_.breathing);
    float glitch = wakeGlitch(snapshot.wakeFlash);
    float share = lerp(1.0F, kIdleParticleShare, idle);
    if (frame_.boot.has_value()) {
        const BootFrame& boot = *frame_.boot;
        const float bootBloom = kBootBloom + (kBootFlashBloom * boot.flash);
        bloom = lerp(bootBloom, bloom, boot.handoff);
        glitch = std::max(glitch, boot.glitch);
        const float assembled =
            progress(boot.time, timeline_.beats().assemblyStart, kBootFieldFade);
        share *= std::max(assembled, boot.handoff);
    }
    frame_.bloom = bloom * scale_.bloom;
    frame_.glitch = clamp01(glitch);
    frame_.particleShare = share;
    frame_.forces = fieldForces(frame_.weights, frame_.energy, frame_.breathing, scale_.motion);
    const auto active = static_cast<std::size_t>(
        std::ceil(static_cast<float>(field_.particles().size()) * frame_.particleShare));
    field_.advance(dt, frame_.forces, active);
}

Pace OverlayAnimator::pace(const OverlaySnapshot& snapshot) const noexcept {
    const bool busy = frame_.boot.has_value() || blend_.transitioning(frame_.time) ||
                      snapshot.wakeFlash > 0.0F || snapshot.state != AssistantState::Idle ||
                      reactor_.energy().level > kSilentLevel ||
                      std::abs(frame_.transcript - (transcriptWanted_ ? 1.0F : 0.0F)) > kSettled;
    return busy ? Pace::Active : Pace::Idle;
}

}  // namespace jarvis::overlay::anim

// Design notes:
// - One speed multiplier (state x audio x intensity) drives every rotor,
//   and each rotor eases towards it with the same time constant, so the
//   whole HUD speeds up and settles together instead of layer by layer.
// - The particle field keeps simulating in idle but only part of it is
//   drawn; switching to Listening then shows a field that was already in
//   motion rather than one that starts from rest.
// - Bloom, glitch and particle share are decided here, not in the
//   renderer, so their bounds are tested and the renderer stays a
//   translation layer.
