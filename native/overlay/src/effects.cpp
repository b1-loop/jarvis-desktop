#include "jarvis/overlay/effects.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <string_view>

namespace jarvis::overlay::anim {

namespace {

constexpr std::array<std::string_view, 3> kIntensityNames{"calm", "normal", "max"};

constexpr IntensityScale kCalm{
    .particles = 0.35F, .bloom = 0.65F, .motion = 0.6F, .bloomLevels = 2};
constexpr IntensityScale kNormal{
    .particles = 1.0F, .bloom = 1.0F, .motion = 1.0F, .bloomLevels = 3};
constexpr IntensityScale kMax{.particles = 2.0F, .bloom = 1.35F, .motion = 1.3F, .bloomLevels = 4};

/// A spike rises linearly over this many seconds before it decays.
constexpr float kSpikeAttack = 0.015F;
/// The wake glitch follows flash^3: a hard hit that fades long before the ripple.
constexpr float kWakeGlitchStrength = 0.65F;

// Audio smoothing time constants, seconds.
constexpr float kLevelAttack = 0.03F;
constexpr float kLevelRelease = 0.25F;
constexpr float kPeakRelease = 0.9F;
constexpr float kSlowTau = 0.45F;
/// A jump of a quarter of full scale above the slow average is a full onset.
constexpr float kOnsetGain = 4.0F;

float clampUnit(float value) noexcept {
    return std::clamp(value, 0.0F, 1.0F);
}

/// 1 - exp(-dt/tau), with tau <= 0 meaning "jump".
float coefficient(float dtSeconds, float tauSeconds) noexcept {
    if (tauSeconds <= 0.0F) {
        return 1.0F;
    }
    return 1.0F - std::exp(-dtSeconds / tauSeconds);
}

}  // namespace

// --- intensity ---------------------------------------------------------------------

std::string_view toString(Intensity intensity) noexcept {
    switch (intensity) {
        case Intensity::Calm:
            return kIntensityNames.at(0);
        case Intensity::Max:
            return kIntensityNames.at(2);
        case Intensity::Normal:
            break;
    }
    return kIntensityNames.at(1);
}

std::optional<Intensity> parseIntensity(std::string_view name) noexcept {
    for (const Intensity intensity : {Intensity::Calm, Intensity::Normal, Intensity::Max}) {
        if (toString(intensity) == name) {
            return intensity;
        }
    }
    return std::nullopt;
}

IntensityScale intensityScale(Intensity intensity) noexcept {
    switch (intensity) {
        case Intensity::Calm:
            return kCalm;
        case Intensity::Max:
            return kMax;
        case Intensity::Normal:
            break;
    }
    return kNormal;
}

std::size_t particleBudget(Intensity intensity, std::size_t base) noexcept {
    const float share = intensityScale(intensity).particles;
    return static_cast<std::size_t>(std::lround(static_cast<float>(base) * share));
}

// --- glitch envelope ------------------------------------------------------------------

float spikeEnvelope(const GlitchSpike& spike, float time) noexcept {
    const float since = time - spike.at;
    if (since < 0.0F || spike.strength <= 0.0F) {
        return 0.0F;
    }
    if (since < kSpikeAttack) {
        return spike.strength * (since / kSpikeAttack);
    }
    if (spike.decay <= 0.0F) {
        return 0.0F;
    }
    return spike.strength * std::exp(-(since - kSpikeAttack) / spike.decay);
}

float glitchAmount(std::span<const GlitchSpike> spikes, float time) noexcept {
    float strongest = 0.0F;
    for (const GlitchSpike& spike : spikes) {
        strongest = std::max(strongest, spikeEnvelope(spike, time));
    }
    return clampUnit(strongest);
}

float wakeGlitch(float flash) noexcept {
    const float f = clampUnit(flash);
    return kWakeGlitchStrength * f * f * f;
}

// --- audio reactivity --------------------------------------------------------------------

float perceptualLevel(float rms) noexcept {
    return std::sqrt(clampUnit(rms));
}

float approach(float current, float target, float dtSeconds, float attackSeconds,
               float releaseSeconds) noexcept {
    if (dtSeconds <= 0.0F) {
        return current;
    }
    const float tau = target > current ? attackSeconds : releaseSeconds;
    return current + ((target - current) * coefficient(dtSeconds, tau));
}

const AudioEnergy& AudioReactor::update(float rms, float dtSeconds) noexcept {
    const float target = perceptualLevel(rms);
    energy_.level = approach(energy_.level, target, dtSeconds, kLevelAttack, kLevelRelease);
    energy_.peak = std::max(target, approach(energy_.peak, target, dtSeconds, 0.0F, kPeakRelease));
    slow_ = approach(slow_, energy_.level, dtSeconds, kSlowTau, kSlowTau);
    energy_.onset = clampUnit((energy_.level - slow_) * kOnsetGain);
    return energy_;
}

void AudioReactor::reset() noexcept {
    slow_ = 0.0F;
    energy_ = {};
}

}  // namespace jarvis::overlay::anim

// Design notes:
// - Intensity is three fixed presets rather than a free 0-1 slider: each
//   preset was checked against the frame budget (native/overlay/README.md),
//   and a slider would let a setting quietly exceed it.
// - The glitch is a max of spikes, not a sum: two beats close together read
//   as one hit, and the amount stays bounded without a soft clip.
// - Onset is the level minus its own slow average, so a loud but steady
//   tone settles to 0 while every syllable start kicks the particles; the
//   HUD then reacts to speech rhythm, not only to loudness.
