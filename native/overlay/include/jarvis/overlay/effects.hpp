/// @file effects.hpp
/// @brief Richness, glitch and audio-reactivity numbers, without any UI library.
///
/// Three small pieces the renderer reads every frame: how rich the scene is
/// allowed to be (the `overlay_intensity` setting), how much "glitch" the
/// post-processing pass applies right now (zero nearly always, short spikes
/// on boot beats and the wake pulse), and how loud the audio the HUD is
/// showing feels after attack/decay smoothing. All of it is a pure function
/// of time and input, unit-tested without a window.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace jarvis::overlay::anim {

// --- intensity ---------------------------------------------------------------------

/// How rich the HUD is: particle count, bloom strength and motion.
/// "calm" suits a busy desktop, "max" a demo.
enum class Intensity : std::uint8_t { Calm, Normal, Max };

/// Lower-case name as used in settings ("calm", "normal", "max").
[[nodiscard]] std::string_view toString(Intensity intensity) noexcept;

/// Inverse of toString(Intensity); std::nullopt for anything else.
[[nodiscard]] std::optional<Intensity> parseIntensity(std::string_view name) noexcept;

/// Multipliers an Intensity applies. Every field grows (or stays) from
/// Calm to Normal to Max, which the tests check.
struct IntensityScale {
    /// Share of the base particle budget that is simulated and drawn.
    float particles{1.0F};
    /// Multiplier of the bloom strength.
    float bloom{1.0F};
    /// Multiplier of every rotation and drift speed.
    float motion{1.0F};
    /// Mip levels of the bloom chain (wider glow with more levels), 2-4.
    int bloomLevels{3};
};

/// The multipliers for @p intensity; Normal is all ones with 3 bloom levels.
[[nodiscard]] IntensityScale intensityScale(Intensity intensity) noexcept;

/// Particles simulated for @p intensity out of @p base (Normal's count).
[[nodiscard]] std::size_t particleBudget(Intensity intensity, std::size_t base) noexcept;

// --- glitch envelope ------------------------------------------------------------------

/// One spike of the glitch envelope: a 15 ms attack to @c strength, then an
/// exponential decay with time constant @c decay seconds.
struct GlitchSpike {
    float at{0.0F};
    float strength{0.0F};
    float decay{0.08F};
};

/// Value of @p spike at @p time, in [0, spike.strength]; 0 before it starts.
[[nodiscard]] float spikeEnvelope(const GlitchSpike& spike, float time) noexcept;

/// The strongest of @p spikes at @p time, clamped to [0, 1].
[[nodiscard]] float glitchAmount(std::span<const GlitchSpike> spikes, float time) noexcept;

/// Glitch of the wake-word pulse for a flash that decays linearly from 1 to
/// 0 (OverlaySnapshot::wakeFlash): a sharp hit that is gone after a third.
[[nodiscard]] float wakeGlitch(float flash) noexcept;

// --- audio reactivity --------------------------------------------------------------------

/// The smoothed view of an audio level that the scene reacts to.
struct AudioEnergy {
    /// Fast-attack, medium-release perceptual level in [0, 1].
    float level{0.0F};
    /// Instant attack, slow release: how loud it was a moment ago.
    float peak{0.0F};
    /// How much the level jumped above its slow average, in [0, 1]; spikes
    /// on syllable onsets and loud bursts, 0 for a steady tone.
    float onset{0.0F};
};

/// Turns raw RMS samples into AudioEnergy.
///
/// Each part is a one-pole filter whose coefficient is 1 - exp(-dt/tau), so
/// the result depends on elapsed time, not on the frame rate.
class AudioReactor {
public:
    AudioReactor() noexcept = default;

    /// Move towards the perceptual value of @p rms over @p dtSeconds.
    const AudioEnergy& update(float rms, float dtSeconds) noexcept;
    void reset() noexcept;
    [[nodiscard]] const AudioEnergy& energy() const noexcept { return energy_; }

private:
    float slow_{0.0F};
    AudioEnergy energy_;
};

/// Perceptual curve for an RMS level: quiet speech (~0.05 RMS) should
/// already move the HUD visibly. sqrt, clamped to [0, 1].
[[nodiscard]] float perceptualLevel(float rms) noexcept;

/// One step of an attack/release one-pole filter from @p current towards
/// @p target; exact for any @p dtSeconds (see AudioReactor).
[[nodiscard]] float approach(float current, float target, float dtSeconds, float attackSeconds,
                             float releaseSeconds) noexcept;

}  // namespace jarvis::overlay::anim
