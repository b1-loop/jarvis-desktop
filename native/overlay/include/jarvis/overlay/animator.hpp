/// @file animator.hpp
/// @brief Combines the boot timeline, the state cross-fade, audio
/// reactivity, the rotors and the particle field into one frame.
///
/// The UI thread owns one OverlayAnimator and calls advance() once per
/// frame. It is the only place that carries state between frames (the
/// smoothers, rotors and the particle simulation), and every piece of that
/// state advances by the real elapsed time in a frame-rate independent way.
/// No ImGui, no Direct3D: the tests drive it with a synthetic clock.

#pragma once

#include "jarvis/overlay/animation.hpp"
#include "jarvis/overlay/effects.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/particles.hpp"
#include "jarvis/overlay/scene.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace jarvis::overlay::anim {

/// Number of orbiting satellites (satelliteOrbits().size()).
inline constexpr std::size_t kSatelliteCount = 4;

/// Tunables of the state animations.
struct AnimatorConfig {
    /// Settle time (99 %) of the state cross-fade spring.
    float crossFadeSeconds = 0.35F;
    float attackSeconds = 0.05F;
    float releaseSeconds = 0.30F;
    float breathingPeriod = 5.5F;
    float rippleSpread = 0.5F;
    std::uint64_t particleSeed = 0x4A415256ULL;  // fixed so every boot looks the same
    /// Particles at Normal intensity; Calm and Max scale it (particleBudget).
    std::size_t particleCount = 900;
    Intensity intensity = Intensity::Normal;
};

/// Everything time-dependent the renderer needs for one frame.
struct AnimationFrame {
    /// Seconds since the animator was created; drives every rotation.
    float time{0.0F};
    /// Present while the boot sequence is playing.
    std::optional<BootFrame> boot;
    StateWeights weights{1.0F, 0.0F, 0.0F, 0.0F};
    /// Smoothed perceptual level of whichever source the state is showing.
    float level{0.0F};
    /// The level history (oldest first) followed per sample with an
    /// attack/release envelope: what the waveforms and telemetry draw, so
    /// they move every frame instead of stepping with each pushed level.
    std::vector<float> levels;
    /// Level, peak and onset of that source (AudioReactor).
    AudioEnergy energy;
    float breathing{0.0F};
    Ripple ripple;
    /// The drifting camera the globe and satellites are projected through.
    Camera camera;
    /// Rotation of the globe about its axis, radians.
    float globeAngle{0.0F};
    /// Rotation of each ring layer (ringLayers() order), radians.
    std::array<float, kRingCount> ringAngles{};
    /// Position of each satellite along its orbit, radians.
    std::array<float, kSatelliteCount> satelliteAngles{};
    /// Phase of the data streams flowing along the rings, radians.
    float streamPhase{0.0F};
    /// Post-processing glitch (chromatic aberration, scanlines) in [0, 1].
    float glitch{0.0F};
    /// Bloom strength for this frame (already scaled by the intensity).
    float bloom{0.0F};
    /// Forces the particle field was advanced with this frame.
    FieldForces forces;
    /// Share of the simulated particles drawn this frame (sparse when idle).
    float particleShare{1.0F};
    /// Multiplier of every speed (the intensity's motion).
    float motion{1.0F};
    /// How far the transcript rows are revealed, 0 (hidden: the idle readout
    /// shows instead) to 1; eases in and out.
    float transcript{0.0F};
};

/// Seconds the transcript stays up after Jarvis went back to Idle.
inline constexpr float kTranscriptLinger = 8.0F;

/// Whether the transcript rows should show: there is text, and Jarvis is
/// busy or went idle less than kTranscriptLinger seconds ago (@p secondsIdle;
/// ignored while not Idle).
[[nodiscard]] bool transcriptWanted(bool hasText, AssistantState state, float secondsIdle) noexcept;

/// How fast the UI thread should draw.
enum class Pace : std::uint8_t { Active, Idle };

/// Bloom strength for the state @p weights, @p energy and idle
/// @p breathing (both in [0, 1]); always in [0, 2] before the intensity
/// multiplier. Idle is dim, speech brightens with level and onsets.
[[nodiscard]] float bloomStrength(const StateWeights& weights, const AudioEnergy& energy,
                                  float breathing) noexcept;

/// Particle forces for the state @p weights and @p energy at @p breathing.
[[nodiscard]] FieldForces fieldForces(const StateWeights& weights, const AudioEnergy& energy,
                                      float breathing, float motion) noexcept;

/// The UI thread's animation state: one per window, advanced once per frame.
class OverlayAnimator {
public:
    using Clock = OverlayModel::Clock;

    explicit OverlayAnimator(Clock::time_point epoch, AnimatorConfig config = {},
                             BootTimeline timeline = BootTimeline{});

    /// Advance to @p now for @p snapshot and return the frame to draw.
    const AnimationFrame& advance(const OverlaySnapshot& snapshot, Clock::time_point now);

    /// Active while booting, cross-fading, rippling, sounding, ramping
    /// the bloom or not idle; Idle lets the window drop to its low rate.
    [[nodiscard]] Pace pace(const OverlaySnapshot& snapshot) const noexcept;

    [[nodiscard]] const AnimationFrame& frame() const noexcept { return frame_; }
    [[nodiscard]] const BootTimeline& timeline() const noexcept { return timeline_; }
    [[nodiscard]] const ParticleField& particles() const noexcept { return bootParticles_; }
    [[nodiscard]] const ParticleSystem& field() const noexcept { return field_; }
    [[nodiscard]] const AnimatorConfig& config() const noexcept { return config_; }
    [[nodiscard]] const IntensityScale& scale() const noexcept { return scale_; }

    /// Switch the richness preset while running (settings page). Rotors and
    /// state keep going; the particle field is reseeded with the preset's
    /// particle count, which reads as the field thinning out or filling in.
    void setIntensity(Intensity intensity);

private:
    [[nodiscard]] std::optional<BootFrame> bootFrame(const OverlaySnapshot& snapshot,
                                                     Clock::time_point now) const noexcept;
    void advanceRotors(float dt);
    void advanceEffects(const OverlaySnapshot& snapshot, float dt);
    void advanceTranscript(const OverlaySnapshot& snapshot, float dt);
    void advanceLevels(const OverlaySnapshot& snapshot, float dt);

    Clock::time_point epoch_;
    AnimatorConfig config_;
    IntensityScale scale_;
    BootTimeline timeline_;
    ParticleField bootParticles_;
    ParticleSystem field_;
    StateBlend blend_;
    AudioReactor reactor_;
    std::array<Rotor, kRingCount> rings_;
    Rotor globe_;
    std::array<Rotor, kSatelliteCount> satellites_;
    Rotor stream_;
    std::optional<float> lastTime_;
    /// When the state last became Idle (animator seconds).
    float idleSince_{0.0F};
    bool wasIdle_{true};
    bool transcriptWanted_{false};
    /// The intensity's bloom multiplier as shown: ramps to scale_.bloom
    /// instead of stepping when the preset changes.
    float bloomGain_{1.0F};
    AnimationFrame frame_;
};

}  // namespace jarvis::overlay::anim
