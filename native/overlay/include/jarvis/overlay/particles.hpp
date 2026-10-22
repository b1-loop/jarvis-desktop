/// @file particles.hpp
/// @brief The HUD's particle field: curl noise and a fixed-step simulation.
///
/// A few hundred to a few thousand points of light orbit the reactor. Idle
/// they breathe slowly, listening pulls them inward with the microphone
/// level, speaking bursts them outward with the playback level and thinking
/// swirls them. The simulation runs at a fixed internal step, so a frame
/// rate of 15 fps and one of 60 fps produce the same field for the same
/// elapsed time, and a fixed seed produces the same field every run. No
/// ImGui, no Direct3D: the renderer only reads positions.

#pragma once

#include "jarvis/overlay/animation.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace jarvis::overlay::anim {

/// A 2D vector of the particle field, in ring radii (x right, y down).
struct Vec2f {
    float x{0.0F};
    float y{0.0F};
};

/// Divergence-free 2D noise: the curl of an analytic, time-varying stream
/// function (a sum of three products of sines). Particles following it
/// swirl and eddy without bunching up or thinning out, and the analytic
/// form makes it cheap, deterministic and exactly divergence-free.
/// The magnitude never exceeds 1.
[[nodiscard]] Vec2f curlNoise(float x, float y, float time) noexcept;

/// One simulated particle.
struct SimParticle {
    float x{0.0F};
    float y{0.0F};
    float vx{0.0F};
    float vy{0.0F};
    /// Radius it orbits at rest, in ring radii.
    float home{1.0F};
    /// Layer in [-1, 1] for parallax and size; positive is nearer.
    float depth{0.0F};
    float age{0.0F};
    float life{1.0F};
    /// Relative dot size in [0.5, 1.5).
    float size{1.0F};
    /// Per-particle randomness in [0, 1) for how strongly it reacts.
    float jitter{0.0F};
    /// Curl-noise force last sampled for this particle (refreshed every
    /// few steps, see ParticleSimConfig::noiseEvery).
    Vec2f eddy;
};

/// What pushes the particles this frame; each field is in [0, 1] and is
/// already weighted by the state cross-fade and the audio level.
struct FieldForces {
    /// Idle breathing phase: the orbit swells and settles.
    float breathe{0.0F};
    /// Listening: drawn inward with the input level.
    float pull{0.0F};
    /// Speaking: pushed outward with the output level and its onsets.
    float burst{0.0F};
    /// Thinking: orbital speed-up into a spiral.
    float swirl{0.0F};
    /// Strength of the curl-noise eddies.
    float noise{1.0F};
    /// Overall speed multiplier (intensity motion and idle calm).
    float motion{1.0F};
};

/// Tunables of the simulation; the defaults are the shipped look.
struct ParticleSimConfig {
    float minHome{0.55F};
    float maxHome{1.45F};
    /// Particles beyond this radius are respawned at their home orbit.
    float maxRadius{2.3F};
    float minLife{2.5F};
    float maxLife{6.0F};
    /// Orbital speed at the home radius, radians per second.
    float orbitSpeed{0.28F};
    float spring{5.0F};
    float damping{1.6F};
    float noiseScale{1.7F};
    float noiseAccel{0.9F};
    float pullAccel{1.2F};
    float burstAccel{8.0F};
    float swirlBoost{2.2F};
    /// The noise field is re-sampled every this many steps (it drifts
    /// slowly, so a few steps of the same eddy are invisible). Tied to the
    /// step count, not to frames, so the field stays frame-rate independent.
    int noiseEvery{4};
    /// Fixed internal step in seconds.
    float stepSeconds{1.0F / 60.0F};
    /// Most steps one advance() runs; a longer stall skips time instead of
    /// spending a frame catching up.
    int maxStepsPerAdvance{15};
};

/// A seeded, fixed-step particle simulation.
class ParticleSystem {
public:
    /// @p count particles drawn from SplitMix64(@p seed).
    ParticleSystem(std::uint64_t seed, std::size_t count, ParticleSimConfig config = {});

    /// Advance by @p dtSeconds of wall time under @p forces (held constant
    /// for the interval). Runs however many fixed steps have become due.
    /// Only the first @p active particles move (the rest hold still until
    /// they are active again): the idle panel draws a sparse field and
    /// should not pay for simulating the hidden part of it.
    void advance(float dtSeconds, const FieldForces& forces,
                 std::size_t active = static_cast<std::size_t>(-1));

    [[nodiscard]] std::span<const SimParticle> particles() const noexcept { return particles_; }

    /// Opacity of @p particle in [0, 1]: fades in after a respawn, out at
    /// the end of its life and near the edge of the field.
    [[nodiscard]] float alpha(const SimParticle& particle) const noexcept;

    /// Fixed steps run so far.
    [[nodiscard]] std::uint64_t steps() const noexcept { return steps_; }

    [[nodiscard]] const ParticleSimConfig& config() const noexcept { return config_; }

private:
    void step(const FieldForces& forces, std::span<SimParticle> moving);
    void spawn(SimParticle& particle, bool scatterAge);

    ParticleSimConfig config_;
    SplitMix64 random_;
    std::vector<SimParticle> particles_;
    /// Wall time requested so far; double so a day of uptime keeps ms precision.
    double requested_{0.0};
    std::uint64_t steps_{0};
};

}  // namespace jarvis::overlay::anim
