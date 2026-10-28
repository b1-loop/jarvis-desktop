#include "jarvis/overlay/particles.hpp"

#include "jarvis/overlay/animation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace jarvis::overlay::anim {

namespace {

/// One term of the stream function: amplitude * sin(fx x + wx t + px) * sin(fy y + wy t + py).
struct NoiseTerm {
    float amplitude;
    float fx;
    float fy;
    float wx;
    float wy;
    float px;
    float py;
};

constexpr std::array<NoiseTerm, 3> kNoise{{
    {.amplitude = 0.50F, .fx = 1.3F, .fy = 1.1F, .wx = 0.21F, .wy = -0.17F, .px = 0.4F, .py = 1.9F},
    {.amplitude = 0.30F, .fx = 2.7F, .fy = 2.3F, .wx = -0.33F, .wy = 0.29F, .px = 2.2F, .py = 0.7F},
    {.amplitude = 0.20F, .fx = 4.9F, .fy = 5.3F, .wx = 0.47F, .wy = 0.41F, .px = 4.1F, .py = 3.3F},
}};

/// Sum over terms of amplitude * max(fx, fy): the largest each derivative can be.
constexpr float noiseBound() noexcept {
    float bound = 0.0F;
    for (const NoiseTerm& term : kNoise) {
        bound += term.amplitude * std::max(term.fx, term.fy);
    }
    return bound;
}
constexpr float kNoiseNorm = 1.0F / noiseBound();

// Breathing swells the orbit by a few percent; pull shrinks it a lot.
constexpr float kBreatheSwell = 0.08F;
constexpr float kPullShrink = 0.5F;
/// How hard a particle is steered towards its orbital speed, 1/s.
constexpr float kOrbitSteer = 2.0F;
/// The noise field drifts slower than wall time.
constexpr float kNoiseTimeScale = 0.3F;
/// Burst strength varies per particle between these factors.
constexpr float kBurstMin = 0.6F;
constexpr float kBurstSpread = 0.8F;
constexpr float kMinRadius = 1e-4F;
constexpr float kFadeIn = 0.4F;
constexpr float kFadeOut = 0.6F;
/// Particles start fading at this share of the maximum radius.
constexpr float kEdgeFade = 0.75F;
constexpr float kMinSize = 0.5F;
constexpr float kMaxSize = 1.5F;
/// Guards the step count against a requested time that is a hair short of a step.
constexpr double kStepEpsilon = 1e-3;

}  // namespace

Vec2f curlNoise(float x, float y, float time) noexcept {
    // psi = sum a sin(A) sin(B); curl psi = (d psi / dy, -d psi / dx).
    float dPsiDx = 0.0F;
    float dPsiDy = 0.0F;
    for (const NoiseTerm& term : kNoise) {
        const float a = (term.fx * x) + (term.wx * time) + term.px;
        const float b = (term.fy * y) + (term.wy * time) + term.py;
        dPsiDx += term.amplitude * term.fx * std::cos(a) * std::sin(b);
        dPsiDy += term.amplitude * term.fy * std::sin(a) * std::cos(b);
    }
    return Vec2f{.x = dPsiDy * kNoiseNorm, .y = -dPsiDx * kNoiseNorm};
}

ParticleSystem::ParticleSystem(std::uint64_t seed, std::size_t count, ParticleSimConfig config)
    : config_(config), random_(seed), particles_(count) {
    for (SimParticle& particle : particles_) {
        spawn(particle, true);
    }
}

void ParticleSystem::spawn(SimParticle& particle, bool scatterAge) {
    // Drawn in a fixed order so the sequence for a seed never changes.
    const float angle = random_.uniform(0.0F, kTau);
    particle.home = random_.uniform(config_.minHome, config_.maxHome);
    particle.depth = random_.uniform(-1.0F, 1.0F);
    particle.life = random_.uniform(config_.minLife, config_.maxLife);
    particle.size = random_.uniform(kMinSize, kMaxSize);
    particle.jitter = random_.nextUnit();
    const float ageShare = random_.nextUnit();
    particle.age = scatterAge ? ageShare * particle.life : 0.0F;
    particle.x = particle.home * std::cos(angle);
    particle.y = particle.home * std::sin(angle);
    // A respawn starts in place: no streak from where it died.
    particle.prevX = particle.x;
    particle.prevY = particle.y;
    const float speed = config_.orbitSpeed * particle.home;
    particle.vx = -std::sin(angle) * speed;
    particle.vy = std::cos(angle) * speed;
    particle.eddy = {};
}

void ParticleSystem::advance(float dtSeconds, const FieldForces& forces, std::size_t active) {
    if (dtSeconds <= 0.0F || config_.stepSeconds <= 0.0F) {
        return;
    }
    requested_ += static_cast<double>(dtSeconds);
    const auto due = static_cast<std::uint64_t>((requested_ / config_.stepSeconds) + kStepEpsilon);
    if (due <= steps_) {
        return;
    }
    const auto limit = static_cast<std::uint64_t>(std::max(config_.maxStepsPerAdvance, 1));
    if (due - steps_ > limit) {
        steps_ = due - limit;  // a stall: skip the backlog rather than catch up
    }
    const std::span<SimParticle> moving =
        std::span{particles_}.first(std::min(active, particles_.size()));
    while (steps_ < due) {
        step(forces, moving);
        ++steps_;
    }
}

void ParticleSystem::step(const FieldForces& forces, std::span<SimParticle> moving) {
    const float h = config_.stepSeconds;
    const float keep = std::exp(-config_.damping * h);
    const float noiseTime = static_cast<float>(steps_) * h * kNoiseTimeScale;
    const auto every = static_cast<std::uint64_t>(std::max(config_.noiseEvery, 1));
    const bool resample = steps_ % every == 0;
    const float swell = 1.0F + (kBreatheSwell * (forces.breathe - 0.5F));
    const float shrink = 1.0F - (kPullShrink * clamp01(forces.pull));
    const float orbit =
        config_.orbitSpeed * (1.0F + (config_.swirlBoost * forces.swirl)) * forces.motion;
    for (SimParticle& p : moving) {
        const float radius = std::max(std::hypot(p.x, p.y), kMinRadius);
        const float ux = p.x / radius;
        const float uy = p.y / radius;
        const float burst =
            config_.burstAccel * forces.burst * (kBurstMin + (kBurstSpread * p.jitter));
        const float radial = (config_.spring * ((p.home * swell * shrink) - radius)) -
                             (config_.pullAccel * forces.pull) + burst;
        const float tangential = (orbit * p.home) - ((p.vx * -uy) + (p.vy * ux));
        if (resample) {
            p.eddy = curlNoise(p.x * config_.noiseScale, p.y * config_.noiseScale, noiseTime);
        }
        const Vec2f eddy = p.eddy;
        const float noise = config_.noiseAccel * forces.noise;
        p.vx = (p.vx + (h * ((radial * ux) - (kOrbitSteer * tangential * uy) + (eddy.x * noise)))) *
               keep;
        p.vy = (p.vy + (h * ((radial * uy) + (kOrbitSteer * tangential * ux) + (eddy.y * noise)))) *
               keep;
        p.prevX = p.x;
        p.prevY = p.y;
        p.x += p.vx * h;
        p.y += p.vy * h;
        p.age += h;
        if (p.age >= p.life || radius > config_.maxRadius) {
            spawn(p, false);
        }
    }
}

float ParticleSystem::blend() const noexcept {
    if (config_.stepSeconds <= 0.0F) {
        return 1.0F;
    }
    const double stepped = static_cast<double>(steps_) * config_.stepSeconds;
    return clamp01(static_cast<float>((requested_ - stepped) / config_.stepSeconds));
}

Vec2f ParticleSystem::position(const SimParticle& particle) const noexcept {
    const float t = blend();
    return Vec2f{.x = lerp(particle.prevX, particle.x, t),
                 .y = lerp(particle.prevY, particle.y, t)};
}

float ParticleSystem::alpha(const SimParticle& particle) const noexcept {
    const float in = smoothstep(0.0F, kFadeIn, particle.age);
    const float out = 1.0F - smoothstep(particle.life - kFadeOut, particle.life, particle.age);
    const float radius = std::hypot(particle.x, particle.y);
    const float edge = 1.0F - smoothstep(config_.maxRadius * kEdgeFade, config_.maxRadius, radius);
    return clamp01(in * out * edge);
}

}  // namespace jarvis::overlay::anim

// Design notes:
// - Fixed step plus a step counter derived from the total requested time
//   (not from accumulated remainders) makes the step count depend only on
//   elapsed time: 120 frames of 1/60 s and 30 frames of 1/15 s run exactly
//   the same 120 steps, so the field is frame-rate independent by
//   construction, not by tolerance.
// - Drawing interpolates between the last two steps (blend()), the
//   classic fixed-timestep render: the simulation stays deterministic and
//   frame-rate independent, and the picture moves every frame by exactly
//   the time that passed, at 30, 60, 72 or 144 fps alike. The cost is
//   drawing up to one step (16 ms) behind the simulation.
// - The forces act through targets (orbit radius, orbital speed) and
//   springs rather than raw accelerations, so whatever the audio does the
//   field returns to its calm orbit and never explodes.
// - Curl noise from an analytic stream function instead of gradient noise:
//   it is exactly divergence-free, needs no permutation tables, and three
//   sine products per particle are cheap enough for a few thousand points.
