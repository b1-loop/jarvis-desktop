/// @file animation.hpp
/// @brief Every number the overlay animates, computed without any UI library.
///
/// The boot sequence and the state animations are functions of elapsed
/// time, never of frame count: the renderer may run at 60 fps, 15 fps or
/// stall for a second and the picture at a given instant is the same. That
/// property, the easing curves, the boot acts, the state cross-fade, the
/// level smoothing and the seeded particle stream are all unit-tested here
/// without a window, a GPU or Dear ImGui. The per-frame composition of all
/// of it lives in animator.hpp; hud.cpp only turns the numbers into draw calls.

#pragma once

#include "jarvis/overlay/effects.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/scene.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::overlay::anim {

/// One full turn in radians.
inline constexpr float kTau = 2.0F * std::numbers::pi_v<float>;

// --- scalar helpers ------------------------------------------------------------

/// @return @p value clamped to [0, 1].
[[nodiscard]] float clamp01(float value) noexcept;

/// Linear interpolation from @p from (t = 0) to @p to (t = 1); t is not clamped.
[[nodiscard]] float lerp(float from, float to, float t) noexcept;

/// How far @p time is through a beat that starts at @p start and lasts
/// @p duration, clamped to [0, 1]. A zero or negative duration is a step.
[[nodiscard]] float progress(float time, float start, float duration) noexcept;

/// Hermite smoothstep of @p value between @p edge0 and @p edge1, in [0, 1].
[[nodiscard]] float smoothstep(float edge0, float edge1, float value) noexcept;

// --- easing ----------------------------------------------------------------------

/// Easing curves. Each maps [0, 1] onto [0, 1] monotonically with f(0) = 0
/// and f(1) = 1; inputs outside [0, 1] are clamped first.
enum class Easing : std::uint8_t { Linear, InCubic, OutCubic, InOutCubic, OutExpo };

[[nodiscard]] float easeInCubic(float t) noexcept;
[[nodiscard]] float easeOutCubic(float t) noexcept;
[[nodiscard]] float easeInOutCubic(float t) noexcept;
/// Fast start, long soft landing; used where a motion should "settle".
[[nodiscard]] float easeOutExpo(float t) noexcept;
/// Dispatch on @p easing.
[[nodiscard]] float ease(Easing easing, float t) noexcept;

// --- keyframe tracks ---------------------------------------------------------------

/// A value at a point in time. @c easing shapes the segment that *ends* at
/// this key, so a track reads like a script: "by 0.7 s, ease out to 1".
struct Keyframe {
    float time{0.0F};
    float value{0.0F};
    Easing easing{Easing::Linear};
};

/// Value of the piecewise track @p keys (sorted by time) at @p time. Before
/// the first key it holds the first value, after the last it holds the last;
/// an empty track is 0.
[[nodiscard]] float sampleTrack(std::span<const Keyframe> keys, float time) noexcept;

// --- deterministic randomness ---------------------------------------------------------

/// SplitMix64: tiny, fast, and bit-identical on every platform and standard
/// library, unlike std::uniform_real_distribution. Fixed seed, fixed picture.
class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) noexcept : state_(seed) {}

    /// Next raw 64-bit value.
    [[nodiscard]] std::uint64_t next() noexcept;
    /// Uniform in [0, 1), from the top 24 bits so every value is exact in a float.
    [[nodiscard]] float nextUnit() noexcept;
    /// Uniform in [@p low, @p high).
    [[nodiscard]] float uniform(float low, float high) noexcept;

private:
    std::uint64_t state_;
};

/// Stateless hash of (@p index, @p step) to [0, 1); the renderer uses it to
/// flicker fragments without keeping per-fragment state between frames.
[[nodiscard]] float hashUnit(std::uint64_t index, std::uint64_t step) noexcept;

// --- boot particle stream -------------------------------------------------------------------

/// Parameters of one particle, drawn once from the seeded generator.
struct Particle {
    /// Final angle on the ring, radians.
    float angle{0.0F};
    /// Starting distance from the centre, in ring radii (> 1: outside).
    float startRadius{0.0F};
    /// Seconds after the particle phase starts before this one moves.
    float delay{0.0F};
    /// Seconds from leaving its start to settling on the ring.
    float travel{0.0F};
    /// Extra angle swept while travelling inward (a spiral, not a spoke).
    float swirl{0.0F};
    /// Relative dot size in [0.5, 1.5).
    float size{1.0F};
};

/// Where one particle is at a given instant, relative to the ring centre in
/// ring radii (x right, y down), with its opacity.
struct ParticlePoint {
    float x{0.0F};
    float y{0.0F};
    float alpha{0.0F};
    float size{1.0F};
};

/// Ranges the particles are drawn from.
struct ParticleConfig {
    std::size_t count = 96;
    float minStartRadius = 1.6F;
    float maxStartRadius = 2.6F;
    float maxDelay = 0.30F;
    float minTravel = 0.35F;
    float maxTravel = 0.55F;
    float maxSwirl = 0.9F;
    /// Radius particles settle on, in ring radii.
    float targetRadius = 1.0F;
};

/// Particles that stream inward and settle on the ring during the boot.
///
/// Positions are closed-form functions of time (no integration), so the
/// field is deterministic for a seed and independent of the frame rate.
class ParticleField {
public:
    ParticleField(std::uint64_t seed, ParticleConfig config = {});

    /// Positions at @p time seconds after the particle phase began. Writes
    /// into @p out (resized to particles().size()) so a frame allocates nothing.
    void evaluate(float time, std::vector<ParticlePoint>& out) const;

    /// Seconds after which every particle has settled.
    [[nodiscard]] float settledAfter() const noexcept;

    [[nodiscard]] const std::vector<Particle>& particles() const noexcept { return particles_; }
    [[nodiscard]] const ParticleConfig& config() const noexcept { return config_; }

private:
    ParticleConfig config_;
    std::vector<Particle> particles_;
};

// --- boot sequence -----------------------------------------------------------------------

/// The four acts of the boot sequence.
enum class BootAct : std::uint8_t { Ignition, Assembly, SystemsCheck, Handoff };

/// When each part of the boot happens, in seconds from the start (~4 s).
struct BootBeats {
    // Act 1, ignition: a point of light, a flash, a shock ring.
    float pointGrow = 0.15F;
    float flashPeak = 0.20F;
    float shockStart = 0.16F;
    float shockDuration = 0.65F;
    // Act 2, assembly: rings fly in from depth and lock, the globe draws itself.
    float assemblyStart = 0.80F;
    float ringFlight = 0.55F;
    float ringStagger = 0.17F;
    float globeStart = 0.95F;
    float globeDuration = 1.15F;
    float particlesStart = 0.75F;
    // Act 3, systems check: labels count up, readout and title type on.
    float systemsStart = 2.20F;
    float labelCount = 0.70F;
    float scanDuration = 0.60F;
    float readoutStart = 2.30F;
    float readoutLineInterval = 0.14F;
    float readoutCharInterval = 0.010F;
    float titleStart = 2.60F;
    /// Seconds between two typed title characters.
    float charInterval = 0.05F;
    float subtitleStart = 3.20F;
    float subtitleDuration = 0.20F;
    // Act 4, hand-off: everything compresses into the corner with a whip pan.
    float handoffStart = 3.45F;
    float handoffDuration = 0.55F;
    /// Cursor blink period once typing has finished.
    float cursorBlink = 0.24F;
};

/// Everything the renderer needs for one boot frame. All alphas and
/// progresses are in [0, 1].
struct BootFrame {
    /// Position on the timeline in seconds (after any skip).
    float time{0.0F};
    BootAct act{BootAct::Ignition};
    /// Growth of the initial point of light; 1 once it is fully lit.
    float ignite{0.0F};
    /// Brightness of the initial flash at the centre; peaks early, decays.
    float flash{0.0F};
    /// The ignition shock ring: radius in ring radii and opacity.
    float shockRadius{0.0F};
    float shockAlpha{0.0F};
    /// Per ring layer (outermost first): 0 far out in front of the screen,
    /// 1 locked into place.
    std::array<float, kRingCount> ringArrival{};
    /// How far the globe wireframe has drawn itself.
    float globe{0.0F};
    /// Seconds since the particle stream started, negative before it.
    float particleTime{0.0F};
    /// Angle of the radial sweep, radians; continuous.
    float sweepAngle{0.0F};
    /// Extra ring rotation, eased from 0 to 1 over the whole boot: the rings
    /// spin fast and wind down to their idle speed. 1 once the boot is over,
    /// so the renderer can add it unconditionally without a jump.
    float spin{0.0F};
    /// Tick labels counting up from 0 to their final degrees.
    float labels{0.0F};
    /// Scan line position, 0 at the top of the canvas, 1 at the bottom.
    float scan{0.0F};
    float scanAlpha{0.0F};
    /// Characters of the title typed so far.
    std::size_t titleChars{0};
    bool cursorVisible{false};
    float subtitle{0.0F};
    /// Eased progress of the move from the centre to the corner panel.
    float handoff{0.0F};
    /// Speed of the hand-off as a share of its peak: drives the whip-pan
    /// motion blur. 0 at both ends, 1 in the middle.
    float whip{0.0F};
    /// Opacity of elements that only exist during boot (fades over the handoff).
    float bootOnly{1.0F};
    /// Glitch amount of the boot beats (flash, ring locks, title, hand-off).
    float glitch{0.0F};
    bool finished{false};
};

/// How a flying ring is posed: scale of its radius, tilt out of the screen
/// plane (radians, projected as an ellipse) and opacity.
struct RingPose {
    float scale{1.0F};
    float tilt{0.0F};
    float alpha{1.0F};
};

/// Pose of a ring at @p arrival (BootFrame::ringArrival): it starts large,
/// tilted and invisible in front of the screen and locks flat at scale 1.
[[nodiscard]] RingPose ringPose(float arrival) noexcept;

/// Maps time since the boot started to a BootFrame.
class BootTimeline {
public:
    explicit BootTimeline(BootBeats beats = {}, std::string title = "J.A.R.V.I.S.");

    /// The frame at @p time seconds on the timeline (see timelineTime()).
    [[nodiscard]] BootFrame evaluate(float time) const noexcept;

    /// Timeline position for a boot that started @p sinceStart seconds ago
    /// and, optionally, was skipped @p skippedAfter seconds after it started.
    /// A skip before the handoff jumps straight to it, so a skipped boot still
    /// glides into the corner instead of popping; a skip during or after the
    /// handoff changes nothing.
    [[nodiscard]] float timelineTime(float sinceStart,
                                     std::optional<float> skippedAfter) const noexcept;

    /// Characters of readout line @p line (@p length long) typed at @p time.
    /// Lines start one after another; each types at its own pace.
    [[nodiscard]] std::size_t readoutChars(std::size_t line, std::size_t length,
                                           float time) const noexcept;

    /// The act @p time falls in.
    [[nodiscard]] BootAct actAt(float time) const noexcept;

    /// Seconds from start to finished when nothing is skipped.
    [[nodiscard]] float duration() const noexcept;

    [[nodiscard]] const BootBeats& beats() const noexcept { return beats_; }
    [[nodiscard]] const std::string& title() const noexcept { return title_; }
    [[nodiscard]] std::span<const GlitchSpike> spikes() const noexcept { return spikes_; }

private:
    void evaluateIgnition(float time, BootFrame& frame) const noexcept;
    void evaluateAssembly(float time, BootFrame& frame) const noexcept;
    void evaluateSystems(float time, BootFrame& frame) const noexcept;
    void evaluateHandoff(float time, BootFrame& frame) const noexcept;

    BootBeats beats_;
    std::string title_;
    std::vector<GlitchSpike> spikes_;
};

// --- state animations ---------------------------------------------------------------------

/// Per-state opacity, indexed by AssistantState. Always sums to 1.
using StateWeights = std::array<float, 4>;

/// Index of @p state in a StateWeights array.
[[nodiscard]] std::size_t stateIndex(AssistantState state) noexcept;

/// Cross-fade between assistant states.
///
/// A transition blends from whatever was on screen at that instant (which
/// may itself be mid-fade) to the new state, so rapid changes never jump.
class StateBlend {
public:
    explicit StateBlend(float durationSeconds = 0.2F,
                        AssistantState initial = AssistantState::Idle) noexcept;

    /// Start fading to @p state at @p now (seconds). Re-targeting the
    /// current target is a no-op.
    void setTarget(AssistantState state, float now) noexcept;

    [[nodiscard]] StateWeights weights(float now) const noexcept;
    [[nodiscard]] bool transitioning(float now) const noexcept;
    [[nodiscard]] AssistantState target() const noexcept { return target_; }

private:
    float duration_;
    StateWeights from_{};
    AssistantState target_;
    float since_{0.0F};
};

/// One-pole smoother with separate attack and release time constants.
///
/// The coefficient is derived from the real frame interval, 1 - exp(-dt/tau),
/// so a constant input converges identically at any frame rate.
class LevelSmoother {
public:
    LevelSmoother(float attackSeconds = 0.05F, float releaseSeconds = 0.30F) noexcept;

    /// Move towards @p target over @p dtSeconds; returns the new value.
    float update(float target, float dtSeconds) noexcept;
    void reset(float value = 0.0F) noexcept { value_ = value; }
    [[nodiscard]] float value() const noexcept { return value_; }

private:
    float attack_;
    float release_;
    float value_{0.0F};
};

/// Calm idle "breathing" in [0, 1]: 0.5 - 0.5 cos(2 pi t / period).
[[nodiscard]] float breathing(float time, float periodSeconds) noexcept;

/// The wake-word ripple for a flash value that decays from 1 to 0.
struct Ripple {
    /// Radius in ring radii, growing from 1.
    float radius{1.0F};
    float alpha{0.0F};
};

/// Ripple for @p flash (OverlaySnapshot::wakeFlash) spreading to 1 + @p spread.
[[nodiscard]] Ripple wakeRipple(float flash, float spread) noexcept;

/// Amplitudes in [0, 1] for @p out.size() spokes around a ring.
///
/// The level history (oldest first) is mirrored around the ring with the
/// newest sample at the top, scaled by the smoothed @p energy and given a
/// slow time-based shimmer so a steady tone still looks alive. Silence is
/// @p floor everywhere. Spoke i and spoke n - i are always equal.
void radialWaveform(std::span<const float> levels, float energy, float time, float floor,
                    std::span<float> out) noexcept;

// --- frame statistics -------------------------------------------------------------------------

/// Frames the percentiles of FrameStats are taken over (the most recent).
inline constexpr std::size_t kStatsWindow = 1024;

/// A frame interval longer than this multiple of its scheduled period is a
/// hitch: at least one refresh (or timer tick) was missed.
inline constexpr double kHitchFactor = 1.5;

/// Frame timings: exponential moving averages for a live readout plus
/// percentiles of the recent window, because an average hides exactly the
/// occasional long frame that reads as a stutter.
struct FrameStats {
    std::uint64_t frames{0};
    /// CPU time spent building and submitting one frame, milliseconds.
    double renderMs{0.0};
    /// Time between the starts of two consecutive frames, milliseconds.
    double intervalMs{0.0};
    /// GPU time of one frame from D3D11 timestamp queries, milliseconds;
    /// 0 until the first query result arrives (a few frames late by design).
    double gpuMs{0.0};
    /// Frames for which a GPU time was recorded.
    std::uint64_t gpuFrames{0};
    /// Frame interval percentiles over the last kStatsWindow frames, ms.
    double intervalP50Ms{0.0};
    double intervalP95Ms{0.0};
    double intervalP99Ms{0.0};
    double intervalMaxMs{0.0};
    /// 99th percentile of the CPU time per frame, ms.
    double renderP99Ms{0.0};
    /// Frames that started more than kHitchFactor x their scheduled period
    /// after the previous one.
    std::uint64_t hitches{0};
    /// Frames handed to the compositor while its vblank grid was known.
    std::uint64_t presentedFrames{0};
    /// Refreshes by which those frames missed their cadence: a frame meant
    /// to stay N refreshes that stayed N + 1 (a visible repeat) or N - 1
    /// counts 1. Zero is perfectly even motion.
    std::uint64_t offCadenceRefreshes{0};
    /// The compositor's refresh rate, Hz (0 until known).
    double refreshHz{0.0};
};

/// Folds frame timings into FrameStats.
///
/// record() only writes into fixed rings (no allocation on the UI thread);
/// stats() sorts copies of them, which only the demo and tests call.
class FrameStatsAccumulator {
public:
    explicit FrameStatsAccumulator(double smoothing = 0.1) noexcept : smoothing_(smoothing) {}

    /// Record one frame that took @p renderMs and started @p intervalMs
    /// after the previous one (ignored for the very first frame), which was
    /// scheduled @p expectedMs after it (0: unknown, never a hitch).
    void record(double renderMs, double intervalMs, double expectedMs = 0.0) noexcept;
    /// Record the GPU time of a (past) frame.
    void recordGpu(double gpuMs) noexcept;
    /// Record that @p presents frames reached the screen over @p refreshes
    /// vertical blanks at @p refreshHz while each was meant to stay
    /// @p refreshesPerFrame (at least 1).
    void recordPresentation(std::uint32_t presents, std::uint32_t refreshes,
                            std::uint32_t refreshesPerFrame, double refreshHz) noexcept;
    void reset() noexcept;
    /// The averages plus the percentiles of the recent window.
    [[nodiscard]] FrameStats stats() const;

private:
    double smoothing_;
    FrameStats stats_;
    std::array<double, kStatsWindow> intervals_{};
    std::array<double, kStatsWindow> renders_{};
    std::size_t intervalCount_{0};
    std::size_t renderCount_{0};
};

/// Nearest-rank percentile @p share (0-1) of @p values, which it sorts; 0
/// for an empty span.
[[nodiscard]] double percentile(std::span<double> values, double share) noexcept;

}  // namespace jarvis::overlay::anim
