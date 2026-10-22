#include "jarvis/overlay/scene.hpp"

#include "jarvis/overlay/animation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

namespace jarvis::overlay::anim {

namespace {

constexpr float kPi = std::numbers::pi_v<float>;

// The resting view: a little above the equator, turned a little to the left.
constexpr float kRestPitch = 0.32F;
constexpr float kRestYaw = -0.25F;
constexpr float kYawSwing = 0.22F;
constexpr float kPitchSwing = 0.09F;
constexpr float kYawFrequency = 0.17F;
constexpr float kPitchFrequency = 0.11F;
/// Phase offset so the two drifts never line up into a straight path.
constexpr float kPitchPhase = 1.3F;
/// Screen shift per radian of camera drift per unit of layer depth.
constexpr float kParallaxGain = 0.5F;

/// A line's reveal window spans this many "line slots" of the whole draw,
/// so roughly that many lines are being drawn at any moment.
constexpr float kRevealOverlapLines = 4.0F;

constexpr std::array<Orbit, 4> kSatellites{{
    {.radius = 1.55F, .inclination = 0.45F, .node = 0.0F, .speed = 0.55F, .phase = 0.0F},
    {.radius = 1.78F, .inclination = -0.75F, .node = 1.2F, .speed = -0.38F, .phase = 2.1F},
    {.radius = 1.42F, .inclination = 1.15F, .node = 2.4F, .speed = 0.72F, .phase = 4.0F},
    {.radius = 1.95F, .inclination = 0.2F, .node = -0.9F, .speed = 0.27F, .phase = 1.0F},
}};

constexpr std::array<RingLayer, kRingCount> kRings{{
    {.radius = 1.00F,
     .thickness = 1.0F,
     .segments = 0,
     .fill = 1.0F,
     .speed = 0.04F,
     .alpha = 0.35F,
     .ticks = 72,
     .majorEvery = 6,
     .labels = true,
     .dashes = 0},
    {.radius = 0.91F,
     .thickness = 2.4F,
     .segments = 3,
     .fill = 0.26F,
     .speed = 0.30F,
     .alpha = 0.90F,
     .ticks = 0,
     .majorEvery = 1,
     .labels = false,
     .dashes = 0},
    {.radius = 0.81F,
     .thickness = 1.2F,
     .segments = 40,
     .fill = 0.50F,
     .speed = -0.12F,
     .alpha = 0.50F,
     .ticks = 0,
     .majorEvery = 1,
     .labels = false,
     .dashes = 6},
    {.radius = 0.70F,
     .thickness = 2.8F,
     .segments = 5,
     .fill = 0.55F,
     .speed = 0.50F,
     .alpha = 0.75F,
     .ticks = 0,
     .majorEvery = 1,
     .labels = false,
     .dashes = 0},
    {.radius = 0.59F,
     .thickness = 1.0F,
     .segments = 0,
     .fill = 1.0F,
     .speed = -0.20F,
     .alpha = 0.30F,
     .ticks = 36,
     .majorEvery = 3,
     .labels = false,
     .dashes = 10},
}};

/// Speed multipliers per state (Idle, Listening, Thinking, Speaking): idle
/// drifts, thinking spins up, speaking and listening sit in between.
constexpr std::array<float, 4> kStateSpeeds{0.45F, 1.1F, 2.4F, 1.4F};

constexpr float kFullCircleDegrees = 360.0F;

Vec3 onSphere(float latitude, float longitude) noexcept {
    const float ring = std::cos(latitude);
    return Vec3{
        .x = ring * std::cos(longitude), .y = std::sin(latitude), .z = ring * std::sin(longitude)};
}

GlobeLine latitudeLine(float latitude, int segments) {
    GlobeLine line{.kind = GlobeLineKind::Latitude, .points = {}, .closed = true};
    line.points.reserve(static_cast<std::size_t>(segments));
    for (int i = 0; i < segments; ++i) {
        const float longitude = kTau * static_cast<float>(i) / static_cast<float>(segments);
        line.points.push_back(onSphere(latitude, longitude));
    }
    return line;
}

GlobeLine meridianLine(float longitude, int segments) {
    const int steps = std::max(segments / 2, 1);
    GlobeLine line{.kind = GlobeLineKind::Meridian, .points = {}, .closed = false};
    line.points.reserve(static_cast<std::size_t>(steps) + 1);
    for (int i = 0; i <= steps; ++i) {
        const float latitude =
            (-kPi / 2.0F) + (kPi * static_cast<float>(i) / static_cast<float>(steps));
        line.points.push_back(onSphere(latitude, longitude));
    }
    return line;
}

GlobeLine orbitLine(const Orbit& orbit, int segments) {
    GlobeLine line{.kind = GlobeLineKind::Orbit, .points = {}, .closed = true};
    line.points.reserve(static_cast<std::size_t>(segments));
    for (int i = 0; i < segments; ++i) {
        line.points.push_back(
            orbitPosition(orbit, kTau * static_cast<float>(i) / static_cast<float>(segments)));
    }
    return line;
}

/// Wrap an angle into [0, 2 pi) so a rotor that runs all day keeps its precision.
float wrapAngle(float angle) noexcept {
    const float wrapped = std::fmod(angle, kTau);
    return wrapped < 0.0F ? wrapped + kTau : wrapped;
}

}  // namespace

// --- vectors and the camera ------------------------------------------------------------

Vec3 operator+(Vec3 a, Vec3 b) noexcept {
    return Vec3{.x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z};
}

Vec3 operator-(Vec3 a, Vec3 b) noexcept {
    return Vec3{.x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z};
}

Vec3 operator*(Vec3 v, float s) noexcept {
    return Vec3{.x = v.x * s, .y = v.y * s, .z = v.z * s};
}

float length(Vec3 v) noexcept {
    return std::sqrt((v.x * v.x) + (v.y * v.y) + (v.z * v.z));
}

Vec3 rotateX(Vec3 v, float angle) noexcept {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return Vec3{.x = v.x, .y = (v.y * c) - (v.z * s), .z = (v.y * s) + (v.z * c)};
}

Vec3 rotateY(Vec3 v, float angle) noexcept {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return Vec3{.x = (v.x * c) + (v.z * s), .y = v.y, .z = (-v.x * s) + (v.z * c)};
}

Vec3 rotateZ(Vec3 v, float angle) noexcept {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return Vec3{.x = (v.x * c) - (v.y * s), .y = (v.x * s) + (v.y * c), .z = v.z};
}

Viewport viewportFor(const Camera& camera, float centreX, float centreY, float radiusPx) noexcept {
    return Viewport{.centreX = centreX, .centreY = centreY, .focal = radiusPx * camera.distance};
}

Projected project(const Camera& camera, const Viewport& viewport, Vec3 world) noexcept {
    // The camera orbits the origin; moving it by (yaw, pitch) is the same
    // as turning the world the opposite way and keeping the eye on -z.
    const Vec3 view = rotateX(rotateY(world, -camera.yaw), -camera.pitch);
    const float depth = view.z + camera.distance;
    if (depth <= camera.nearPlane) {
        return Projected{};
    }
    const float scale = viewport.focal / depth;
    return Projected{.x = viewport.centreX + (view.x * scale),
                     .y = viewport.centreY - (view.y * scale),
                     .depth = depth,
                     .scale = scale,
                     .visible = true};
}

Projector::Projector(const Camera& camera, const Viewport& viewport, float spin,
                     float tilt) noexcept
    : camera_(camera), viewport_(viewport) {
    // Columns of the combined rotation are the images of the basis vectors.
    const auto image = [&camera, spin, tilt](Vec3 basis) {
        return rotateX(rotateY(rotateZ(rotateY(basis, spin), tilt), -camera.yaw), -camera.pitch);
    };
    const Vec3 ex = image({.x = 1.0F, .y = 0.0F, .z = 0.0F});
    const Vec3 ey = image({.x = 0.0F, .y = 1.0F, .z = 0.0F});
    const Vec3 ez = image({.x = 0.0F, .y = 0.0F, .z = 1.0F});
    rows_ = {Vec3{.x = ex.x, .y = ey.x, .z = ez.x}, Vec3{.x = ex.y, .y = ey.y, .z = ez.y},
             Vec3{.x = ex.z, .y = ey.z, .z = ez.z}};
}

Projected Projector::operator()(Vec3 local) const noexcept {
    const auto dot = [local](Vec3 row) {
        return (row.x * local.x) + (row.y * local.y) + (row.z * local.z);
    };
    const float depth = dot(std::get<2>(rows_)) + camera_.distance;
    if (depth <= camera_.nearPlane) {
        return Projected{};
    }
    const float scale = viewport_.focal / depth;
    return Projected{.x = viewport_.centreX + (dot(std::get<0>(rows_)) * scale),
                     .y = viewport_.centreY - (dot(std::get<1>(rows_)) * scale),
                     .depth = depth,
                     .scale = scale,
                     .visible = true};
}

Camera driftingCamera(float time, float motion) noexcept {
    const float t = time * motion;
    Camera camera;
    camera.yaw = kRestYaw + (kYawSwing * std::sin(t * kYawFrequency * kTau));
    camera.pitch =
        kRestPitch + (kPitchSwing * std::sin((t * kPitchFrequency * kTau) + kPitchPhase));
    return camera;
}

Parallax parallax(const Camera& camera, float depth) noexcept {
    return Parallax{.x = (camera.yaw - kRestYaw) * depth * kParallaxGain,
                    .y = -(camera.pitch - kRestPitch) * depth * kParallaxGain};
}

// --- the globe ---------------------------------------------------------------------------

std::vector<GlobeLine> buildGlobe(const GlobeSpec& spec) {
    std::vector<GlobeLine> lines;
    const int latitudes = std::max(spec.latitudes, 0);
    const int meridians = std::max(spec.meridians, 0);
    const int orbits = std::max(spec.orbits, 0);
    const int segments = std::max(spec.segments, 3);
    lines.reserve(static_cast<std::size_t>(latitudes) + static_cast<std::size_t>(meridians) +
                  static_cast<std::size_t>(orbits));
    for (int i = 0; i < latitudes; ++i) {
        const float latitude =
            (-kPi / 2.0F) + (kPi * static_cast<float>(i + 1) / static_cast<float>(latitudes + 1));
        lines.push_back(latitudeLine(latitude, segments));
    }
    for (int i = 0; i < meridians; ++i) {
        lines.push_back(
            meridianLine(kTau * static_cast<float>(i) / static_cast<float>(meridians), segments));
    }
    for (int i = 0; i < orbits; ++i) {
        const Orbit ring{.radius = spec.orbitRadius,
                         .inclination = 0.5F + (0.55F * static_cast<float>(i)),
                         .node = 0.9F * static_cast<float>(i),
                         .speed = 0.0F,
                         .phase = 0.0F};
        lines.push_back(orbitLine(ring, segments));
    }
    return lines;
}

float lineReveal(float progress, std::size_t index, std::size_t count) noexcept {
    if (count <= 1 || progress >= 1.0F) {
        return clamp01(progress);
    }
    const auto lines = static_cast<float>(count);
    const float window = std::min(1.0F, kRevealOverlapLines / lines);
    const float stride = (1.0F - window) / (lines - 1.0F);
    return anim::progress(progress, stride * static_cast<float>(index), window);
}

// --- orbits (satellites) --------------------------------------------------------------------

Vec3 orbitPosition(const Orbit& orbit, float angle) noexcept {
    const Vec3 flat{
        .x = orbit.radius * std::cos(angle), .y = 0.0F, .z = orbit.radius * std::sin(angle)};
    return rotateY(rotateX(flat, orbit.inclination), orbit.node);
}

std::span<const Orbit> satelliteOrbits() noexcept {
    return kSatellites;
}

// --- rings ----------------------------------------------------------------------------------

std::span<const RingLayer, kRingCount> ringLayers() noexcept {
    return kRings;
}

int tickLabel(std::size_t index, std::size_t count, float countUp) noexcept {
    if (count == 0) {
        return 0;
    }
    const float degrees =
        kFullCircleDegrees * static_cast<float>(index) / static_cast<float>(count);
    return static_cast<int>(std::lround(degrees * clamp01(countUp)));
}

float stateSpeedFactor(std::span<const float, 4> weights) noexcept {
    float factor = 0.0F;
    std::size_t index = 0;
    for (const float weight : weights) {
        factor += weight * kStateSpeeds.at(index++);
    }
    return factor;
}

Rotor::Rotor(float tauSeconds, float angle, float speed) noexcept
    : tau_(std::max(tauSeconds, 0.0F)), angle_(wrapAngle(angle)), speed_(speed) {}

void Rotor::advance(float targetSpeed, float dtSeconds) noexcept {
    if (dtSeconds <= 0.0F) {
        return;
    }
    if (tau_ <= 0.0F) {
        speed_ = targetSpeed;
        angle_ = wrapAngle(angle_ + (targetSpeed * dtSeconds));
        return;
    }
    // v(t) = target + (v0 - target) e^(-t/tau), integrated exactly over dt.
    const float decay = std::exp(-dtSeconds / tau_);
    const float excess = speed_ - targetSpeed;
    angle_ = wrapAngle(angle_ + (targetSpeed * dtSeconds) + (excess * tau_ * (1.0F - decay)));
    speed_ = targetSpeed + (excess * decay);
}

}  // namespace jarvis::overlay::anim

// Design notes:
// - The camera orbits the origin instead of being a free look-at camera:
//   two angles and a distance are all the HUD needs, and the projection of
//   a known point is then easy to state exactly in a test.
// - Meridians are half circles from pole to pole rather than full great
//   circles, so the "draw itself line by line" reveal sweeps around the
//   globe instead of drawing each great circle twice.
// - Rotors integrate their eased speed in closed form and wrap the angle,
//   which keeps them frame-rate independent and precise after a day of
//   uptime (a raw float angle would lose its fractional bits).
