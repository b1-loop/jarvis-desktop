/// @file scene.hpp
/// @brief Geometry of the holographic scene: camera, globe, orbits, rings.
///
/// The HUD has depth: a wireframe globe and its satellites live in 3D and
/// are projected through a slowly drifting perspective camera, and the ring
/// layers fly in from depth during the boot. Everything here is plain maths
/// on floats (no ImGui, no Direct3D) so projections, line counts, ring
/// layouts and rotor speeds are unit-tested exactly.

#pragma once

#include "jarvis/overlay/model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace jarvis::overlay::anim {

// --- vectors and the camera ------------------------------------------------------------

/// A point or direction in the scene. World axes: x right, y up, z away
/// from the viewer; the unit of length is the globe's radius.
struct Vec3 {
    float x{0.0F};
    float y{0.0F};
    float z{0.0F};

    [[nodiscard]] bool operator==(const Vec3&) const noexcept = default;
};

[[nodiscard]] Vec3 operator+(Vec3 a, Vec3 b) noexcept;
[[nodiscard]] Vec3 operator-(Vec3 a, Vec3 b) noexcept;
[[nodiscard]] Vec3 operator*(Vec3 v, float s) noexcept;
[[nodiscard]] float length(Vec3 v) noexcept;

/// Rotations by @p angle radians about the named axis (right-handed).
[[nodiscard]] Vec3 rotateX(Vec3 v, float angle) noexcept;
[[nodiscard]] Vec3 rotateY(Vec3 v, float angle) noexcept;
[[nodiscard]] Vec3 rotateZ(Vec3 v, float angle) noexcept;

/// A camera that orbits the origin and always looks at it.
struct Camera {
    /// Turn about the vertical axis, radians.
    float yaw{0.0F};
    /// Elevation above the equator, radians (positive looks down on the globe).
    float pitch{0.0F};
    /// Distance from the origin, in globe radii.
    float distance{4.0F};
    /// Points closer to the eye than this are culled.
    float nearPlane{0.1F};
};

/// A projected point in screen pixels (y down).
struct Projected {
    float x{0.0F};
    float y{0.0F};
    /// View-space depth: distance in front of the eye (larger is farther).
    float depth{0.0F};
    /// Pixels per world unit at this depth (focal / depth); for sizes.
    float scale{0.0F};
    /// False when the point is behind the near plane; x and y are then 0.
    bool visible{false};
};

/// Where the projection lands: the screen position of the origin and the
/// focal length in pixels.
struct Viewport {
    float centreX{0.0F};
    float centreY{0.0F};
    float focal{1.0F};
};

/// A viewport in which a point at the origin's depth, one world unit off
/// axis, lands @p radiusPx from the centre: focal = radius * distance.
[[nodiscard]] Viewport viewportFor(const Camera& camera, float centreX, float centreY,
                                   float radiusPx) noexcept;

/// Perspective projection of @p world through @p camera onto @p viewport.
[[nodiscard]] Projected project(const Camera& camera, const Viewport& viewport,
                                Vec3 world) noexcept;

/// Projects many points through one camera after one model rotation
/// (rotateY by @p spin, then rotateZ by @p tilt). The trigonometry is done
/// once in the constructor; each point is then a 3x3 multiply, which is
/// what lets the globe re-project hundreds of points per frame cheaply.
class Projector {
public:
    Projector(const Camera& camera, const Viewport& viewport, float spin = 0.0F,
              float tilt = 0.0F) noexcept;

    /// Same result as project(camera, viewport, rotateZ(rotateY(local, spin), tilt)).
    [[nodiscard]] Projected operator()(Vec3 local) const noexcept;

    [[nodiscard]] const Camera& camera() const noexcept { return camera_; }

private:
    Camera camera_;
    Viewport viewport_;
    /// Rows of the combined model-view rotation.
    std::array<Vec3, 3> rows_{};
};

/// The camera at @p time: a slow Lissajous drift of yaw and pitch around a
/// three-quarter view, scaled by @p motion (0 holds it still). Gives the
/// HUD parallax without ever moving fast enough to distract.
[[nodiscard]] Camera driftingCamera(float time, float motion) noexcept;

/// A 2D screen offset in units of the projection radius (x right, y down).
struct Parallax {
    float x{0.0F};
    float y{0.0F};
};

/// How far a flat layer at @p depth (positive is nearer the viewer, in
/// globe radii) shifts on screen when @p camera has drifted away from its
/// rest pose: nearer layers move more, which is what reads as depth.
/// Zero at the rest pose and for depth 0.
[[nodiscard]] Parallax parallax(const Camera& camera, float depth) noexcept;

// --- the globe ---------------------------------------------------------------------------

/// What a globe line is, so the renderer can style each kind.
enum class GlobeLineKind : std::uint8_t { Latitude, Meridian, Orbit };

/// One polyline of the wireframe.
struct GlobeLine {
    GlobeLineKind kind{GlobeLineKind::Latitude};
    /// Points on the line, in globe radii; a closed line does not repeat its start.
    std::vector<Vec3> points;
    bool closed{false};
};

/// Shape of the wireframe.
struct GlobeSpec {
    /// Parallels, evenly spaced in latitude, poles excluded.
    int latitudes{7};
    /// Half great circles from pole to pole, evenly spaced in longitude.
    int meridians{12};
    /// Points per full circle; a meridian (half circle) gets half plus one.
    int segments{48};
    /// Tilted rings around the globe at orbitRadius.
    int orbits{2};
    float orbitRadius{1.28F};
};

/// Every line of the globe: latitudes first (south to north), then the
/// meridians, then the orbits. Count: latitudes + meridians + orbits.
[[nodiscard]] std::vector<GlobeLine> buildGlobe(const GlobeSpec& spec);

/// How much of line @p index (of @p count) is drawn when the globe is
/// @p progress through drawing itself: lines start one after another with
/// overlapping windows, so the wireframe appears line by line. Monotonic in
/// @p progress, 0 everywhere at 0 and 1 everywhere at 1, and an earlier
/// line is always at least as complete as a later one.
[[nodiscard]] float lineReveal(float progress, std::size_t index, std::size_t count) noexcept;

// --- orbits (satellites) --------------------------------------------------------------------

/// A circular orbit around the origin.
struct Orbit {
    float radius{1.5F};
    /// Tilt of the orbit plane about the x axis, radians.
    float inclination{0.0F};
    /// Turn of the tilted plane about the y axis, radians.
    float node{0.0F};
    /// Base angular speed, radians per second (sign is the direction).
    float speed{0.5F};
    /// Angle at time 0.
    float phase{0.0F};
};

/// Position on @p orbit at @p angle radians along it.
[[nodiscard]] Vec3 orbitPosition(const Orbit& orbit, float angle) noexcept;

/// The satellites' orbits.
[[nodiscard]] std::span<const Orbit> satelliteOrbits() noexcept;

// --- rings ----------------------------------------------------------------------------------

/// One concentric ring layer of the HUD. Radii are fractions of the outer
/// radius, speeds radians per second (negative turns anticlockwise).
struct RingLayer {
    float radius{1.0F};
    float thickness{1.0F};
    /// Arc segments; 0 draws a full circle.
    int segments{0};
    /// Share of each segment's slot that is lit.
    float fill{1.0F};
    float speed{0.0F};
    float alpha{1.0F};
    /// Tick marks around the layer (0 for none) and every how many is major.
    int ticks{0};
    int majorEvery{1};
    /// Degree labels at the major ticks.
    bool labels{false};
    /// Data-stream dashes flowing along the layer.
    int dashes{0};
};

/// Number of ring layers.
inline constexpr std::size_t kRingCount = 5;

/// The ring layers, outermost first.
[[nodiscard]] std::span<const RingLayer, kRingCount> ringLayers() noexcept;

/// Value of the tick label at major tick @p index of @p count, in degrees,
/// while the boot counts labels up (@p countUp from 0 to 1).
[[nodiscard]] int tickLabel(std::size_t index, std::size_t count, float countUp) noexcept;

/// How much faster than its base speed everything turns in each state;
/// blended by the cross-fade weights, so a state change eases the speed.
[[nodiscard]] float stateSpeedFactor(std::span<const float, 4> weights) noexcept;

/// An angle driven by a speed that eases towards a target.
///
/// The speed approaches the target with time constant @c tau, and the angle
/// integrates that speed in closed form, so advancing by dt once or by
/// dt/2 twice gives the same angle and speed (up to rounding): the motion
/// is frame-rate independent and a state change never jerks.
class Rotor {
public:
    explicit Rotor(float tauSeconds = 0.6F, float angle = 0.0F, float speed = 0.0F) noexcept;

    /// Advance by @p dtSeconds while the speed eases towards @p targetSpeed.
    void advance(float targetSpeed, float dtSeconds) noexcept;

    [[nodiscard]] float angle() const noexcept { return angle_; }
    [[nodiscard]] float speed() const noexcept { return speed_; }

private:
    float tau_;
    float angle_;
    float speed_;
};

}  // namespace jarvis::overlay::anim
