/// @file orb.hpp
/// @brief The minimized HUD, a small glowing "orb": where it docks, how the
/// HUD morphs into it, and how it shows the assistant's state.
///
/// Minimizing (ADR-0024, update) shrinks the corner HUD into a circle of
/// about 48 logical pixels docked at the nearest work-area edge (or in the
/// corner the HUD sat in). Every decision about that circle is a pure
/// function here, unit-tested without a window: the docking maths, the
/// timeline of the morph in both directions, and the per-state look
/// (brightness, pulse, orbit, wake ripple) derived from the same
/// AnimationFrame the full HUD is drawn from.

#pragma once

#include "jarvis/overlay/animator.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/placement.hpp"

#include <cstdint>

namespace jarvis::overlay {

/// Diameter of the orb, 96-DPI pixels.
inline constexpr int kOrbDiameter = 48;
/// Gap between the orb and the work-area edge it docks to, 96-DPI pixels.
inline constexpr int kOrbMargin = 12;
/// A HUD whose nearest horizontal and nearest vertical edge are both this
/// close (96-DPI pixels) sits in a corner, and the orb docks in it. The
/// default corner placement (24 px margin) and a snapped corner both count.
inline constexpr int kOrbCornerReach = 48;
/// Transparent margin around the orb for its halo and the wake ripple,
/// 96-DPI pixels; click-through like every other transparent pixel.
inline constexpr int kOrbBleed = 30;
/// Seconds the HUD-to-orb morph (and its reverse) takes.
inline constexpr float kOrbMorphSeconds = 0.5F;

/// A work-area edge.
enum class DockEdge : std::uint8_t { Left, Top, Right, Bottom };

/// The edge of @p area nearest to @p panel; ties go to Left, Right, Top,
/// Bottom in that order.
[[nodiscard]] DockEdge nearestEdge(Placement panel, PixelRect area) noexcept;

/// Physical side of the orb on a monitor with DPI scale @p scale.
[[nodiscard]] int orbSide(float scale) noexcept;

/// Where the orb goes for a HUD at @p panel (screen pixels, no bleed) whose
/// ring centre is (@p anchorX, @p anchorY), on @p monitor.
///
/// In a corner (see kOrbCornerReach) the orb docks in that corner;
/// otherwise at the nearest edge, centred on the ring along that edge so
/// the ring visibly shrinks into it. The result keeps kOrbMargin from the
/// edge, is DPI-scaled with the monitor, and is clamped fully inside the
/// work area.
[[nodiscard]] Placement dockOrb(Placement panel, float anchorX, float anchorY,
                                const MonitorArea& monitor) noexcept;

/// The square of side @p side centred on (@p centreX, @p centreY).
[[nodiscard]] Placement squareAt(float centreX, float centreY, int side) noexcept;

/// One point of the HUD/orb morph. Progress 0 is the full HUD, 1 the orb;
/// restoring plays the same curve backwards, so the two directions mirror.
struct OrbMorph {
    /// Eased share of the way the ring has flown and shrunk into the orb.
    float travel{0.0F};
    /// The HUD's text column, scrims and telemetry: gone in the first third,
    /// so what flies is a glowing ring, never a panel or a rectangle.
    float panelAlpha{1.0F};
    /// The HUD's ring scene (globe, rings, particles): fades as it arrives.
    float sceneAlpha{1.0F};
    /// The orb itself: fades in as the shrinking ring reaches its size.
    float orbAlpha{0.0F};

    [[nodiscard]] bool operator==(const OrbMorph&) const noexcept = default;
};

/// The morph at linear progress @p t (clamped to [0, 1]).
[[nodiscard]] OrbMorph orbMorphAt(float t) noexcept;

/// What the orb draws for one frame, in [0, 1] unless noted.
struct OrbLook {
    /// Brightness of the glowing core.
    float core{0.0F};
    /// Radius multiplier of the core: the pulse (about 0.9 to 1.35).
    float pulse{1.0F};
    /// Brightness of the rim ring.
    float rim{0.0F};
    /// Brightness of the thinking orbit (its state weight).
    float orbit{0.0F};
    /// Angle of the orbiting light, radians.
    float orbitAngle{0.0F};
    /// The wake-word ripple, in orb radii (from 1) and its alpha.
    float rippleRadius{1.0F};
    float rippleAlpha{0.0F};
};

/// The orb's look for @p frame: idle breathes slowly, listening pulses
/// with the input level, thinking orbits, speaking pulses with the output
/// level and its syllable onsets; states cross-fade by their weights.
[[nodiscard]] OrbLook orbLook(const anim::AnimationFrame& frame) noexcept;

}  // namespace jarvis::overlay
