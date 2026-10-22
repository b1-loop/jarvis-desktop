#include "jarvis/overlay/placement.hpp"

#include "jarvis/overlay/layout.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vector>

using jarvis::overlay::anchorFor;
using jarvis::overlay::clampInto;
using jarvis::overlay::dragTo;
using jarvis::overlay::MonitorArea;
using jarvis::overlay::monitorFor;
using jarvis::overlay::monitorNamed;
using jarvis::overlay::PixelRect;
using jarvis::overlay::Placement;
using jarvis::overlay::resizeFrom;
using jarvis::overlay::restorePlacement;
using jarvis::overlay::SavedPlacement;
using jarvis::overlay::savePlacement;
using jarvis::overlay::snapToEdges;

namespace {

constexpr PixelRect kArea{.left = 0, .top = 0, .right = 1920, .bottom = 1040};
constexpr int kThreshold = 16;

/// A 1920x1080 primary at 100 % and a 2560x1440 monitor at 150 % to its right.
std::vector<MonitorArea> twoMonitors() {
    return {
        MonitorArea{.name = "DISPLAY1", .work = kArea, .scale = 1.0F, .primary = true},
        MonitorArea{.name = "DISPLAY2",
                    .work = PixelRect{.left = 1920, .top = 0, .right = 4480, .bottom = 1400},
                    .scale = 1.5F,
                    .primary = false},
    };
}

}  // namespace

TEST_CASE("an edge within the threshold snaps onto the work area's edge", "[overlay][placement]") {
    const Placement nearLeft{.x = 12, .y = 400, .width = 300, .height = 200};
    REQUIRE(snapToEdges(nearLeft, kArea, kThreshold).x == 0);
    REQUIRE(snapToEdges(nearLeft, kArea, kThreshold).y == 400);

    const Placement nearBottomRight{.x = 1610, .y = 830, .width = 300, .height = 200};
    const Placement snapped = snapToEdges(nearBottomRight, kArea, kThreshold);
    REQUIRE(snapped == Placement{.x = 1620, .y = 840, .width = 300, .height = 200});
}

TEST_CASE("snapping both axes at once snaps into a corner", "[overlay][placement]") {
    const Placement nearCorner{.x = -9, .y = 14, .width = 300, .height = 200};
    REQUIRE(snapToEdges(nearCorner, kArea, kThreshold) ==
            Placement{.x = 0, .y = 0, .width = 300, .height = 200});
}

TEST_CASE("an edge beyond the threshold stays put", "[overlay][placement]") {
    const Placement free{.x = 17, .y = 500, .width = 300, .height = 200};
    REQUIRE(snapToEdges(free, kArea, kThreshold) == free);
    const Placement onBoundary{.x = 16, .y = 500, .width = 300, .height = 200};
    REQUIRE(snapToEdges(onBoundary, kArea, kThreshold).x == 0);
}

TEST_CASE("clamping moves a window fully on screen without resizing it", "[overlay][placement]") {
    const Placement inside{.x = 100, .y = 100, .width = 300, .height = 200};
    REQUIRE(clampInto(inside, kArea) == inside);

    const Placement offRight{.x = 1800, .y = 950, .width = 300, .height = 200};
    REQUIRE(clampInto(offRight, kArea) ==
            Placement{.x = 1620, .y = 840, .width = 300, .height = 200});

    const Placement offTopLeft{.x = -50, .y = -80, .width = 300, .height = 200};
    REQUIRE(clampInto(offTopLeft, kArea) == Placement{.x = 0, .y = 0, .width = 300, .height = 200});
}

TEST_CASE("a window larger than the area is aligned to its top-left", "[overlay][placement]") {
    const Placement huge{.x = 300, .y = 300, .width = 2500, .height = 1200};
    REQUIRE(clampInto(huge, kArea) == Placement{.x = 0, .y = 0, .width = 2500, .height = 1200});
}

TEST_CASE("the monitor is the one the window overlaps most", "[overlay][placement]") {
    const auto monitors = twoMonitors();
    const Placement mostlyRight{.x = 1800, .y = 100, .width = 400, .height = 200};
    REQUIRE(monitorFor(mostlyRight, monitors) == 1);
    const Placement mostlyLeft{.x = 1700, .y = 100, .width = 400, .height = 200};
    REQUIRE(monitorFor(mostlyLeft, monitors) == 0);
}

TEST_CASE("a window on no monitor belongs to the nearest one", "[overlay][placement]") {
    const auto monitors = twoMonitors();
    const Placement farRight{.x = 9000, .y = 100, .width = 300, .height = 200};
    REQUIRE(monitorFor(farRight, monitors) == 1);
    REQUIRE(monitorFor(farRight, std::vector<MonitorArea>{}) == 0);
}

TEST_CASE("a monitor is found by name, else the primary", "[overlay][placement]") {
    const auto monitors = twoMonitors();
    REQUIRE(monitorNamed(monitors, "DISPLAY2") == 1);
    REQUIRE(monitorNamed(monitors, "DISPLAY9") == 0);
}

TEST_CASE("a saved position is restored at its monitor's DPI", "[overlay][placement]") {
    const auto monitors = twoMonitors();
    const SavedPlacement saved{.monitor = "DISPLAY2", .x = 100.0F, .y = 40.0F};
    const Placement restored = restorePlacement(saved, monitors, 760, 300);
    REQUIRE(restored == Placement{.x = 1920 + 150, .y = 60, .width = 1140, .height = 450});
}

TEST_CASE("an unplugged monitor sends the HUD to the primary, fully on screen",
          "[overlay][placement]") {
    const auto monitors = twoMonitors();
    const SavedPlacement saved{.monitor = "DISPLAY2", .x = 2400.0F, .y = 1300.0F};
    const std::vector<MonitorArea> primaryOnly{monitors.at(0)};
    const Placement restored = restorePlacement(saved, primaryOnly, 760, 300);
    REQUIRE(restored == Placement{.x = 1160, .y = 740, .width = 760, .height = 300});
}

TEST_CASE("save and restore round-trip a position", "[overlay][placement]") {
    const auto monitors = twoMonitors();
    const Placement window{.x = 2400, .y = 300, .width = 1140, .height = 450};
    const SavedPlacement saved = savePlacement(window, monitors.at(1));
    REQUIRE(saved.monitor == "DISPLAY2");
    REQUIRE(saved.x == 320.0F);
    REQUIRE(saved.width == 760.0F);
    REQUIRE(restorePlacement(saved, monitors, 760, 300) == window);
}

TEST_CASE("the grab point stays under the cursor, also when the size changes",
          "[overlay][placement]") {
    const Placement window{.x = 100, .y = 100, .width = 400, .height = 200};
    const auto anchor = anchorFor(window, 200, 150);  // a quarter across, a quarter down
    REQUIRE(dragTo(anchor, 600, 500, 400, 200) ==
            Placement{.x = 500, .y = 450, .width = 400, .height = 200});
    // Crossing onto a 150 % monitor makes the window 600 x 300.
    REQUIRE(dragTo(anchor, 600, 500, 600, 300) ==
            Placement{.x = 450, .y = 425, .width = 600, .height = 300});
}

TEST_CASE("resizing never goes below the minimum", "[overlay][placement]") {
    const Placement start{.x = 10, .y = 10, .width = 900, .height = 640};
    REQUIRE(resizeFrom(start, 100, 50, 560, 420) ==
            Placement{.x = 10, .y = 10, .width = 1000, .height = 690});
    REQUIRE(resizeFrom(start, -800, -800, 560, 420) ==
            Placement{.x = 10, .y = 10, .width = 560, .height = 420});
}
