#include "jarvis/overlay/model.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using jarvis::overlay::AssistantState;
using jarvis::overlay::LevelRing;
using jarvis::overlay::ModelLimits;
using jarvis::overlay::OverlayModel;
using jarvis::overlay::parseAssistantState;
using jarvis::overlay::toString;

namespace {

constexpr std::chrono::milliseconds kFlash{600};

OverlayModel::Clock::time_point at(int millis) {
    return OverlayModel::Clock::time_point{std::chrono::milliseconds(millis)};
}

}  // namespace

TEST_CASE("state names round-trip through their Python spelling", "[overlay][model]") {
    for (const auto state : {AssistantState::Idle, AssistantState::Listening,
                             AssistantState::Thinking, AssistantState::Speaking}) {
        REQUIRE(parseAssistantState(toString(state)) == state);
    }
    REQUIRE(toString(AssistantState::Listening) == "LISTENING");
    REQUIRE_FALSE(parseAssistantState("listening").has_value());
    REQUIRE_FALSE(parseAssistantState("").has_value());
}

TEST_CASE("a fresh level ring is silent and fixed in size", "[overlay][model]") {
    const LevelRing ring{4};

    REQUIRE(ring.capacity() == 4);
    REQUIRE(ring.ordered() == std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F});
    REQUIRE(ring.latest() == 0.0F);
}

TEST_CASE("the level ring wraps around and keeps chronological order", "[overlay][model]") {
    LevelRing ring{3};
    ring.push(0.1F);
    ring.push(0.2F);
    ring.push(0.3F);
    REQUIRE(ring.ordered() == std::vector<float>{0.1F, 0.2F, 0.3F});

    ring.push(0.4F);
    ring.push(0.5F);
    REQUIRE(ring.ordered() == std::vector<float>{0.3F, 0.4F, 0.5F});
    REQUIRE(ring.latest() == 0.5F);
}

TEST_CASE("levels are clamped to the unit range", "[overlay][model]") {
    LevelRing ring{2};
    ring.push(-3.0F);
    ring.push(7.0F);

    REQUIRE(ring.ordered() == std::vector<float>{0.0F, 1.0F});
}

TEST_CASE("clearing the ring resets every sample", "[overlay][model]") {
    LevelRing ring{2};
    ring.push(0.9F);
    ring.clear();

    REQUIRE(ring.ordered() == std::vector<float>{0.0F, 0.0F});
}

TEST_CASE("a zero-capacity ring still holds one sample", "[overlay][model]") {
    LevelRing ring{0};
    ring.push(0.5F);
    REQUIRE(ring.capacity() == 1);
    REQUIRE(ring.latest() == 0.5F);
}

TEST_CASE("the model starts idle, visible and empty", "[overlay][model]") {
    const OverlayModel model{ModelLimits{.levelHistory = 8}};
    const auto snapshot = model.snapshot();

    REQUIRE(snapshot.state == AssistantState::Idle);
    REQUIRE(snapshot.visible);
    REQUIRE(snapshot.levels.size() == 8);
    REQUIRE(snapshot.transcript.empty());
    REQUIRE(snapshot.response.empty());
    REQUIRE(snapshot.wakeFlash == 0.0F);
    REQUIRE(model.revision() == 0);
}

TEST_CASE("state transitions are reflected in the snapshot and bump the revision",
          "[overlay][model]") {
    OverlayModel model;
    for (const auto state : {AssistantState::Listening, AssistantState::Thinking,
                             AssistantState::Speaking, AssistantState::Idle}) {
        const auto before = model.revision();
        model.setState(state);
        REQUIRE(model.state() == state);
        REQUIRE(model.snapshot().state == state);
        REQUIRE(model.revision() == before + 1);
    }
}

TEST_CASE("setting the same state again is not a change", "[overlay][model]") {
    OverlayModel model;
    model.setState(AssistantState::Listening);
    const auto revision = model.revision();

    model.setState(AssistantState::Listening);

    REQUIRE(model.revision() == revision);
}

TEST_CASE("returning to idle flattens the waveform", "[overlay][model]") {
    OverlayModel model{ModelLimits{.levelHistory = 4}};
    model.setState(AssistantState::Listening);
    model.pushLevel(0.8F);
    model.pushLevel(0.6F);
    REQUIRE(model.snapshot().levels.back() == 0.6F);

    model.setState(AssistantState::Idle);

    REQUIRE(model.snapshot().levels == std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F});
}

TEST_CASE("pushed levels arrive oldest first", "[overlay][model]") {
    OverlayModel model{ModelLimits{.levelHistory = 3}};
    model.pushLevel(0.1F);
    model.pushLevel(0.2F);
    model.pushLevel(0.3F);
    model.pushLevel(0.4F);

    REQUIRE(model.snapshot().levels == std::vector<float>{0.2F, 0.3F, 0.4F});
}

TEST_CASE("text is bounded to the configured width, keeping the tail", "[overlay][model]") {
    OverlayModel model{ModelLimits{.maxTextChars = 6}};
    model.setTranscript("hello world");
    model.setResponse("short");

    REQUIRE(model.snapshot().transcript == "…world");
    REQUIRE(model.snapshot().response == "short");
}

TEST_CASE("visibility toggles and only changes count as revisions", "[overlay][model]") {
    OverlayModel model;
    model.setVisible(true);
    REQUIRE(model.revision() == 0);

    model.setVisible(false);
    REQUIRE_FALSE(model.visible());
    REQUIRE_FALSE(model.snapshot().visible);
    REQUIRE(model.revision() == 1);

    model.setVisible(true);
    REQUIRE(model.visible());
}

TEST_CASE("the wake-word flash decays linearly to zero", "[overlay][model]") {
    OverlayModel model{ModelLimits{.wakeFlashDuration = kFlash}};
    model.flashWakeWord(at(1000));

    REQUIRE_THAT(model.snapshot(at(1000)).wakeFlash, WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(model.snapshot(at(1300)).wakeFlash, WithinAbs(0.5, 1e-6));
    REQUIRE_THAT(model.snapshot(at(1600)).wakeFlash, WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(model.snapshot(at(5000)).wakeFlash, WithinAbs(0.0, 1e-6));
}

TEST_CASE("a zero flash duration disables the flash", "[overlay][model]") {
    OverlayModel model{ModelLimits{.wakeFlashDuration = std::chrono::milliseconds{0}}};
    model.flashWakeWord(at(10));

    REQUIRE(model.snapshot(at(10)).wakeFlash == 0.0F);
}
