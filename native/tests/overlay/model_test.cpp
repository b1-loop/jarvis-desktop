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

TEST_CASE("the boot sequence starts once, however often it is requested", "[overlay][model]") {
    OverlayModel model;
    REQUIRE_FALSE(model.snapshot().bootStartedAt.has_value());

    REQUIRE(model.playBoot(at(100)));
    const auto revision = model.revision();
    REQUIRE_FALSE(model.playBoot(at(900)));
    REQUIRE(model.revision() == revision);
    REQUIRE(model.snapshot().bootStartedAt == at(100));
}

TEST_CASE("a boot can be skipped once, and only after it started", "[overlay][model]") {
    OverlayModel model;
    REQUIRE_FALSE(model.skipBoot(at(50)));

    model.playBoot(at(100));
    REQUIRE(model.skipBoot(at(400)));
    REQUIRE_FALSE(model.skipBoot(at(500)));
    REQUIRE(model.snapshot().bootSkippedAt == at(400));
}

TEST_CASE("leaving Idle during the boot skips it", "[overlay][model]") {
    OverlayModel model;
    model.playBoot(at(0));
    model.setState(AssistantState::Idle, at(100));  // no change, no skip
    REQUIRE_FALSE(model.snapshot().bootSkippedAt.has_value());

    model.setState(AssistantState::Listening, at(700));
    REQUIRE(model.snapshot().bootSkippedAt == at(700));
}

TEST_CASE("the boot readout is kept as given, within its limits", "[overlay][model][boot]") {
    ModelLimits limits;
    limits.maxReadoutLines = 2;
    limits.maxReadoutLabelChars = 5;
    limits.maxReadoutValueChars = 8;
    OverlayModel model{limits};
    const jarvis::overlay::BootReadout readout{
        {.label = "MODEL", .value = "qwen2.5:7b-instruct"},
        {.label = "MICROPHONE", .value = "Mic"},
        {.label = "DROPPED", .value = "never shown"},
    };
    REQUIRE(model.playBoot(readout, at(0)));
    const auto snapshot = model.snapshot(at(1));
    REQUIRE(snapshot.bootReadout != nullptr);
    REQUIRE(snapshot.bootReadout->size() == 2);
    REQUIRE(snapshot.bootReadout->at(0).label == "MODEL");
    REQUIRE(snapshot.bootReadout->at(0).value == "qwen2.5…");
    REQUIRE(snapshot.bootReadout->at(1).label == "MICR…");
    REQUIRE(snapshot.bootReadout->at(1).value == "Mic");
}

TEST_CASE("a replay restarts a finished or skipped boot from the top", "[overlay][model][boot]") {
    OverlayModel model;
    REQUIRE(model.playBoot({{.label = "MIC", .value = "Old mic"}}, at(0)));
    REQUIRE(model.skipBoot(at(300)));
    const auto revision = model.revision();

    model.replayBoot({{.label = "MIC", .value = "New mic"}}, at(9000));

    const auto snapshot = model.snapshot(at(9001));
    REQUIRE(model.revision() > revision);
    REQUIRE(snapshot.bootStartedAt == at(9000));
    REQUIRE_FALSE(snapshot.bootSkippedAt.has_value());
    REQUIRE(snapshot.bootReadout->front().value == "New mic");
    REQUIRE(model.skipBoot(at(9100)));  // the replay can be skipped like the first boot
}

TEST_CASE("a replay plays even when no boot played before", "[overlay][model][boot]") {
    OverlayModel model;
    model.replayBoot({}, at(50));
    REQUIRE(model.snapshot(at(60)).bootStartedAt == at(50));
    REQUIRE_FALSE(model.playBoot(at(70)));  // the start-up boot counts as played
}

TEST_CASE("a replay restarts a boot that is still playing", "[overlay][model][boot]") {
    OverlayModel model;
    model.playBoot(at(0));
    model.replayBoot({}, at(1200));
    REQUIRE(model.snapshot(at(1300)).bootStartedAt == at(1200));
}

TEST_CASE("a replay during a conversation only glides into place", "[overlay][model][boot]") {
    OverlayModel model;
    model.setState(AssistantState::Speaking, at(0));
    model.replayBoot({{.label = "MODE", .value = "CLOUD"}}, at(500));
    const auto snapshot = model.snapshot(at(510));
    REQUIRE(snapshot.bootStartedAt == at(500));
    REQUIRE(snapshot.bootSkippedAt == at(500));
    REQUIRE(snapshot.bootReadout->front().value == "CLOUD");
}

TEST_CASE("a replayed readout is bounded like the first one", "[overlay][model][boot]") {
    ModelLimits limits;
    limits.maxReadoutLines = 1;
    OverlayModel model{limits};
    model.replayBoot({{.label = "A", .value = "1"}, {.label = "B", .value = "2"}}, at(0));
    REQUIRE(model.snapshot(at(1)).bootReadout->size() == 1);
}

TEST_CASE("a repeated boot request keeps the first readout", "[overlay][model][boot]") {
    OverlayModel model;
    REQUIRE(model.playBoot({{.label = "MODE", .value = "CLOUD"}}, at(0)));
    REQUIRE_FALSE(model.playBoot({{.label = "MODE", .value = "LOCAL"}}, at(10)));
    REQUIRE(model.snapshot(at(20)).bootReadout->front().value == "CLOUD");
    OverlayModel plain;
    REQUIRE(plain.playBoot(at(0)));
    REQUIRE(plain.snapshot(at(1)).bootReadout->empty());
    REQUIRE(OverlayModel{}.snapshot(at(0)).bootReadout == nullptr);
}
