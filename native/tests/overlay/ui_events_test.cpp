#include "jarvis/overlay/ui_events.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <map>
#include <string>
#include <thread>
#include <variant>
#include <vector>

using jarvis::overlay::CloseRequested;
using jarvis::overlay::coalesces;
using jarvis::overlay::kindOf;
using jarvis::overlay::PlacementChanged;
using jarvis::overlay::QuitRequested;
using jarvis::overlay::RestartRequested;
using jarvis::overlay::SettingChanged;
using jarvis::overlay::SubmitText;
using jarvis::overlay::ToggleMute;
using jarvis::overlay::UiEvent;
using jarvis::overlay::UiEventQueue;

namespace {

std::string textOf(const UiEvent& event) {
    return std::get<SubmitText>(event).text;
}

}  // namespace

TEST_CASE("every event kind has the snake-case name Python dispatches on", "[overlay][events]") {
    REQUIRE(kindOf(SubmitText{.text = "hej"}) == "submit_text");
    REQUIRE(kindOf(CloseRequested{}) == "close");
    REQUIRE(kindOf(QuitRequested{}) == "quit");
    REQUIRE(kindOf(ToggleMute{}) == "toggle_mute");
    REQUIRE(kindOf(jarvis::overlay::SetExpanded{.expanded = true}) == "set_expanded");
    REQUIRE(kindOf(SettingChanged{.key = "k", .value = "v"}) == "setting_changed");
    REQUIRE(kindOf(jarvis::overlay::OpenLog{}) == "open_log");
    REQUIRE(kindOf(RestartRequested{}) == "restart");
    REQUIRE(kindOf(jarvis::overlay::SettingsOpened{}) == "settings_opened");
    REQUIRE(kindOf(PlacementChanged{}) == "placement_changed");
}

TEST_CASE("events come out in the order they went in, and drain empties", "[overlay][events]") {
    UiEventQueue queue;
    queue.push(SubmitText{.text = "first"});
    queue.push(ToggleMute{});
    queue.push(SubmitText{.text = "second"});

    const std::vector<UiEvent> events = queue.drain();
    REQUIRE(events.size() == 3);
    REQUIRE(textOf(events.at(0)) == "first");
    REQUIRE(std::holds_alternative<ToggleMute>(events.at(1)));
    REQUIRE(textOf(events.at(2)) == "second");
    REQUIRE(queue.size() == 0);
    REQUIRE(queue.drain().empty());
}

TEST_CASE("a setting edited repeatedly coalesces into its latest value", "[overlay][events]") {
    UiEventQueue queue;
    queue.push(SettingChanged{.key = "wake_word_threshold", .value = "0.70"});
    queue.push(SettingChanged{.key = "wake_word_threshold", .value = "0.75"});
    queue.push(SettingChanged{.key = "wake_word_threshold", .value = "0.80"});

    const std::vector<UiEvent> events = queue.drain();
    REQUIRE(events.size() == 1);
    REQUIRE(std::get<SettingChanged>(events.at(0)).value == "0.80");
}

TEST_CASE("coalescing never reorders past another event", "[overlay][events]") {
    UiEventQueue queue;
    queue.push(SettingChanged{.key = "tts_voice", .value = "a"});
    queue.push(RestartRequested{});
    queue.push(SettingChanged{.key = "tts_voice", .value = "b"});
    queue.push(SettingChanged{.key = "overlay_theme", .value = "amber"});

    const std::vector<UiEvent> events = queue.drain();
    REQUIRE(events.size() == 4);
    REQUIRE(std::get<SettingChanged>(events.at(0)).value == "a");
    REQUIRE(std::holds_alternative<RestartRequested>(events.at(1)));
    REQUIRE(std::get<SettingChanged>(events.at(2)).value == "b");
    REQUIRE(std::get<SettingChanged>(events.at(3)).key == "overlay_theme");
}

TEST_CASE("placements coalesce per panel, typed text never coalesces", "[overlay][events]") {
    REQUIRE(coalesces(PlacementChanged{.expanded = false}, PlacementChanged{.expanded = false}));
    REQUIRE_FALSE(
        coalesces(PlacementChanged{.expanded = false}, PlacementChanged{.expanded = true}));
    REQUIRE_FALSE(coalesces(SubmitText{.text = "a"}, SubmitText{.text = "a"}));
    REQUIRE_FALSE(coalesces(ToggleMute{}, ToggleMute{}));

    UiEventQueue queue;
    queue.push(ToggleMute{});
    queue.push(ToggleMute{});
    REQUIRE(queue.size() == 2);  // two clicks are two toggles
}

TEST_CASE("a full queue drops its oldest event and counts it", "[overlay][events]") {
    UiEventQueue queue(3);
    for (int i = 0; i < 5; ++i) {
        queue.push(SubmitText{.text = std::to_string(i)});
    }

    REQUIRE(queue.dropped() == 2);
    const std::vector<UiEvent> events = queue.drain();
    REQUIRE(events.size() == 3);
    REQUIRE(textOf(events.at(0)) == "2");
    REQUIRE(textOf(events.at(2)) == "4");
}

TEST_CASE("a capacity of zero still keeps the newest event", "[overlay][events]") {
    UiEventQueue queue(0);
    queue.push(QuitRequested{});
    REQUIRE(queue.capacity() == 1);
    REQUIRE(queue.drain().size() == 1);
}

TEST_CASE("concurrent producers lose nothing and keep their own order", "[overlay][events]") {
    constexpr int kProducers = 4;
    constexpr int kPerProducer = 500;
    constexpr auto kTotal = static_cast<std::size_t>(kProducers) * kPerProducer;
    UiEventQueue queue(kTotal);
    std::vector<UiEvent> collected;
    {
        std::vector<std::jthread> producers;
        producers.reserve(kProducers);
        for (int p = 0; p < kProducers; ++p) {
            producers.emplace_back([&queue, p] {
                for (int i = 0; i < kPerProducer; ++i) {
                    queue.push(SubmitText{.text = std::to_string(p) + ":" + std::to_string(i)});
                }
            });
        }
        // Drain while they push, as Python's 30 ms poll would.
        for (int round = 0; round < 50; ++round) {
            for (UiEvent& event : queue.drain()) {
                collected.push_back(std::move(event));
            }
        }
    }
    for (UiEvent& event : queue.drain()) {
        collected.push_back(std::move(event));
    }

    REQUIRE(collected.size() == kTotal);
    REQUIRE(queue.dropped() == 0);
    std::map<int, int> next;
    for (const UiEvent& event : collected) {
        const std::string text = textOf(event);
        const auto colon = text.find(':');
        const int producer = std::stoi(text.substr(0, colon));
        const int index = std::stoi(text.substr(colon + 1));
        REQUIRE(index == next[producer]);
        next[producer] = index + 1;
    }
}
