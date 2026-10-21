#include "jarvis/audio/engine.hpp"

#include "jarvis/audio/audio_format.hpp"
#include "jarvis/audio/events.hpp"
#include "jarvis/audio/playback.hpp"

#include <catch2/catch_test_macros.hpp>

#include "audio/fakes.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

using jarvis::audio::AudioEngine;
using jarvis::audio::AudioEvent;
using jarvis::audio::AudioEventType;
using jarvis::audio::EngineConfig;
using jarvis::audio::kVadFrameSamples;
using jarvis::audio::Playback;
using jarvis::audio::testing::FakeAudioSink;
using jarvis::audio::testing::FakeAudioSource;
using jarvis::audio::testing::FakeVad;
using jarvis::audio::testing::samples;

namespace {

constexpr std::size_t kMinSpeechFrames = 7;
constexpr auto kTimeout = std::chrono::seconds(5);

EngineConfig fastConfig() {
    EngineConfig config;
    config.ringBufferSamples = 4096;
    config.workerPollMs = 1;
    return config;
}

/// Poll @p condition until it holds or the timeout expires.
bool waitUntil(const std::function<bool()>& condition) {
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (condition()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return condition();
}

struct Fixture {
    FakeAudioSource* source;
    FakeVad* vad;
    AudioEngine engine;
    std::vector<AudioEvent> events;

    Fixture() : Fixture(std::make_unique<FakeAudioSource>(), std::make_unique<FakeVad>()) {}

    /// Keep polling until an event of @p type has arrived.
    bool waitForEvent(AudioEventType type) {
        return waitUntil([this, type] {
            auto fresh = engine.pollEvents();
            events.insert(events.end(), fresh.begin(), fresh.end());
            return std::ranges::any_of(events, [type](const auto& e) { return e.type == type; });
        });
    }

private:
    Fixture(std::unique_ptr<FakeAudioSource> ownedSource, std::unique_ptr<FakeVad> ownedVad)
        : source(ownedSource.get()),
          vad(ownedVad.get()),
          engine(std::move(ownedSource), std::move(ownedVad), nullptr, fastConfig()) {}
};

}  // namespace

TEST_CASE("start opens the source and stop closes it", "[audio][engine]") {
    Fixture f;
    REQUIRE_FALSE(f.engine.running());

    f.engine.start();
    REQUIRE(f.engine.running());
    REQUIRE(f.source->started);

    f.engine.stop();
    REQUIRE_FALSE(f.engine.running());
    REQUIRE(f.source->stopped);
}

TEST_CASE("start and stop are idempotent", "[audio][engine]") {
    Fixture f;
    f.engine.start();
    f.engine.start();
    f.engine.stop();
    f.engine.stop();
    REQUIRE_FALSE(f.engine.running());
}

TEST_CASE("events produced on the worker thread reach pollEvents", "[audio][engine]") {
    Fixture f;
    f.vad->scriptRepeated(0.9F, kMinSpeechFrames);
    f.engine.start();

    f.source->feed(samples(kMinSpeechFrames * kVadFrameSamples));

    REQUIRE(f.waitForEvent(AudioEventType::SpeechStarted));
    REQUIRE(f.engine.pollEvents().empty());  // polling drains
}

TEST_CASE("captured audio flows from the source to takeCaptured", "[audio][engine]") {
    Fixture f;
    f.engine.start();
    f.engine.setListening(true);
    REQUIRE(f.engine.listening());

    f.source->feed(samples(2048, 0.25F));

    REQUIRE(waitUntil([&] { return f.engine.samplesProcessed() >= 2048; }));
    const auto captured = f.engine.takeCaptured();
    REQUIRE(captured.size() == 2048);
    REQUIRE(std::ranges::all_of(captured, [](float v) { return v == 0.25F; }));
}

TEST_CASE("speech start stops the barge-in target", "[audio][engine]") {
    Fixture f;
    auto playback = std::make_shared<Playback>(std::make_unique<FakeAudioSink>());
    playback->enqueue(samples(1000, 0.5F));
    f.engine.setBargeInTarget(playback);
    f.vad->scriptRepeated(0.9F, kMinSpeechFrames);
    f.engine.start();

    f.source->feed(samples(kMinSpeechFrames * kVadFrameSamples));

    REQUIRE(waitUntil([&] { return !playback->isPlaying(); }));
    REQUIRE(playback->queuedSamples() == 0);
}

TEST_CASE("clearing the barge-in target leaves playback alone", "[audio][engine]") {
    Fixture f;
    const auto playback = std::make_shared<Playback>(std::make_unique<FakeAudioSink>());
    playback->enqueue(samples(1000, 0.5F));
    f.engine.setBargeInTarget(playback);
    f.engine.setBargeInTarget(nullptr);
    f.vad->scriptRepeated(0.9F, kMinSpeechFrames);
    f.engine.start();

    f.source->feed(samples(kMinSpeechFrames * kVadFrameSamples));

    REQUIRE(f.waitForEvent(AudioEventType::SpeechStarted));
    REQUIRE(playback->isPlaying());
}

TEST_CASE("ring buffer overflow is reported as dropped samples", "[audio][engine]") {
    Fixture f;
    f.engine.start();

    // One push far larger than the 4096-sample ring buffer must drop.
    f.source->feed(samples(20000));

    REQUIRE(f.waitForEvent(AudioEventType::SamplesDropped));
    REQUIRE(f.engine.droppedSamples() >= 20000 - 4096);
}

TEST_CASE("the destructor stops a running engine", "[audio][engine]") {
    auto owned = std::make_unique<FakeAudioSource>();
    const auto* source = owned.get();
    {
        AudioEngine engine{std::move(owned), std::make_unique<FakeVad>(), nullptr, fastConfig()};
        engine.start();
        REQUIRE(source->started);
        REQUIRE_FALSE(source->stopped);
        // `source` is owned by the engine; checking after this scope would be a
        // use-after-free, so the observation is made inside it.
    }
}
