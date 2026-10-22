#include "jarvis/audio/pipeline.hpp"

#include "jarvis/audio/audio_format.hpp"
#include "jarvis/audio/events.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "audio/fakes.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

using jarvis::audio::AudioEvent;
using jarvis::audio::AudioEventType;
using jarvis::audio::AudioPipeline;
using jarvis::audio::kVadFrameSamples;
using jarvis::audio::kWakeWordChunkSamples;
using jarvis::audio::PipelineConfig;
using jarvis::audio::testing::FakeVad;
using jarvis::audio::testing::FakeWakeWord;
using jarvis::audio::testing::samples;

namespace {

constexpr std::size_t kMinSpeechFrames = 7;
constexpr std::size_t kSilenceEndFrames = 22;
constexpr int kCooldownMs = 2000;
// 2000 ms of cooldown is 32000 samples, i.e. 25 chunks of 1280.
constexpr std::size_t kCooldownChunks = 25;

struct Fixture {
    FakeVad* vad;
    FakeWakeWord* wakeWord;
    std::vector<AudioEvent> events;
    AudioPipeline pipeline;

    explicit Fixture(PipelineConfig config = {}, bool withWakeWord = true)
        : Fixture(std::make_unique<FakeVad>(),
                  withWakeWord ? std::make_unique<FakeWakeWord>() : nullptr, config) {}

    void feedFrames(std::size_t frames, float value = 0.0F) {
        pipeline.process(samples(frames * kVadFrameSamples, value));
    }

    void feedChunks(std::size_t chunks, float value = 0.0F) {
        pipeline.process(samples(chunks * kWakeWordChunkSamples, value));
    }

    [[nodiscard]] std::size_t count(AudioEventType type) const {
        return static_cast<std::size_t>(
            std::ranges::count_if(events, [type](const auto& e) { return e.type == type; }));
    }

private:
    Fixture(std::unique_ptr<FakeVad> ownedVad, std::unique_ptr<FakeWakeWord> ownedWakeWord,
            PipelineConfig config)
        : vad(ownedVad.get()),
          wakeWord(ownedWakeWord.get()),
          pipeline(std::move(ownedVad), std::move(ownedWakeWord), config,
                   [this](const AudioEvent& event) { events.push_back(event); }) {}
};

PipelineConfig configWithCooldown() {
    PipelineConfig config;
    config.wakeWordCooldownMs = kCooldownMs;
    return config;
}

}  // namespace

TEST_CASE("a wake-word score above the threshold is reported once", "[audio][pipeline]") {
    Fixture f;
    f.wakeWord->script({0.1F, 0.9F, 0.1F});

    f.feedChunks(3);

    REQUIRE(f.events.size() == 1);
    REQUIRE(f.events.at(0).type == AudioEventType::WakeWordDetected);
    REQUIRE_THAT(f.events.at(0).value, Catch::Matchers::WithinAbs(0.9, 1e-6));
    REQUIRE(f.events.at(0).samplePosition <= 3 * kWakeWordChunkSamples);
}

TEST_CASE("the cooldown suppresses repeated detections", "[audio][pipeline]") {
    Fixture f{configWithCooldown()};
    // A detector that fires on every single chunk.
    for (std::size_t i = 0; i < kCooldownChunks + 3; ++i) {
        f.wakeWord->script({0.9F});
    }

    f.feedChunks(1 + kCooldownChunks);
    REQUIRE(f.count(AudioEventType::WakeWordDetected) == 1);

    f.feedChunks(1);
    REQUIRE(f.count(AudioEventType::WakeWordDetected) == 2);
}

TEST_CASE("speech start and end are reported from VAD probabilities", "[audio][pipeline]") {
    Fixture f;
    f.vad->scriptRepeated(0.9F, kMinSpeechFrames);

    f.feedFrames(kMinSpeechFrames);
    REQUIRE(f.count(AudioEventType::SpeechStarted) == 1);
    REQUIRE(f.count(AudioEventType::SpeechEnded) == 0);

    f.feedFrames(kSilenceEndFrames);
    REQUIRE(f.count(AudioEventType::SpeechEnded) == 1);
}

TEST_CASE("frames are assembled across arbitrary batch sizes", "[audio][pipeline]") {
    Fixture f;

    f.pipeline.process(samples(100));
    f.pipeline.process(samples(500));
    REQUIRE(f.vad->framesSeen == 1);

    f.pipeline.process(samples(kVadFrameSamples * 3));
    REQUIRE(f.vad->framesSeen == 4);
    REQUIRE(f.pipeline.samplesProcessed() == 600 + (kVadFrameSamples * 3));
}

TEST_CASE("listening captures audio seeded with the pre-roll", "[audio][pipeline]") {
    PipelineConfig config;
    config.preRollMs = 300;  // 4800 samples
    Fixture f{config};
    const std::size_t preRoll = 4800;

    f.pipeline.process(samples(6000, 1.0F));
    f.pipeline.setListening(true);
    f.pipeline.process(samples(1000, 2.0F));

    const auto captured = f.pipeline.takeCaptured();
    REQUIRE(captured.size() == preRoll + 1000);
    REQUIRE(
        std::ranges::all_of(std::span{captured}.first(preRoll), [](float v) { return v == 1.0F; }));
    REQUIRE(std::ranges::all_of(std::span{captured}.last(1000), [](float v) { return v == 2.0F; }));
    REQUIRE(f.pipeline.takeCaptured().empty());
}

TEST_CASE("audio is not captured while not listening", "[audio][pipeline]") {
    Fixture f;

    f.pipeline.process(samples(3000, 1.0F));

    REQUIRE(f.pipeline.takeCaptured().empty());
}

TEST_CASE("wake-word detection pauses while listening", "[audio][pipeline]") {
    Fixture f;
    f.pipeline.setListening(true);

    f.feedChunks(5);

    REQUIRE(f.wakeWord->chunksSeen == 0);
    REQUIRE(f.events.empty());

    f.pipeline.setListening(false);
    f.feedChunks(1);
    REQUIRE(f.wakeWord->chunksSeen == 1);
}

TEST_CASE("the pipeline works without a wake-word detector", "[audio][pipeline]") {
    Fixture f{PipelineConfig{}, false};
    f.vad->scriptRepeated(0.9F, kMinSpeechFrames);

    f.feedChunks(4);

    REQUIRE(f.count(AudioEventType::SpeechStarted) == 1);
    REQUIRE(f.count(AudioEventType::WakeWordDetected) == 0);
}

TEST_CASE("input level tracks the most recent frame", "[audio][pipeline]") {
    Fixture f;

    f.feedFrames(1, 0.5F);

    REQUIRE_THAT(f.pipeline.inputLevel(), Catch::Matchers::WithinAbs(0.5, 1e-5));
}

TEST_CASE("reset clears detectors, buffers and listening", "[audio][pipeline]") {
    Fixture f;
    f.pipeline.setListening(true);
    f.pipeline.process(samples(700, 1.0F));

    f.pipeline.reset();

    REQUIRE(f.vad->resets == 1);
    REQUIRE(f.wakeWord->resets == 1);
    REQUIRE_FALSE(f.pipeline.listening());
    REQUIRE(f.pipeline.takeCaptured().empty());
    REQUIRE(f.pipeline.inputLevel() == 0.0F);
}

TEST_CASE("the wake-word threshold can be changed while running", "[audio][pipeline]") {
    Fixture f;
    f.wakeWord->script({0.7F, 0.7F});
    f.pipeline.setWakeWordThreshold(0.8F);
    f.feedChunks(1);
    REQUIRE(f.count(AudioEventType::WakeWordDetected) == 0);

    f.pipeline.setWakeWordThreshold(0.6F);
    f.feedChunks(1);
    REQUIRE(f.count(AudioEventType::WakeWordDetected) == 1);
}

TEST_CASE("the wake-word threshold stays a usable probability", "[audio][pipeline]") {
    Fixture f;
    f.pipeline.setWakeWordThreshold(0.0F);
    REQUIRE(f.pipeline.wakeWordThreshold() > 0.0F);
    f.pipeline.setWakeWordThreshold(1.5F);
    REQUIRE(f.pipeline.wakeWordThreshold() < 1.0F);
}
