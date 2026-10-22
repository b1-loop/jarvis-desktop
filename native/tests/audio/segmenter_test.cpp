#include "jarvis/audio/vad.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>

using jarvis::audio::SegmenterConfig;
using jarvis::audio::SpeechSegmenter;
using Transition = SpeechSegmenter::Transition;

namespace {

// 512-sample frames at 16 kHz are 32 ms: 200 ms needs 7 frames, 700 ms needs 22.
constexpr std::size_t kMinSpeechFrames = 7;
constexpr std::size_t kSilenceEndFrames = 22;

SegmenterConfig defaultConfig() {
    return SegmenterConfig{
        .startThreshold = 0.5F, .endThreshold = 0.35F, .minSpeechMs = 200, .silenceEndMs = 700};
}

Transition feed(SpeechSegmenter& segmenter, float probability, std::size_t frames) {
    Transition last = Transition::None;
    for (std::size_t i = 0; i < frames; ++i) {
        last = segmenter.update(probability);
    }
    return last;
}

}  // namespace

TEST_CASE("silence produces no transitions", "[audio][segmenter]") {
    SpeechSegmenter segmenter{defaultConfig()};

    REQUIRE(feed(segmenter, 0.1F, 50) == Transition::None);
    REQUIRE_FALSE(segmenter.inSpeech());
}

TEST_CASE("speech starts once it has lasted the minimum duration", "[audio][segmenter]") {
    SpeechSegmenter segmenter{defaultConfig()};

    REQUIRE(feed(segmenter, 0.9F, kMinSpeechFrames - 1) == Transition::None);
    REQUIRE_FALSE(segmenter.inSpeech());
    REQUIRE(segmenter.update(0.9F) == Transition::SpeechStarted);
    REQUIRE(segmenter.inSpeech());
}

TEST_CASE("a brief blip below the start threshold resets the count", "[audio][segmenter]") {
    SpeechSegmenter segmenter{defaultConfig()};

    feed(segmenter, 0.9F, kMinSpeechFrames - 1);
    REQUIRE(segmenter.update(0.1F) == Transition::None);
    REQUIRE(feed(segmenter, 0.9F, kMinSpeechFrames - 1) == Transition::None);
    REQUIRE_FALSE(segmenter.inSpeech());
}

TEST_CASE("speech ends after enough continuous silence", "[audio][segmenter]") {
    SpeechSegmenter segmenter{defaultConfig()};
    feed(segmenter, 0.9F, kMinSpeechFrames);

    REQUIRE(feed(segmenter, 0.1F, kSilenceEndFrames - 1) == Transition::None);
    REQUIRE(segmenter.inSpeech());
    REQUIRE(segmenter.update(0.1F) == Transition::SpeechEnded);
    REQUIRE_FALSE(segmenter.inSpeech());
}

TEST_CASE("probabilities between the thresholds keep speech alive", "[audio][segmenter]") {
    SpeechSegmenter segmenter{defaultConfig()};
    feed(segmenter, 0.9F, kMinSpeechFrames);

    // 0.4 is below the start threshold but above the end threshold.
    REQUIRE(feed(segmenter, 0.4F, kSilenceEndFrames * 2) == Transition::None);
    REQUIRE(segmenter.inSpeech());
}

TEST_CASE("a pause shorter than the silence window does not end speech", "[audio][segmenter]") {
    SpeechSegmenter segmenter{defaultConfig()};
    feed(segmenter, 0.9F, kMinSpeechFrames);

    feed(segmenter, 0.1F, kSilenceEndFrames - 1);
    REQUIRE(segmenter.update(0.9F) == Transition::None);
    REQUIRE(feed(segmenter, 0.1F, kSilenceEndFrames - 1) == Transition::None);
    REQUIRE(segmenter.inSpeech());
}

TEST_CASE("reset returns to silence", "[audio][segmenter]") {
    SpeechSegmenter segmenter{defaultConfig()};
    feed(segmenter, 0.9F, kMinSpeechFrames);

    segmenter.reset();

    REQUIRE_FALSE(segmenter.inSpeech());
    REQUIRE(feed(segmenter, 0.9F, kMinSpeechFrames - 1) == Transition::None);
}

TEST_CASE("a dip between the thresholds holds the start count", "[audio][segmenter]") {
    // A word's quieter middle scores between the thresholds: not silence, so
    // the frames already counted are kept rather than thrown away.
    SpeechSegmenter segmenter{defaultConfig()};

    feed(segmenter, 0.9F, kMinSpeechFrames - 1);
    REQUIRE(feed(segmenter, 0.4F, 3) == Transition::None);
    REQUIRE_FALSE(segmenter.inSpeech());
    REQUIRE(segmenter.update(0.9F) == Transition::SpeechStarted);
}
