#include "jarvis/inference/whisper_transcriber.hpp"

#include "jarvis/common/errors.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <numbers>
#include <vector>

using jarvis::inference::kWhisperSampleRate;
using jarvis::inference::TranscribeOptions;
using jarvis::inference::WhisperTranscriber;

namespace {

std::filesystem::path modelPath() {
    return std::filesystem::path{JARVIS_MODELS_DIR} / "ggml-tiny.bin";
}

void requireModel() {
    if (!std::filesystem::exists(modelPath())) {
        SKIP("ggml-tiny.bin not downloaded; run scripts/download_models.py --local");
    }
}

std::vector<float> silence(double seconds) {
    return std::vector<float>(static_cast<std::size_t>(seconds * kWhisperSampleRate), 0.0F);
}

std::vector<float> tone(double seconds, float frequencyHz) {
    std::vector<float> samples(static_cast<std::size_t>(seconds * kWhisperSampleRate));
    std::size_t index = 0;
    std::ranges::generate(samples, [&] {
        const auto t = static_cast<float>(index++) / static_cast<float>(kWhisperSampleRate);
        return 0.3F * std::sin(2.0F * std::numbers::pi_v<float> * frequencyHz * t);
    });
    return samples;
}

/// Whisper tiny occasionally hallucinates a short token on silence
/// ("[BLANK_AUDIO]", "you"); anything longer than this is a real failure.
constexpr std::size_t kNearEmptyChars = 32;

}  // namespace

TEST_CASE("a missing whisper model raises ModelError", "[inference][whisper][model]") {
    REQUIRE_THROWS_AS(WhisperTranscriber{std::filesystem::path{"does-not-exist.bin"}},
                      jarvis::ModelError);
}

TEST_CASE("one second of silence yields a near-empty transcript", "[inference][whisper][model]") {
    requireModel();
    WhisperTranscriber transcriber{modelPath()};

    const auto transcript =
        transcriber.transcribe(silence(1.0), TranscribeOptions{.language = "en"});

    REQUIRE(transcript.text.size() <= kNearEmptyChars);
    REQUIRE(transcript.language == "en");
}

TEST_CASE("a synthetic tone is transcribed without error", "[inference][whisper][model]") {
    requireModel();
    WhisperTranscriber transcriber{modelPath()};

    const auto transcript = transcriber.transcribe(tone(1.5, 440.0F));

    REQUIRE(transcript.segments.size() <= 4);
    for (const auto& segment : transcript.segments) {
        REQUIRE(segment.startMs >= 0);
        REQUIRE(segment.endMs >= segment.startMs);
        REQUIRE_FALSE(segment.text.empty());
    }
}

TEST_CASE("empty audio returns an empty transcript", "[inference][whisper][model]") {
    requireModel();
    WhisperTranscriber transcriber{modelPath()};

    const auto transcript = transcriber.transcribe({});

    REQUIRE(transcript.text.empty());
    REQUIRE(transcript.segments.empty());
}

TEST_CASE("the tiny model is multilingual and accepts beam search", "[inference][whisper][model]") {
    requireModel();
    WhisperTranscriber transcriber{modelPath()};
    REQUIRE(transcriber.multilingual());

    const TranscribeOptions options{.language = "en",
                                    .strategy = jarvis::inference::DecodingStrategy::BeamSearch,
                                    .beamSize = 2};
    REQUIRE_NOTHROW(transcriber.transcribe(silence(0.5), options));
}
