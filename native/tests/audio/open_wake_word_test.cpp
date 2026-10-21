#include "jarvis/audio/open_wake_word.hpp"

#include "jarvis/audio/audio_format.hpp"
#include "jarvis/common/errors.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>

using jarvis::audio::kWakeWordChunkSamples;
using jarvis::audio::OpenWakeWord;
using Chunk = std::array<float, kWakeWordChunkSamples>;

namespace {

OpenWakeWord::ModelPaths modelPaths() {
    const std::filesystem::path modelsDir{JARVIS_MODELS_DIR};
    return OpenWakeWord::ModelPaths{.melspectrogram = modelsDir / "melspectrogram.onnx",
                                    .embedding = modelsDir / "embedding_model.onnx",
                                    .classifier = modelsDir / "hey_jarvis_v0.1.onnx"};
}

void requireModels() {
    const auto paths = modelPaths();
    if (!std::filesystem::exists(paths.melspectrogram) ||
        !std::filesystem::exists(paths.embedding) || !std::filesystem::exists(paths.classifier)) {
        SKIP("openWakeWord models not downloaded; run scripts/download_models.py");
    }
}

Chunk noiseChunk(std::uint32_t seed) {
    Chunk chunk{};
    std::uint32_t state = seed;
    for (auto& sample : chunk) {
        state = (state * 1664525U) + 1013904223U;
        sample = ((static_cast<float>(state >> 8U) / 16777216.0F) * 0.2F) - 0.1F;
    }
    return chunk;
}

// 76 mel frames (8 per chunk) plus 16 embeddings: the first score that can
// be non-zero arrives after ceil(76 / 8) + 15 = 25 chunks.
constexpr std::size_t kWarmUpChunks = 25;

}  // namespace

TEST_CASE("missing wake-word models raise ModelError", "[audio][wake_word][model]") {
    OpenWakeWord::ModelPaths paths = modelPaths();
    paths.classifier = "does-not-exist.onnx";
    REQUIRE_THROWS_AS(OpenWakeWord{paths}, jarvis::ModelError);
}

TEST_CASE("scores are zero until enough context has accumulated", "[audio][wake_word][model]") {
    requireModels();
    OpenWakeWord detector{modelPaths()};
    const Chunk silence{};

    for (std::size_t i = 0; i < kWarmUpChunks - 1; ++i) {
        REQUIRE(detector.score(silence) == 0.0F);
    }
}

TEST_CASE("silence does not trigger the wake word", "[audio][wake_word][model]") {
    requireModels();
    OpenWakeWord detector{modelPaths()};
    const Chunk silence{};

    for (std::size_t i = 0; i < kWarmUpChunks + 20; ++i) {
        REQUIRE(detector.score(silence) < 0.5F);
    }
}

TEST_CASE("noise keeps scores within [0, 1] and below the threshold", "[audio][wake_word][model]") {
    requireModels();
    OpenWakeWord detector{modelPaths()};

    for (std::uint32_t seed = 1; seed <= kWarmUpChunks + 20; ++seed) {
        const float score = detector.score(noiseChunk(seed));
        REQUIRE(score >= 0.0F);
        REQUIRE(score <= 1.0F);
        REQUIRE(score < 0.5F);
    }
}

TEST_CASE("reset forgets accumulated context", "[audio][wake_word][model]") {
    requireModels();
    OpenWakeWord detector{modelPaths()};
    const Chunk silence{};
    for (std::size_t i = 0; i < kWarmUpChunks + 5; ++i) {
        detector.score(silence);
    }

    detector.reset();

    REQUIRE(detector.score(silence) == 0.0F);
}
