#include "jarvis/audio/silero_vad.hpp"

#include "jarvis/audio/audio_format.hpp"
#include "jarvis/common/errors.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

using jarvis::audio::kVadFrameSamples;
using jarvis::audio::SileroVad;
using Frame = std::array<float, kVadFrameSamples>;

namespace {

std::filesystem::path modelPath() {
    return std::filesystem::path{JARVIS_MODELS_DIR} / "silero_vad.onnx";
}

/// Deterministic pseudo-noise so the tests are repeatable without files.
Frame noiseFrame(std::uint32_t seed) {
    Frame frame{};
    std::uint32_t state = seed;
    for (auto& sample : frame) {
        state = (state * 1664525U) + 1013904223U;
        sample = ((static_cast<float>(state >> 8U) / 16777216.0F) * 0.2F) - 0.1F;
    }
    return frame;
}

void requireModel() {
    if (!std::filesystem::exists(modelPath())) {
        SKIP("silero_vad.onnx not downloaded; run scripts/download_models.py");
    }
}

}  // namespace

TEST_CASE("a missing model file raises ModelError", "[audio][vad][model]") {
    REQUIRE_THROWS_AS(SileroVad{std::filesystem::path{"does-not-exist.onnx"}}, jarvis::ModelError);
}

TEST_CASE("silence has a low speech probability", "[audio][vad][model]") {
    requireModel();
    SileroVad vad{modelPath()};
    const Frame silence{};

    float probability = 1.0F;
    for (int i = 0; i < 10; ++i) {
        probability = vad.speechProbability(silence);
    }

    REQUIRE(probability < 0.1F);
}

TEST_CASE("probabilities stay within [0, 1] on noise", "[audio][vad][model]") {
    requireModel();
    SileroVad vad{modelPath()};

    for (std::uint32_t seed = 1; seed <= 20; ++seed) {
        const float probability = vad.speechProbability(noiseFrame(seed));
        REQUIRE(probability >= 0.0F);
        REQUIRE(probability <= 1.0F);
    }
}

TEST_CASE("reset restores the initial state", "[audio][vad][model]") {
    requireModel();
    SileroVad vad{modelPath()};
    const Frame frame = noiseFrame(7);

    const float first = vad.speechProbability(frame);
    for (std::uint32_t seed = 1; seed <= 5; ++seed) {
        vad.speechProbability(noiseFrame(seed));
    }
    vad.reset();
    const float afterReset = vad.speechProbability(frame);

    REQUIRE_THAT(afterReset, Catch::Matchers::WithinAbs(first, 1e-5));
}
