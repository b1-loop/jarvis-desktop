#include "jarvis/audio/silero_vad.hpp"

#include "jarvis/audio/audio_format.hpp"
#include "jarvis/audio/mp3_decoder.hpp"
#include "jarvis/audio/vad.hpp"
#include "jarvis/common/errors.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <vector>

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

namespace {

std::vector<std::byte> readBytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> raw{std::istreambuf_iterator<char>(file),
                                std::istreambuf_iterator<char>()};
    std::vector<std::byte> bytes(raw.size());
    std::ranges::transform(raw, bytes.begin(), [](char c) { return static_cast<std::byte>(c); });
    return bytes;
}

/// "Vad är klockan" at 16 kHz, scaled to a quiet laptop-microphone level.
std::vector<float> quietQuestion(const std::filesystem::path& path) {
    const jarvis::audio::DecodedAudio decoded = jarvis::audio::decodeMp3(readBytes(path));
    const std::vector<float> mono = jarvis::audio::toMono(decoded);
    const double step = static_cast<double>(decoded.sampleRate) / jarvis::audio::kSampleRate;
    std::vector<float> speech(static_cast<std::size_t>(static_cast<double>(mono.size()) / step));
    for (std::size_t i = 0; i < speech.size(); ++i) {
        const double position = static_cast<double>(i) * step;
        const auto left = static_cast<std::size_t>(position);
        const std::size_t right = std::min(left + 1, mono.size() - 1);
        const auto fraction = static_cast<float>(position - static_cast<double>(left));
        speech.at(i) = (mono.at(left) * (1.0F - fraction)) + (mono.at(right) * fraction);
    }
    double energy = 0.0;
    std::size_t voiced = 0;
    for (const float sample : speech) {
        if (std::abs(sample) > 1e-3F) {
            energy += static_cast<double>(sample) * sample;
            ++voiced;
        }
    }
    const auto gain = static_cast<float>(0.035 / std::sqrt(energy / static_cast<double>(voiced)));
    std::ranges::transform(speech, speech.begin(), [gain](float s) { return s * gain; });
    return speech;
}

}  // namespace

TEST_CASE("quiet follow-up questions all start speech", "[audio][vad][model]") {
    // Regression: frames were scored without the 64-sample context, so a short
    // quiet question flickered around the start threshold and often never
    // reached SpeechStarted. The user heard the earcon and then nothing.
    requireModel();
    const std::filesystem::path fixture =
        std::filesystem::path{JARVIS_FIXTURES_DIR} / "edge_tts_sv_question.mp3";
    if (!std::filesystem::exists(fixture)) {
        SKIP("fixture " << fixture << " is missing");
    }
    const std::vector<float> question = quietQuestion(fixture);

    constexpr int kRepeats = 12;
    std::vector<float> stream;
    std::uint32_t seed = 7;
    const auto next = [&seed] {
        seed = (seed * 1664525U) + 1013904223U;
        return static_cast<float>(seed >> 8U) / 16777216.0F;
    };
    for (int i = 0; i < kRepeats; ++i) {
        const auto gap = static_cast<std::size_t>((1.0F + (2.0F * next())) * 16000.0F);
        stream.insert(stream.end(), gap, 0.0F);
        stream.insert(stream.end(), question.begin(), question.end());
    }
    stream.insert(stream.end(), 16000, 0.0F);
    for (float& sample : stream) {
        sample += (next() - 0.5F) * 3e-4F;  // quiet-room noise floor
    }

    SileroVad vad{modelPath()};
    jarvis::audio::SpeechSegmenter segmenter{};
    int starts = 0;
    for (std::size_t offset = 0; offset + kVadFrameSamples <= stream.size();
         offset += kVadFrameSamples) {
        const std::span<const float, kVadFrameSamples> frame{stream.data() + offset,
                                                             kVadFrameSamples};
        if (segmenter.update(vad.speechProbability(frame)) ==
            jarvis::audio::SpeechSegmenter::Transition::SpeechStarted) {
            ++starts;
        }
    }

    REQUIRE(starts == kRepeats);
}
