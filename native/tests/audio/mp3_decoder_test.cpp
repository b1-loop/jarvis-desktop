#include "jarvis/audio/mp3_decoder.hpp"

#include "jarvis/common/errors.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string_view>
#include <type_traits>
#include <vector>

using jarvis::DecodeError;
using jarvis::audio::DecodedAudio;
using jarvis::audio::decodeMp3;
using jarvis::audio::toMono;

namespace {

// MPEG-2 Layer III, 48 kbps, 24 kHz, no padding, mono: the format edge-tts
// streams. A frame whose side info and main data are all zero decodes to
// silence, which makes a valid MP3 stream that needs no fixture file.
constexpr std::array<std::byte, 4> kSilentFrameHeader{std::byte{0xFF}, std::byte{0xF3},
                                                      std::byte{0x64}, std::byte{0xC0}};
constexpr std::size_t kFrameBytes = 144;
constexpr std::size_t kSamplesPerFrame = 576;
constexpr int kSampleRate = 24000;

std::vector<std::byte> silentFrames(std::size_t count) {
    std::vector<std::byte> stream;
    for (std::size_t i = 0; i < count; ++i) {
        stream.insert(stream.end(), kSilentFrameHeader.begin(), kSilentFrameHeader.end());
        stream.insert(stream.end(), kFrameBytes - kSilentFrameHeader.size(), std::byte{0});
    }
    return stream;
}

std::vector<std::byte> asBytes(std::string_view text) {
    std::vector<std::byte> bytes;
    std::ranges::transform(text, std::back_inserter(bytes),
                           [](char c) { return static_cast<std::byte>(c); });
    return bytes;
}

std::vector<std::byte> readFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<std::byte> bytes;
    std::ranges::transform(raw, std::back_inserter(bytes),
                           [](char c) { return static_cast<std::byte>(c); });
    return bytes;
}

float peak(const std::vector<float>& samples) {
    float largest = 0.0F;
    for (const float sample : samples) {
        largest = std::max(largest, std::fabs(sample));
    }
    return largest;
}

}  // namespace

TEST_CASE("DecodeError is part of the native error family", "[audio][mp3]") {
    STATIC_REQUIRE(std::is_base_of_v<jarvis::NativeError, DecodeError>);
}

TEST_CASE("an empty buffer is rejected", "[audio][mp3]") {
    REQUIRE_THROWS_AS(decodeMp3({}), DecodeError);
}

TEST_CASE("bytes that are not an MP3 stream are rejected", "[audio][mp3]") {
    REQUIRE_THROWS_AS(decodeMp3(asBytes("definitely not an mp3 file, just text")), DecodeError);
}

TEST_CASE("synthetic silent frames decode to silence in the header's format", "[audio][mp3]") {
    constexpr std::size_t kFrames = 8;

    const DecodedAudio audio = decodeMp3(silentFrames(kFrames));

    REQUIRE(audio.sampleRate == kSampleRate);
    REQUIRE(audio.channels == 1);
    REQUIRE(audio.samples.size() == kFrames * kSamplesPerFrame);
    REQUIRE(peak(audio.samples) < 1e-4F);
}

TEST_CASE("to_mono averages interleaved channels and copies mono", "[audio][mp3]") {
    const DecodedAudio stereo{
        .samples = {0.5F, -0.5F, 1.0F, 0.0F}, .sampleRate = 48000, .channels = 2};
    REQUIRE(toMono(stereo) == std::vector<float>{0.0F, 0.5F});

    const DecodedAudio mono{.samples = {0.25F, -0.25F}, .sampleRate = 24000, .channels = 1};
    REQUIRE(toMono(mono) == mono.samples);
}

TEST_CASE("a real edge-tts clip decodes to audible 24 kHz mono speech", "[audio][mp3]") {
    const std::filesystem::path path =
        std::filesystem::path(JARVIS_FIXTURES_DIR) / "edge_tts_hello.mp3";
    if (!std::filesystem::exists(path)) {
        SKIP("fixture " << path << " is missing");
    }

    const DecodedAudio audio = decodeMp3(readFile(path));

    REQUIRE(audio.sampleRate == kSampleRate);
    REQUIRE(audio.channels == 1);
    REQUIRE(audio.samples.size() > kSampleRate / 2);  // "Hello from Jarvis." is over 0.5 s
    REQUIRE(peak(audio.samples) > 0.05F);
    REQUIRE(peak(audio.samples) <= 1.0F);
}
