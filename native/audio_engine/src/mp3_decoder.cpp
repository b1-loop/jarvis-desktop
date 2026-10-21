#include "jarvis/audio/mp3_decoder.hpp"

#include "jarvis/common/errors.hpp"

#include <miniaudio.h>

#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

namespace jarvis::audio {

namespace {

/// Frames pulled per read call; small enough to keep the scratch buffer
/// modest, large enough that the per-call overhead is irrelevant.
constexpr ma_uint64 kReadChunkFrames = 2048;

/// RAII owner of an ma_decoder so every exit path uninitialises it.
class DecoderHandle {
public:
    DecoderHandle(std::span<const std::byte> data, const ma_decoder_config& config) {
        if (ma_decoder_init_memory(data.data(), data.size(), &config, &decoder_) != MA_SUCCESS) {
            throw DecodeError("data is not a decodable MP3 stream");
        }
    }
    ~DecoderHandle() { ma_decoder_uninit(&decoder_); }

    DecoderHandle(const DecoderHandle&) = delete;
    DecoderHandle& operator=(const DecoderHandle&) = delete;
    DecoderHandle(DecoderHandle&&) = delete;
    DecoderHandle& operator=(DecoderHandle&&) = delete;

    [[nodiscard]] ma_decoder* get() noexcept { return &decoder_; }

private:
    // Zero-initialised; ma_decoder_init_memory fills in the rest, including
    // the enums clang-tidy objects to having no zero value.
    ma_decoder decoder_{};  // NOLINT(bugprone-invalid-enum-default-initialization)
};

void readFormat(ma_decoder* decoder, DecodedAudio& audio) {
    ma_format format = ma_format_unknown;
    ma_uint32 channels = 0;
    ma_uint32 sampleRate = 0;
    if (ma_decoder_get_data_format(decoder, &format, &channels, &sampleRate, nullptr, 0) !=
            MA_SUCCESS ||
        channels == 0 || sampleRate == 0) {
        throw DecodeError("the MP3 stream reports no usable format");
    }
    audio.channels = static_cast<int>(channels);
    audio.sampleRate = static_cast<int>(sampleRate);
}

void readAllFrames(ma_decoder* decoder, DecodedAudio& audio) {
    const auto channels = static_cast<std::size_t>(audio.channels);
    std::vector<float> scratch(static_cast<std::size_t>(kReadChunkFrames) * channels);
    for (;;) {
        ma_uint64 framesRead = 0;
        const ma_result result =
            ma_decoder_read_pcm_frames(decoder, scratch.data(), kReadChunkFrames, &framesRead);
        const auto samplesRead = static_cast<std::size_t>(framesRead) * channels;
        audio.samples.insert(audio.samples.end(), scratch.begin(),
                             scratch.begin() + static_cast<std::ptrdiff_t>(samplesRead));
        if (result != MA_SUCCESS || framesRead < kReadChunkFrames) {
            return;  // MA_AT_END, or a short read that means the same
        }
    }
}

}  // namespace

DecodedAudio decodeMp3(std::span<const std::byte> data) {
    if (data.empty()) {
        throw DecodeError("cannot decode an empty buffer");
    }
    // Channels and rate left at 0 request the stream's own format: no
    // conversion, so decoded sample counts stay frame-exact for the
    // overlap arithmetic on the Python side.
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 0, 0);
    config.encodingFormat = ma_encoding_format_mp3;

    DecoderHandle handle{data, config};
    DecodedAudio audio;
    readFormat(handle.get(), audio);
    readAllFrames(handle.get(), audio);
    if (audio.samples.empty()) {
        throw DecodeError("no MP3 frames could be decoded");
    }
    return audio;
}

std::vector<float> toMono(const DecodedAudio& audio) {
    if (audio.channels <= 1) {
        return audio.samples;
    }
    const auto channels = static_cast<std::size_t>(audio.channels);
    const std::size_t frames = audio.samples.size() / channels;
    std::vector<float> mono;
    mono.reserve(frames);
    const std::span<const float> interleaved{audio.samples};
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const auto frameSamples = interleaved.subspan(frame * channels, channels);
        mono.push_back(std::reduce(frameSamples.begin(), frameSamples.end(), 0.0F) /
                       static_cast<float>(channels));
    }
    return mono;
}

}  // namespace jarvis::audio
