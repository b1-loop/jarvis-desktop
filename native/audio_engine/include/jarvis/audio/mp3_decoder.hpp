/// @file mp3_decoder.hpp
/// @brief Decoding a complete MP3 buffer to float32 PCM.
///
/// edge-tts delivers speech as MP3. Decoding it in C++ (miniaudio's dr_mp3)
/// keeps the Python side free of audio codecs and their wheels, and takes
/// well under a millisecond per batch, so the Python TTS provider can decode
/// as the bytes arrive instead of after the whole sentence has downloaded.

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace jarvis::audio {

/// PCM produced by decodeMp3().
struct DecodedAudio {
    /// Interleaved float32 samples in [-1, 1], @c channels per frame.
    std::vector<float> samples;
    /// Output rate in Hz, as encoded in the stream.
    int sampleRate{0};
    /// Channel count of the stream (1 for edge-tts).
    int channels{0};
};

/// Decode every frame in @p data.
///
/// The buffer must hold whole MP3 frames; a trailing partial frame is
/// ignored by the decoder. The decoder keeps no state between calls, so a
/// stream split across calls must be primed with overlap by the caller (see
/// jarvis/tts/mp3_frames.py).
///
/// @throws DecodeError if @p data is empty or holds no decodable frame.
[[nodiscard]] DecodedAudio decodeMp3(std::span<const std::byte> data);

/// Average the channels of @p audio into one, frame by frame.
///
/// Mono input is returned as a copy, so callers need not special-case it.
[[nodiscard]] std::vector<float> toMono(const DecodedAudio& audio);

}  // namespace jarvis::audio
