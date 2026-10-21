/// @file audio_format.hpp
/// @brief The one audio format the engine works in internally.
///
/// Both Silero VAD and openWakeWord are trained on 16 kHz mono, so the
/// capture device is opened at that rate and every downstream consumer
/// (STT included) receives float32 samples in [-1, 1] without resampling.

#pragma once

#include <cstddef>

namespace jarvis::audio {

/// Capture sample rate in Hz.
inline constexpr int kSampleRate = 16000;

/// Samples per VAD frame: 32 ms, the window Silero VAD expects at 16 kHz.
inline constexpr std::size_t kVadFrameSamples = 512;

/// Samples per wake-word chunk: 80 ms, the step openWakeWord expects.
inline constexpr std::size_t kWakeWordChunkSamples = 1280;

/// Hop between consecutive mel-spectrogram frames: 10 ms.
inline constexpr std::size_t kMelHopSamples = 160;

/// Convert a duration in milliseconds to a sample count at kSampleRate.
constexpr std::size_t millisecondsToSamples(int milliseconds) noexcept {
    return static_cast<std::size_t>(milliseconds) * static_cast<std::size_t>(kSampleRate) / 1000U;
}

}  // namespace jarvis::audio
