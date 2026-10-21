/// @file events.hpp
/// @brief Events the audio engine reports to the Python side.

#pragma once

#include <cstdint>

namespace jarvis::audio {

enum class AudioEventType : std::uint8_t {
    /// The wake phrase was heard. `value` is the detector score.
    WakeWordDetected,
    /// Speech began after a period of silence. `value` is the VAD probability.
    SpeechStarted,
    /// Speech ended after enough silence. `value` is the VAD probability.
    SpeechEnded,
    /// The capture ring buffer overflowed. `value` is the number of samples lost.
    SamplesDropped,
};

/// One event, timestamped in samples since the engine started so latency can
/// be measured without a wall clock.
struct AudioEvent {
    AudioEventType type;
    float value;
    std::uint64_t samplePosition;
};

}  // namespace jarvis::audio
