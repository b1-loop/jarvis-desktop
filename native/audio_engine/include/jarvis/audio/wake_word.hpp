/// @file wake_word.hpp
/// @brief Wake-word detection interface.

#pragma once

#include "jarvis/audio/audio_format.hpp"

#include <span>

namespace jarvis::audio {

/// Scores the likelihood that the wake phrase just ended.
class WakeWordDetector {
public:
    WakeWordDetector() = default;
    WakeWordDetector(const WakeWordDetector&) = delete;
    WakeWordDetector& operator=(const WakeWordDetector&) = delete;
    WakeWordDetector(WakeWordDetector&&) = delete;
    WakeWordDetector& operator=(WakeWordDetector&&) = delete;
    virtual ~WakeWordDetector() = default;

    /// Consume one 80 ms chunk and return a score in [0, 1]. Stateful: chunks
    /// must be fed in order. Returns 0 until enough context has accumulated.
    virtual float score(std::span<const float, kWakeWordChunkSamples> chunk) = 0;

    /// Forget accumulated context.
    virtual void reset() = 0;
};

}  // namespace jarvis::audio
