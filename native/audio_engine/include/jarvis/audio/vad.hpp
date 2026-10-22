/// @file vad.hpp
/// @brief Voice activity detection interface and speech segmentation.

#pragma once

#include "jarvis/audio/audio_format.hpp"

#include <span>

namespace jarvis::audio {

/// Estimates how likely one frame is to contain speech.
class VoiceActivityDetector {
public:
    VoiceActivityDetector() = default;
    VoiceActivityDetector(const VoiceActivityDetector&) = delete;
    VoiceActivityDetector& operator=(const VoiceActivityDetector&) = delete;
    VoiceActivityDetector(VoiceActivityDetector&&) = delete;
    VoiceActivityDetector& operator=(VoiceActivityDetector&&) = delete;
    virtual ~VoiceActivityDetector() = default;

    /// Probability in [0, 1] that @p frame contains speech. Stateful: frames
    /// must be fed in order.
    virtual float speechProbability(std::span<const float, kVadFrameSamples> frame) = 0;

    /// Forget any internal state, e.g. between utterances.
    virtual void reset() = 0;
};

/// Tuning for SpeechSegmenter. Defaults are conservative enough for a
/// laptop microphone in a quiet room; the Python side exposes them in config.
struct SegmenterConfig {
    /// Probability at or above which a frame counts as speech while silent.
    float startThreshold = 0.5F;
    /// Probability below which a frame counts as silence while speaking.
    /// Lower than startThreshold on purpose (hysteresis) so brief dips do
    /// not split an utterance.
    float endThreshold = 0.35F;
    /// Speech must persist this long before SpeechStarted is reported, which
    /// filters out clicks and single-frame false positives. Frames between
    /// the two thresholds pause the count; only a frame below endThreshold
    /// restarts it.
    int minSpeechMs = 200;
    /// Silence must persist this long before SpeechEnded is reported, which
    /// lets the user pause mid-sentence.
    int silenceEndMs = 700;
};

/// Turns a stream of per-frame speech probabilities into start/end decisions.
///
/// Pure logic with no clock: time is counted in frames, which makes the
/// behaviour deterministic and the tests exact.
class SpeechSegmenter {
public:
    enum class Transition { None, SpeechStarted, SpeechEnded };

    explicit SpeechSegmenter(SegmenterConfig config = {},
                             std::size_t frameSamples = kVadFrameSamples);

    /// Feed one frame's probability. @return The transition it caused, if any.
    Transition update(float probability) noexcept;

    [[nodiscard]] bool inSpeech() const noexcept { return inSpeech_; }

    void reset() noexcept;

private:
    [[nodiscard]] Transition updateWhileSilent(float probability) noexcept;
    [[nodiscard]] Transition updateWhileSpeaking(float probability) noexcept;

    SegmenterConfig config_;
    std::size_t minSpeechFrames_;
    std::size_t silenceEndFrames_;
    std::size_t speechFrames_{0};
    std::size_t silenceFrames_{0};
    bool inSpeech_{false};
};

}  // namespace jarvis::audio
