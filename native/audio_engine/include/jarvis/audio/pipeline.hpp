/// @file pipeline.hpp
/// @brief Synchronous VAD + wake-word processing over a sample stream.

#pragma once

#include "jarvis/audio/audio_format.hpp"
#include "jarvis/audio/events.hpp"
#include "jarvis/audio/vad.hpp"
#include "jarvis/audio/wake_word.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace jarvis::audio {

struct PipelineConfig {
    SegmenterConfig segmenter;
    /// Wake-word score at or above which WakeWordDetected is reported.
    float wakeWordThreshold = 0.5F;
    /// Ignore further wake-word hits for this long after one, so a single
    /// "hey Jarvis" cannot trigger twice.
    int wakeWordCooldownMs = 2000;
    /// Audio kept from just before listening starts, so the first syllable
    /// of an utterance is not lost to detection latency.
    int preRollMs = 300;
};

/// The engine's signal-processing core, with no threads of its own.
///
/// Keeping this class synchronous means every behaviour (wake-word cooldown,
/// pre-roll, listening hand-over) can be unit-tested deterministically with
/// fake detectors. AudioEngine wraps it with the device thread, the ring
/// buffer and a worker thread.
class AudioPipeline {
public:
    using EventSink = std::function<void(const AudioEvent&)>;

    /// @param wakeWord May be null to run without wake-word detection
    ///        (hotkey-only mode).
    AudioPipeline(std::unique_ptr<VoiceActivityDetector> vad,
                  std::unique_ptr<WakeWordDetector> wakeWord, PipelineConfig config,
                  EventSink sink);

    /// Feed samples of any length; frames are assembled internally.
    void process(std::span<const float> samples);

    /// While listening, incoming audio is retained for takeCaptured() and
    /// wake-word detection is paused. Switching listening on seeds the
    /// capture with the pre-roll buffer.
    void setListening(bool listening);
    [[nodiscard]] bool listening() const noexcept { return listening_; }

    /// Move out the audio captured since listening started (or since the
    /// previous take).
    [[nodiscard]] std::vector<float> takeCaptured();

    /// RMS of the most recent VAD frame, for the overlay waveform.
    [[nodiscard]] float inputLevel() const noexcept { return inputLevel_; }

    [[nodiscard]] std::uint64_t samplesProcessed() const noexcept { return samplesProcessed_; }

    /// Drop all buffered audio and detector state.
    void reset();

private:
    void processVadFrame(std::span<const float, kVadFrameSamples> frame);
    void processWakeWordChunk(std::span<const float, kWakeWordChunkSamples> chunk);
    void retain(std::span<const float> samples);
    void emit(AudioEventType type, float value);

    std::unique_ptr<VoiceActivityDetector> vad_;
    std::unique_ptr<WakeWordDetector> wakeWord_;
    PipelineConfig config_;
    EventSink sink_;
    SpeechSegmenter segmenter_;
    std::vector<float> vadAccumulator_;
    std::vector<float> wakeWordAccumulator_;
    std::deque<float> preRoll_;
    std::size_t preRollCapacity_;
    std::vector<float> captured_;
    std::size_t cooldownRemaining_{0};
    bool listening_{false};
    float inputLevel_{0.0F};
    std::uint64_t samplesProcessed_{0};
};

}  // namespace jarvis::audio
