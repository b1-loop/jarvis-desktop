/// @file engine.hpp
/// @brief Threaded audio engine: capture, detection and barge-in.

#pragma once

#include "jarvis/audio/audio_format.hpp"
#include "jarvis/audio/audio_source.hpp"
#include "jarvis/audio/events.hpp"
#include "jarvis/audio/pipeline.hpp"
#include "jarvis/audio/playback.hpp"
#include "jarvis/audio/ring_buffer.hpp"
#include "jarvis/audio/vad.hpp"
#include "jarvis/audio/wake_word.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

namespace jarvis::audio {

struct EngineConfig {
    PipelineConfig pipeline;
    /// Capture ring buffer size; ten seconds absorbs any GC pause or
    /// model-loading hiccup on the consumer side without dropping audio.
    std::size_t ringBufferSamples = static_cast<std::size_t>(kSampleRate) * 10;
    /// How long the worker sleeps when the ring buffer is empty.
    int workerPollMs = 5;
};

/// Owns the capture device, the worker thread and the event queue.
///
/// Threads:
///  - device thread (inside AudioSource): pushes frames into the ring buffer;
///  - worker thread: pops frames and runs AudioPipeline;
///  - caller threads (Python): poll events, toggle listening, take audio.
///
/// The worker never calls out to Python; the Python side polls, so the GIL
/// is never needed on a native thread.
class AudioEngine {
public:
    AudioEngine(std::unique_ptr<AudioSource> source, std::unique_ptr<VoiceActivityDetector> vad,
                std::unique_ptr<WakeWordDetector> wakeWord, EngineConfig config = {});
    ~AudioEngine();

    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;
    AudioEngine(AudioEngine&&) = delete;
    AudioEngine& operator=(AudioEngine&&) = delete;

    /// Open the device and start processing. Throws AudioDeviceError.
    void start();

    /// Stop the device and the worker. Safe to call repeatedly.
    void stop();

    [[nodiscard]] bool running() const noexcept { return running_.load(); }

    /// See AudioPipeline::setListening.
    void setListening(bool listening);
    [[nodiscard]] bool listening() const;

    /// See AudioPipeline::takeCaptured.
    [[nodiscard]] std::vector<float> takeCaptured();

    /// Drain and return every event reported since the previous poll.
    [[nodiscard]] std::vector<AudioEvent> pollEvents();

    /// While set, SpeechStarted stops @p playback immediately (barge-in).
    /// Pass nullptr to disable.
    void setBargeInTarget(std::shared_ptr<Playback> playback);

    /// See AudioPipeline::setWakeWordThreshold; applies from the next chunk.
    void setWakeWordThreshold(float threshold);
    [[nodiscard]] float wakeWordThreshold() const;

    [[nodiscard]] float inputLevel() const;
    [[nodiscard]] std::size_t droppedSamples() const noexcept { return ring_.droppedSamples(); }
    [[nodiscard]] std::uint64_t samplesProcessed() const;

private:
    /// Called by the pipeline with mutex_ already held.
    void onEvent(const AudioEvent& event);
    void workerLoop(const std::stop_token& stopToken);
    void reportDroppedSamples();

    std::unique_ptr<AudioSource> source_;
    EngineConfig config_;
    SpscRingBuffer ring_;
    mutable std::mutex mutex_;  // guards pipeline_, events_, bargeIn_, reportedDrops_
    AudioPipeline pipeline_;
    std::vector<AudioEvent> events_;
    std::shared_ptr<Playback> bargeIn_;
    std::size_t reportedDrops_{0};
    std::jthread worker_;
    std::atomic<bool> running_{false};
};

}  // namespace jarvis::audio
