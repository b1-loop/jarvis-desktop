/// @file playback.hpp
/// @brief Streaming playback queue with instant stop for barge-in.

#pragma once

#include "jarvis/audio/audio_sink.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace jarvis::audio {

/// Plays a stream of PCM chunks as they arrive and can be silenced at once.
///
/// TTS produces audio sentence by sentence; chunks are enqueued as soon as
/// they exist so speech starts before the whole answer is synthesised.
/// stop() discards everything queued and the next device period is silence,
/// which is what makes barge-in feel immediate.
///
/// Thread model: enqueue(), stop() and markEndOfStream() may be called from
/// any thread (in practice the Python event loop). render() runs on the
/// device thread and uses try_lock, so a rare contended period renders
/// silence instead of ever blocking the audio callback.
class Playback {
public:
    explicit Playback(std::unique_ptr<AudioSink> sink);
    ~Playback();

    Playback(const Playback&) = delete;
    Playback& operator=(const Playback&) = delete;
    Playback(Playback&&) = delete;
    Playback& operator=(Playback&&) = delete;

    /// Queue samples for playback. Copies; the caller keeps ownership.
    void enqueue(std::span<const float> samples);

    /// Discard everything queued and fall silent immediately.
    void stop() noexcept;

    /// Signal that the current utterance has no more chunks coming, so
    /// isPlaying() turns false once the queue drains.
    void markEndOfStream() noexcept;

    /// True while samples are queued or more chunks are expected.
    [[nodiscard]] bool isPlaying() const noexcept;

    [[nodiscard]] std::size_t queuedSamples() const noexcept;

    /// RMS level of the most recent output period, for the overlay waveform.
    [[nodiscard]] float outputLevel() const noexcept;

    /// Number of stop() calls so far. The Python side compares it before and
    /// after an utterance to learn whether barge-in cut the speech short,
    /// since the engine stops playback natively without a round trip.
    [[nodiscard]] std::uint64_t stopCount() const noexcept;

    [[nodiscard]] int sampleRate() const noexcept { return sink_->sampleRate(); }

private:
    void render(std::span<float> out) noexcept;
    std::size_t fillFromQueue(std::span<float> out) noexcept;

    std::unique_ptr<AudioSink> sink_;
    mutable std::mutex mutex_;
    std::deque<std::vector<float>> chunks_;
    std::size_t offset_{0};
    std::atomic<std::size_t> queued_{0};
    std::atomic<bool> endOfStream_{true};
    std::atomic<float> outputLevel_{0.0F};
    std::atomic<std::uint64_t> stopCount_{0};
};

}  // namespace jarvis::audio
