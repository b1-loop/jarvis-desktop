/// @file replay_source.hpp
/// @brief A virtual microphone that replays queued audio in real time.

#pragma once

#include "jarvis/audio/audio_format.hpp"
#include "jarvis/audio/audio_source.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <thread>

namespace jarvis::audio {

/// Audio waiting to be "heard" by a ReplaySource, shared with whoever feeds it.
///
/// The latency benchmark pushes a synthesised prompt here while the engine
/// runs; the source delivers it frame by frame and fills the gaps with
/// silence, like a quiet room between sentences. Thread-safe: push() runs on
/// a Python thread, take() on the source's thread.
class ReplayFeed {
public:
    /// Queue @p samples (16 kHz mono float32) after anything still pending.
    void push(std::span<const float> samples);

    /// Fill @p out with the next queued samples, then silence.
    /// @return How many of the samples were queued audio (not silence).
    std::size_t take(std::span<float> out);

    /// Queued samples not delivered yet.
    [[nodiscard]] std::size_t pending() const;

    /// Total samples delivered so far, audio and silence; the engine's
    /// sample clock (AudioEvent::samplePosition) counts the same samples.
    [[nodiscard]] std::uint64_t delivered() const noexcept { return delivered_.load(); }

private:
    mutable std::mutex mutex_;
    std::deque<float> queue_;
    std::atomic<std::uint64_t> delivered_{0};
};

/// AudioSource that delivers a ReplayFeed at the real-time rate.
///
/// Lets the full engine (ring buffer, Silero VAD, segmenter, wake word,
/// native barge-in) run on recorded speech without a microphone, so the
/// end-to-end benchmark measures the real detection path headlessly.
class ReplaySource final : public AudioSource {
public:
    /// @param feed Shared with the code that pushes audio.
    /// @param period Frame period; 10 ms matches a typical WASAPI device period.
    explicit ReplaySource(std::shared_ptr<ReplayFeed> feed,
                          std::chrono::milliseconds period = std::chrono::milliseconds(10));
    ~ReplaySource() override;

    ReplaySource(const ReplaySource&) = delete;
    ReplaySource& operator=(const ReplaySource&) = delete;
    ReplaySource(ReplaySource&&) = delete;
    ReplaySource& operator=(ReplaySource&&) = delete;

    void start(FrameCallback onFrames, DeviceCallback onDevice) override;
    void stop() override;

private:
    void run(const std::stop_token& stopToken, const FrameCallback& onFrames) const;

    std::shared_ptr<ReplayFeed> feed_;
    std::chrono::milliseconds period_;
    std::jthread thread_;
};

}  // namespace jarvis::audio

// Design notes:
// - Real-time pacing (not "as fast as possible") on purpose: the segmenter's
//   silence hangover and the barge-in echo guard are wall-clock behaviour,
//   so the benchmark must hear the audio at the speed a user would speak it.
// - The feed is separate from the source because the engine owns its source
//   through a unique_ptr; the Python side keeps the shared feed to push into.
