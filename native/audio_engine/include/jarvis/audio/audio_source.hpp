/// @file audio_source.hpp
/// @brief Interface for anything that produces capture frames.

#pragma once

#include <functional>
#include <span>

namespace jarvis::audio {

/// A source of 16 kHz mono float32 frames, usually a microphone.
///
/// Frames are delivered on the source's own thread (the device callback for
/// a real microphone). Implementations of the callback must be real-time
/// safe: no blocking, no allocation, no inference. The engine satisfies this
/// by only pushing into a SpscRingBuffer. Tests substitute a fake source
/// that feeds frames synchronously.
class AudioSource {
public:
    using FrameCallback = std::function<void(std::span<const float>)>;

    AudioSource() = default;
    AudioSource(const AudioSource&) = delete;
    AudioSource& operator=(const AudioSource&) = delete;
    AudioSource(AudioSource&&) = delete;
    AudioSource& operator=(AudioSource&&) = delete;
    virtual ~AudioSource() = default;

    /// Begin delivering frames to @p onFrames. Throws AudioDeviceError on failure.
    virtual void start(FrameCallback onFrames) = 0;

    /// Stop delivering frames. Safe to call when not started.
    virtual void stop() = 0;
};

}  // namespace jarvis::audio
