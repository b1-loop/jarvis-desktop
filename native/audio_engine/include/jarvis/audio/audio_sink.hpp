/// @file audio_sink.hpp
/// @brief Interface for anything that consumes playback samples.

#pragma once

#include <functional>
#include <span>

namespace jarvis::audio {

/// A sink that repeatedly asks for mono float32 samples to play.
///
/// The sink calls the render callback from its own thread (the device
/// callback for a real speaker) with a buffer to fill. The callback must be
/// real-time safe and must fill the whole buffer, using silence when nothing
/// is queued. Tests substitute a fake sink and invoke the callback directly.
class AudioSink {
public:
    using RenderCallback = std::function<void(std::span<float>)>;

    AudioSink() = default;
    AudioSink(const AudioSink&) = delete;
    AudioSink& operator=(const AudioSink&) = delete;
    AudioSink(AudioSink&&) = delete;
    AudioSink& operator=(AudioSink&&) = delete;
    virtual ~AudioSink() = default;

    /// Open the device and start pulling samples through @p render.
    virtual void start(RenderCallback render) = 0;

    /// Stop pulling samples. Safe to call when not started.
    virtual void stop() = 0;

    /// Sample rate the sink plays at, in Hz.
    [[nodiscard]] virtual int sampleRate() const noexcept = 0;
};

}  // namespace jarvis::audio
