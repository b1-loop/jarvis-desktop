/// @file audio_source.hpp
/// @brief Interface for anything that produces capture frames.

#pragma once

#include <cstdint>
#include <functional>
#include <span>

namespace jarvis::audio {

/// A change in the capture device that frames alone cannot express.
enum class DeviceNotification : std::uint8_t {
    /// The device stopped without being asked to: unplugged, invalidated by
    /// the OS, or a WASAPI read that timed out (a long system stall or a
    /// resume from sleep). No more frames arrive until it is restarted.
    Stopped,
    /// The stream moved to another endpoint, usually because the default
    /// input device changed (a headset was plugged in).
    Rerouted,
    /// The OS took the device away for a while (an audio-session interruption).
    InterruptionBegan,
    /// The interruption ended; the device may deliver frames again.
    InterruptionEnded,
};

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
    using DeviceCallback = std::function<void(DeviceNotification)>;

    AudioSource() = default;
    AudioSource(const AudioSource&) = delete;
    AudioSource& operator=(const AudioSource&) = delete;
    AudioSource(AudioSource&&) = delete;
    AudioSource& operator=(AudioSource&&) = delete;
    virtual ~AudioSource() = default;

    /// Begin delivering frames to @p onFrames. Throws AudioDeviceError on failure.
    ///
    /// @param onDevice Told about device changes the source did not cause
    ///        itself; a stop() requested by the owner is never reported.
    ///        Called on a device thread, so it must not call start() or stop().
    virtual void start(FrameCallback onFrames, DeviceCallback onDevice) = 0;

    /// Stop delivering frames. Safe to call when not started.
    virtual void stop() = 0;
};

}  // namespace jarvis::audio
