/// @file microphone.hpp
/// @brief Default capture device via miniaudio.

#pragma once

#include "jarvis/audio/audio_format.hpp"
#include "jarvis/audio/audio_source.hpp"

#include <memory>

namespace jarvis::audio {

/// Captures 16 kHz mono float32 from the system's default input device.
///
/// miniaudio handles the backend (WASAPI on Windows, CoreAudio on macOS,
/// ALSA/PulseAudio on Linux) and resamples to the requested rate, which is
/// what keeps this class free of platform-specific code.
///
/// The device is opened with no device id, so every start() opens whatever
/// the default input is at that moment, and while running miniaudio's WASAPI
/// backend follows a change of the default by itself (reported as
/// DeviceNotification::Rerouted). A device that stops on its own (unplugged,
/// invalidated, a five-second WASAPI read timeout after a system stall) is
/// reported as DeviceNotification::Stopped; restarting it is the owner's job,
/// because miniaudio forbids reopening a device from its own callbacks.
class MiniaudioMicrophone final : public AudioSource {
public:
    explicit MiniaudioMicrophone(int sampleRate = kSampleRate);
    ~MiniaudioMicrophone() override;

    MiniaudioMicrophone(const MiniaudioMicrophone&) = delete;
    MiniaudioMicrophone& operator=(const MiniaudioMicrophone&) = delete;
    MiniaudioMicrophone(MiniaudioMicrophone&&) = delete;
    MiniaudioMicrophone& operator=(MiniaudioMicrophone&&) = delete;

    void start(FrameCallback onFrames, DeviceCallback onDevice) override;
    void stop() override;

private:
    struct Impl;  // keeps miniaudio out of the public header
    std::unique_ptr<Impl> impl_;
    int sampleRate_;
};

}  // namespace jarvis::audio
