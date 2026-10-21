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
class MiniaudioMicrophone final : public AudioSource {
public:
    explicit MiniaudioMicrophone(int sampleRate = kSampleRate);
    ~MiniaudioMicrophone() override;

    MiniaudioMicrophone(const MiniaudioMicrophone&) = delete;
    MiniaudioMicrophone& operator=(const MiniaudioMicrophone&) = delete;
    MiniaudioMicrophone(MiniaudioMicrophone&&) = delete;
    MiniaudioMicrophone& operator=(MiniaudioMicrophone&&) = delete;

    void start(FrameCallback onFrames) override;
    void stop() override;

private:
    struct Impl;  // keeps miniaudio out of the public header
    std::unique_ptr<Impl> impl_;
    int sampleRate_;
};

}  // namespace jarvis::audio
