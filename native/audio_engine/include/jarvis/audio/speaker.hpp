/// @file speaker.hpp
/// @brief Default playback device via miniaudio.

#pragma once

#include "jarvis/audio/audio_sink.hpp"

#include <memory>

namespace jarvis::audio {

/// Plays mono float32 samples on the system's default output device.
///
/// The device runs continuously once started and renders silence when the
/// callback has nothing queued, so starting speech never pays the cost of
/// opening a device.
class MiniaudioSpeaker final : public AudioSink {
public:
    explicit MiniaudioSpeaker(int sampleRate);
    ~MiniaudioSpeaker() override;

    MiniaudioSpeaker(const MiniaudioSpeaker&) = delete;
    MiniaudioSpeaker& operator=(const MiniaudioSpeaker&) = delete;
    MiniaudioSpeaker(MiniaudioSpeaker&&) = delete;
    MiniaudioSpeaker& operator=(MiniaudioSpeaker&&) = delete;

    void start(RenderCallback render) override;
    void stop() override;
    [[nodiscard]] int sampleRate() const noexcept override { return sampleRate_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int sampleRate_;
};

}  // namespace jarvis::audio
