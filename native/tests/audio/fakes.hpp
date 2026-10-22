/// @file fakes.hpp
/// @brief Scriptable test doubles for the audio interfaces.

#pragma once

#include "jarvis/audio/audio_format.hpp"
#include "jarvis/audio/audio_sink.hpp"
#include "jarvis/audio/audio_source.hpp"
#include "jarvis/audio/vad.hpp"
#include "jarvis/audio/wake_word.hpp"
#include "jarvis/common/errors.hpp"

#include <cstddef>
#include <deque>
#include <span>
#include <vector>

namespace jarvis::audio::testing {

/// Returns scripted probabilities in order, then a default forever.
class FakeVad final : public VoiceActivityDetector {
public:
    explicit FakeVad(float defaultProbability = 0.0F) : default_(defaultProbability) {}

    void script(std::initializer_list<float> probabilities) {
        scripted_.insert(scripted_.end(), probabilities);
    }

    /// Script @p count frames of the same probability.
    void scriptRepeated(float probability, std::size_t count) {
        scripted_.insert(scripted_.end(), count, probability);
    }

    float speechProbability(std::span<const float, kVadFrameSamples> /*frame*/) override {
        ++framesSeen;
        if (scripted_.empty()) {
            return default_;
        }
        const float next = scripted_.front();
        scripted_.pop_front();
        return next;
    }

    void reset() override { ++resets; }

    std::size_t framesSeen{0};
    std::size_t resets{0};

private:
    float default_;
    std::deque<float> scripted_;
};

/// Returns scripted scores in order, then a default forever.
class FakeWakeWord final : public WakeWordDetector {
public:
    explicit FakeWakeWord(float defaultScore = 0.0F) : default_(defaultScore) {}

    void script(std::initializer_list<float> scores) { scripted_.insert(scripted_.end(), scores); }

    float score(std::span<const float, kWakeWordChunkSamples> /*chunk*/) override {
        ++chunksSeen;
        if (scripted_.empty()) {
            return default_;
        }
        const float next = scripted_.front();
        scripted_.pop_front();
        return next;
    }

    void reset() override { ++resets; }

    std::size_t chunksSeen{0};
    std::size_t resets{0};

private:
    float default_;
    std::deque<float> scripted_;
};

/// A source the test drives by hand: frames, device notifications and
/// failures to open are all triggered explicitly.
class FakeAudioSource final : public AudioSource {
public:
    void start(FrameCallback onFrames, DeviceCallback onDevice) override {
        if (failStarts > 0) {
            --failStarts;
            throw AudioDeviceError("fake: no capture device");
        }
        callback_ = std::move(onFrames);
        deviceCallback_ = std::move(onDevice);
        started = true;
        ++startCount;
    }

    void stop() override {
        callback_ = nullptr;
        deviceCallback_ = nullptr;
        stopped = true;
    }

    /// Deliver @p samples as if the device callback had fired.
    void feed(std::span<const float> samples) {
        if (callback_) {
            callback_(samples);
        }
    }

    /// Report @p notification as if the device had changed on its own.
    void notify(DeviceNotification notification) {
        if (deviceCallback_) {
            deviceCallback_(notification);
        }
    }

    bool started{false};
    bool stopped{false};
    std::size_t startCount{0};
    /// The next this many start() calls throw, like a device that is gone.
    std::size_t failStarts{0};

private:
    FrameCallback callback_;
    DeviceCallback deviceCallback_;
};

/// A sink whose render callback the test invokes directly.
class FakeAudioSink final : public AudioSink {
public:
    explicit FakeAudioSink(int sampleRate = kSampleRate) : sampleRate_(sampleRate) {}

    void start(RenderCallback render) override {
        render_ = std::move(render);
        started = true;
    }

    void stop() override {
        render_ = nullptr;
        stopped = true;
    }

    [[nodiscard]] int sampleRate() const noexcept override { return sampleRate_; }

    /// Ask the owner for @p count samples, as the device would.
    std::vector<float> render(std::size_t count) {
        std::vector<float> out(count, -1.0F);  // sentinel: must be overwritten
        if (render_) {
            render_(out);
        }
        return out;
    }

    bool started{false};
    bool stopped{false};

private:
    int sampleRate_;
    RenderCallback render_;
};

/// A buffer of @p count samples with the given value.
inline std::vector<float> samples(std::size_t count, float value = 0.0F) {
    return std::vector<float>(count, value);
}

}  // namespace jarvis::audio::testing
