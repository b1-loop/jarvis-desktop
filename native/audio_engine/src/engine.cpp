#include "jarvis/audio/engine.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace jarvis::audio {

namespace {

/// Samples the worker pops per iteration; two VAD frames keeps latency low
/// without waking up for every device period.
constexpr std::size_t kWorkerChunkSamples = 2 * kVadFrameSamples;

AudioEventType toEventType(DeviceNotification notification) noexcept {
    switch (notification) {
        case DeviceNotification::Stopped:
            return AudioEventType::DeviceStopped;
        case DeviceNotification::Rerouted:
            return AudioEventType::DeviceRerouted;
        case DeviceNotification::InterruptionBegan:
            return AudioEventType::DeviceInterrupted;
        case DeviceNotification::InterruptionEnded:
            return AudioEventType::DeviceResumed;
    }
    // Unreachable for a valid enumerator; "stopped" is the safe reading of
    // garbage because it makes the watchdog check that frames still arrive.
    return AudioEventType::DeviceStopped;
}

}  // namespace

AudioEngine::AudioEngine(std::unique_ptr<AudioSource> source,
                         std::unique_ptr<VoiceActivityDetector> vad,
                         std::unique_ptr<WakeWordDetector> wakeWord, EngineConfig config)
    : source_(std::move(source)),
      config_(config),
      ring_(config.ringBufferSamples),
      pipeline_(std::move(vad), std::move(wakeWord), config.pipeline,
                [this](const AudioEvent& event) { onEvent(event); }) {}

AudioEngine::~AudioEngine() {
    stop();
}

void AudioEngine::start() {
    const std::scoped_lock lifecycle(lifecycleMutex_);
    if (running_.load()) {
        return;
    }
    worker_ = std::jthread([this](const std::stop_token& token) { workerLoop(token); });
    startSource();
    running_.store(true);
}

void AudioEngine::stop() {
    const std::scoped_lock lifecycle(lifecycleMutex_);
    if (!running_.exchange(false)) {
        return;
    }
    source_->stop();
    worker_.request_stop();
    if (worker_.joinable()) {
        worker_.join();
    }
}

bool AudioEngine::restartCapture() {
    const std::scoped_lock lifecycle(lifecycleMutex_);
    if (!running_.load()) {
        return false;
    }
    source_->stop();
    {
        const std::scoped_lock lock(mutex_);
        const bool wasListening = pipeline_.listening();
        pipeline_.reset();
        pipeline_.setListening(wasListening);
    }
    startSource();
    return true;
}

void AudioEngine::startSource() {
    source_->start([this](std::span<const float> frames) { onFrames(frames); },
                   [this](DeviceNotification notification) { onDeviceNotification(notification); });
}

void AudioEngine::onFrames(std::span<const float> frames) noexcept {
    ring_.push(frames);
    samplesReceived_.fetch_add(frames.size(), std::memory_order_relaxed);
}

void AudioEngine::onDeviceNotification(DeviceNotification notification) {
    const std::scoped_lock lock(mutex_);
    onEvent(AudioEvent{.type = toEventType(notification),
                       .value = 0.0F,
                       .samplePosition = pipeline_.samplesProcessed()});
}

void AudioEngine::setListening(bool listening) {
    const std::scoped_lock lock(mutex_);
    pipeline_.setListening(listening);
}

bool AudioEngine::listening() const {
    const std::scoped_lock lock(mutex_);
    return pipeline_.listening();
}

std::vector<float> AudioEngine::takeCaptured() {
    const std::scoped_lock lock(mutex_);
    return pipeline_.takeCaptured();
}

std::vector<AudioEvent> AudioEngine::pollEvents() {
    const std::scoped_lock lock(mutex_);
    return std::exchange(events_, {});
}

void AudioEngine::setBargeInTarget(std::shared_ptr<Playback> playback) {
    const std::scoped_lock lock(mutex_);
    bargeIn_ = std::move(playback);
}

void AudioEngine::setWakeWordThreshold(float threshold) {
    const std::scoped_lock lock(mutex_);
    pipeline_.setWakeWordThreshold(threshold);
}

float AudioEngine::wakeWordThreshold() const {
    const std::scoped_lock lock(mutex_);
    return pipeline_.wakeWordThreshold();
}

float AudioEngine::inputLevel() const {
    const std::scoped_lock lock(mutex_);
    return pipeline_.inputLevel();
}

std::uint64_t AudioEngine::samplesProcessed() const {
    const std::scoped_lock lock(mutex_);
    return pipeline_.samplesProcessed();
}

void AudioEngine::onEvent(const AudioEvent& event) {
    events_.push_back(event);
    if (event.type == AudioEventType::SpeechStarted && bargeIn_ != nullptr) {
        bargeIn_->stop();
    }
}

void AudioEngine::workerLoop(const std::stop_token& stopToken) {
    std::vector<float> scratch(kWorkerChunkSamples);
    while (!stopToken.stop_requested()) {
        const std::size_t count = ring_.pop(scratch);
        if (count == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.workerPollMs));
            continue;
        }
        const std::scoped_lock lock(mutex_);
        pipeline_.process(std::span<const float>{scratch}.first(count));
        reportDroppedSamples();
    }
}

void AudioEngine::reportDroppedSamples() {
    const std::size_t total = ring_.droppedSamples();
    if (total > reportedDrops_) {
        onEvent(AudioEvent{.type = AudioEventType::SamplesDropped,
                           .value = static_cast<float>(total - reportedDrops_),
                           .samplePosition = pipeline_.samplesProcessed()});
        reportedDrops_ = total;
    }
}

}  // namespace jarvis::audio
