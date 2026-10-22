#include "jarvis/audio/pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace jarvis::audio {

namespace {

/// Append @p samples to @p buffer and hand every complete N-sample frame to
/// @p onFrame, keeping the remainder for next time.
template <std::size_t N, typename OnFrame>
void assembleFrames(std::vector<float>& buffer, std::span<const float> samples,
                    const OnFrame& onFrame) {
    buffer.insert(buffer.end(), samples.begin(), samples.end());
    std::size_t consumed = 0;
    while (buffer.size() - consumed >= N) {
        onFrame(std::span<const float, N>{std::span<const float>{buffer}.subspan(consumed, N)});
        consumed += N;
    }
    buffer.erase(buffer.begin(), std::next(buffer.begin(), static_cast<std::ptrdiff_t>(consumed)));
}

float rootMeanSquare(std::span<const float> samples) noexcept {
    const float sumOfSquares = std::transform_reduce(samples.begin(), samples.end(), 0.0F,
                                                     std::plus<>{}, [](float s) { return s * s; });
    return std::sqrt(sumOfSquares / static_cast<float>(samples.size()));
}

}  // namespace

AudioPipeline::AudioPipeline(std::unique_ptr<VoiceActivityDetector> vad,
                             std::unique_ptr<WakeWordDetector> wakeWord, PipelineConfig config,
                             EventSink sink)
    : vad_(std::move(vad)),
      wakeWord_(std::move(wakeWord)),
      config_(config),
      sink_(std::move(sink)),
      segmenter_(config.segmenter),
      preRollCapacity_(millisecondsToSamples(config.preRollMs)) {}

void AudioPipeline::process(std::span<const float> samples) {
    retain(samples);
    assembleFrames<kVadFrameSamples>(vadAccumulator_, samples,
                                     [this](auto frame) { processVadFrame(frame); });
    if (wakeWord_ != nullptr && !listening_) {
        assembleFrames<kWakeWordChunkSamples>(wakeWordAccumulator_, samples,
                                              [this](auto chunk) { processWakeWordChunk(chunk); });
    }
    samplesProcessed_ += samples.size();
}

void AudioPipeline::setListening(bool listening) {
    if (listening == listening_) {
        return;
    }
    listening_ = listening;
    if (listening) {
        captured_.assign(preRoll_.begin(), preRoll_.end());
        preRoll_.clear();
        wakeWordAccumulator_.clear();
    }
}

std::vector<float> AudioPipeline::takeCaptured() {
    return std::exchange(captured_, {});
}

void AudioPipeline::reset() {
    // Explicit dereference: `vad_->reset()` would read as resetting the
    // smart pointer itself.
    (*vad_).reset();
    if (wakeWord_ != nullptr) {
        (*wakeWord_).reset();
    }
    segmenter_.reset();
    vadAccumulator_.clear();
    wakeWordAccumulator_.clear();
    preRoll_.clear();
    captured_.clear();
    cooldownRemaining_ = 0;
    listening_ = false;
    inputLevel_ = 0.0F;
}

void AudioPipeline::processVadFrame(std::span<const float, kVadFrameSamples> frame) {
    const float probability = vad_->speechProbability(frame);
    inputLevel_ = rootMeanSquare(frame);
    switch (segmenter_.update(probability)) {
        case SpeechSegmenter::Transition::SpeechStarted:
            emit(AudioEventType::SpeechStarted, probability);
            break;
        case SpeechSegmenter::Transition::SpeechEnded:
            emit(AudioEventType::SpeechEnded, probability);
            break;
        case SpeechSegmenter::Transition::None:
            break;
    }
}

void AudioPipeline::setWakeWordThreshold(float threshold) noexcept {
    // A threshold of 0 would fire on silence and 1 never; both are bugs.
    constexpr float kMinimum = 0.01F;
    constexpr float kMaximum = 0.99F;
    config_.wakeWordThreshold = std::clamp(threshold, kMinimum, kMaximum);
}

void AudioPipeline::processWakeWordChunk(std::span<const float, kWakeWordChunkSamples> chunk) {
    const float score = wakeWord_->score(chunk);
    if (cooldownRemaining_ > 0) {
        cooldownRemaining_ -= std::min(cooldownRemaining_, chunk.size());
        return;
    }
    if (score >= config_.wakeWordThreshold) {
        emit(AudioEventType::WakeWordDetected, score);
        cooldownRemaining_ = millisecondsToSamples(config_.wakeWordCooldownMs);
    }
}

void AudioPipeline::retain(std::span<const float> samples) {
    if (listening_) {
        captured_.insert(captured_.end(), samples.begin(), samples.end());
        return;
    }
    preRoll_.insert(preRoll_.end(), samples.begin(), samples.end());
    if (preRoll_.size() > preRollCapacity_) {
        const auto excess = static_cast<std::ptrdiff_t>(preRoll_.size() - preRollCapacity_);
        preRoll_.erase(preRoll_.begin(), std::next(preRoll_.begin(), excess));
    }
}

void AudioPipeline::emit(AudioEventType type, float value) {
    sink_(AudioEvent{.type = type, .value = value, .samplePosition = samplesProcessed_});
}

}  // namespace jarvis::audio
