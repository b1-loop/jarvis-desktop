#include "jarvis/audio/vad.hpp"

#include <algorithm>
#include <cstddef>

namespace jarvis::audio {

namespace {

/// Frames needed to cover @p milliseconds, rounded up so a threshold is
/// never reached early.
std::size_t framesFor(int milliseconds, std::size_t frameSamples) noexcept {
    const std::size_t samples = millisecondsToSamples(milliseconds);
    return std::max<std::size_t>(1, (samples + frameSamples - 1) / frameSamples);
}

}  // namespace

SpeechSegmenter::SpeechSegmenter(SegmenterConfig config, std::size_t frameSamples)
    : config_(config),
      minSpeechFrames_(framesFor(config.minSpeechMs, frameSamples)),
      silenceEndFrames_(framesFor(config.silenceEndMs, frameSamples)) {}

SpeechSegmenter::Transition SpeechSegmenter::update(float probability) noexcept {
    return inSpeech_ ? updateWhileSpeaking(probability) : updateWhileSilent(probability);
}

SpeechSegmenter::Transition SpeechSegmenter::updateWhileSilent(float probability) noexcept {
    if (probability < config_.startThreshold) {
        speechFrames_ = 0;
        return Transition::None;
    }
    ++speechFrames_;
    if (speechFrames_ < minSpeechFrames_) {
        return Transition::None;
    }
    inSpeech_ = true;
    speechFrames_ = 0;
    silenceFrames_ = 0;
    return Transition::SpeechStarted;
}

SpeechSegmenter::Transition SpeechSegmenter::updateWhileSpeaking(float probability) noexcept {
    if (probability >= config_.endThreshold) {
        silenceFrames_ = 0;
        return Transition::None;
    }
    ++silenceFrames_;
    if (silenceFrames_ < silenceEndFrames_) {
        return Transition::None;
    }
    inSpeech_ = false;
    silenceFrames_ = 0;
    return Transition::SpeechEnded;
}

void SpeechSegmenter::reset() noexcept {
    speechFrames_ = 0;
    silenceFrames_ = 0;
    inSpeech_ = false;
}

}  // namespace jarvis::audio
