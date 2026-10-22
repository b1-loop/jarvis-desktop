#include "jarvis/audio/silero_vad.hpp"

#include "jarvis/common/errors.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <span>
#include <utility>

namespace jarvis::audio {

SileroVad::SileroVad(const std::filesystem::path& model, int sampleRate)
    : session_(model),
      sampleRate_(sampleRate),
      inputIndex_(session_.inputIndex("input")),
      stateIndex_(session_.inputIndex("state")),
      sampleRateIndex_(session_.inputIndex("sr")),
      outputIndex_(session_.outputIndex("output")),
      nextStateIndex_(session_.outputIndex("stateN")),
      state_(kStateLayers * kStateWidth, 0.0F),
      contextSamples_(sampleRate == kSampleRate ? kContextSamples16k : kContextSamples8k),
      input_(contextSamples_ + kVadFrameSamples, 0.0F) {
    if (session_.inputCount() != kInputCount) {
        throw ModelError("silero_vad.onnx: expected the v5 model with three inputs");
    }
}

float SileroVad::speechProbability(std::span<const float, kVadFrameSamples> frame) {
    const auto context = static_cast<std::ptrdiff_t>(contextSamples_);
    std::ranges::copy(frame, std::next(input_.begin(), context));

    std::array<OnnxSession::Input, kInputCount> inputs;
    inputs.at(inputIndex_) = OnnxSession::FloatInput{
        .data = input_, .shape = {1, static_cast<std::int64_t>(input_.size())}};
    inputs.at(stateIndex_) =
        OnnxSession::FloatInput{.data = state_,
                                .shape = {static_cast<std::int64_t>(kStateLayers), 1,
                                          static_cast<std::int64_t>(kStateWidth)}};
    inputs.at(sampleRateIndex_) = OnnxSession::Int64Input{
        .data = std::span<const std::int64_t>{&sampleRate_, 1}, .shape = {}};

    auto outputs = session_.run(inputs);
    state_ = std::move(outputs.at(nextStateIndex_).data);
    // The tail of this frame is the context of the next one.
    std::copy(std::prev(input_.end(), context), input_.end(), input_.begin());
    return std::clamp(outputs.at(outputIndex_).data.at(0), 0.0F, 1.0F);
}

void SileroVad::reset() {
    std::ranges::fill(state_, 0.0F);
    std::ranges::fill(input_, 0.0F);
}

}  // namespace jarvis::audio
