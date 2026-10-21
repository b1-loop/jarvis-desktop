#include "jarvis/audio/open_wake_word.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace jarvis::audio {

OpenWakeWord::OpenWakeWord(const ModelPaths& models)
    : melspectrogram_(models.melspectrogram),
      embedding_(models.embedding),
      classifier_(models.classifier) {}

float OpenWakeWord::score(std::span<const float, kWakeWordChunkSamples> chunk) {
    appendMelFrames(chunk);
    if (melFrames_.size() < kMelWindowFrames) {
        return 0.0F;
    }
    appendEmbedding();
    if (embeddings_.size() < kClassifierWindow) {
        return 0.0F;
    }
    return classify();
}

void OpenWakeWord::reset() {
    context_.clear();
    melFrames_.clear();
    embeddings_.clear();
}

void OpenWakeWord::appendMelFrames(std::span<const float, kWakeWordChunkSamples> chunk) {
    std::vector<float> input(context_.begin(), context_.end());
    input.insert(input.end(), chunk.begin(), chunk.end());
    const std::array<OnnxSession::Input, 1> inputs{OnnxSession::FloatInput{
        .data = input, .shape = {1, static_cast<std::int64_t>(input.size())}}};

    const auto output = melspectrogram_.run(inputs).at(0);  // [1, 1, frames, 32]
    const std::size_t frames = output.data.size() / kMelBands;
    // Only the frames that correspond to the new chunk are fresh; the ones
    // computed over the context samples were already emitted last time.
    const std::size_t fresh = std::min(frames, chunk.size() / kMelHopSamples);
    const std::span<const float> values{output.data};
    for (std::size_t frame = frames - fresh; frame < frames; ++frame) {
        MelFrame mel{};
        const auto row = values.subspan(frame * kMelBands, kMelBands);
        std::ranges::transform(row, mel.begin(),
                               [](float value) { return (value * kMelScale) + kMelOffset; });
        melFrames_.push_back(mel);
    }
    while (melFrames_.size() > kMelWindowFrames) {
        melFrames_.pop_front();
    }
    const auto tail = chunk.last(kContextSamples);
    context_.assign(tail.begin(), tail.end());
}

void OpenWakeWord::appendEmbedding() {
    std::vector<float> window;
    window.reserve(kMelWindowFrames * kMelBands);
    for (const auto& frame : melFrames_) {
        window.insert(window.end(), frame.begin(), frame.end());
    }
    const std::array<OnnxSession::Input, 1> inputs{
        OnnxSession::FloatInput{.data = window,
                                .shape = {1, static_cast<std::int64_t>(kMelWindowFrames),
                                          static_cast<std::int64_t>(kMelBands), 1}}};

    const auto output = embedding_.run(inputs).at(0);  // [1, 1, 1, 96]
    Embedding embedding{};
    std::ranges::copy_n(output.data.begin(), static_cast<std::ptrdiff_t>(kEmbeddingSize),
                        embedding.begin());
    embeddings_.push_back(embedding);
    while (embeddings_.size() > kClassifierWindow) {
        embeddings_.pop_front();
    }
}

float OpenWakeWord::classify() {
    std::vector<float> window;
    window.reserve(kClassifierWindow * kEmbeddingSize);
    for (const auto& embedding : embeddings_) {
        window.insert(window.end(), embedding.begin(), embedding.end());
    }
    const std::array<OnnxSession::Input, 1> inputs{
        OnnxSession::FloatInput{.data = window,
                                .shape = {1, static_cast<std::int64_t>(kClassifierWindow),
                                          static_cast<std::int64_t>(kEmbeddingSize)}}};

    const auto output = classifier_.run(inputs).at(0);  // [1, 1]
    return std::clamp(output.data.at(0), 0.0F, 1.0F);
}

}  // namespace jarvis::audio
