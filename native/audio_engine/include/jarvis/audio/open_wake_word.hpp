/// @file open_wake_word.hpp
/// @brief openWakeWord three-stage detector running on ONNX Runtime.

#pragma once

#include "jarvis/audio/audio_format.hpp"
#include "jarvis/audio/onnx_session.hpp"
#include "jarvis/audio/wake_word.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <span>
#include <vector>

namespace jarvis::audio {

/// WakeWordDetector backed by openWakeWord models.
///
/// openWakeWord splits detection into three small models so that the two
/// expensive stages (mel spectrogram, speech embedding) are shared by every
/// wake phrase and only the tiny classifier is phrase-specific. That is what
/// makes a custom wake word a config change rather than a training job.
///
/// Streaming works on 80 ms chunks: each chunk yields 8 new mel frames, the
/// last 76 mel frames yield one 96-dimensional embedding, and the last 16
/// embeddings yield one score.
class OpenWakeWord final : public WakeWordDetector {
public:
    struct ModelPaths {
        std::filesystem::path melspectrogram;
        std::filesystem::path embedding;
        std::filesystem::path classifier;
    };

    explicit OpenWakeWord(const ModelPaths& models);

    float score(std::span<const float, kWakeWordChunkSamples> chunk) override;
    void reset() override;

private:
    static constexpr std::size_t kMelBands = 32;
    static constexpr std::size_t kMelWindowFrames = 76;
    static constexpr std::size_t kEmbeddingSize = 96;
    static constexpr std::size_t kClassifierWindow = 16;
    /// Extra trailing samples from the previous chunk the mel model needs
    /// so that frame boundaries line up with the non-streaming reference.
    static constexpr std::size_t kContextSamples = 3 * kMelHopSamples;
    static constexpr float kMelScale = 0.1F;
    static constexpr float kMelOffset = 2.0F;

    using MelFrame = std::array<float, kMelBands>;
    using Embedding = std::array<float, kEmbeddingSize>;

    void appendMelFrames(std::span<const float, kWakeWordChunkSamples> chunk);
    void appendEmbedding();
    [[nodiscard]] float classify();

    OnnxSession melspectrogram_;
    OnnxSession embedding_;
    OnnxSession classifier_;
    std::vector<float> context_;
    std::deque<MelFrame> melFrames_;
    std::deque<Embedding> embeddings_;
};

}  // namespace jarvis::audio
