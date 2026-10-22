/// @file silero_vad.hpp
/// @brief Silero VAD (v5) running on ONNX Runtime.

#pragma once

#include "jarvis/audio/audio_format.hpp"
#include "jarvis/audio/onnx_session.hpp"
#include "jarvis/audio/vad.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace jarvis::audio {

/// VoiceActivityDetector backed by the Silero VAD v5 ONNX model.
///
/// Silero is small (about 2 MB), runs in well under a millisecond per frame
/// on a CPU and is markedly more robust to keyboard noise and music than
/// energy-based detectors, which is why it was chosen over WebRTC VAD.
///
/// The v5 model expects each frame to be preceded by the last samples of the
/// previous one (64 at 16 kHz, 32 at 8 kHz), as the reference implementation
/// does. Without that context the probability flickers on quiet or fast
/// speech, and the segmenter never sees enough consecutive speech frames.
class SileroVad final : public VoiceActivityDetector {
public:
    /// @param model Path to silero_vad.onnx (v5 layout: input, state, sr).
    /// @param sampleRate Must be 16000 or 8000; the engine uses 16000.
    explicit SileroVad(const std::filesystem::path& model, int sampleRate = kSampleRate);

    float speechProbability(std::span<const float, kVadFrameSamples> frame) override;
    void reset() override;

private:
    static constexpr std::size_t kStateLayers = 2;
    static constexpr std::size_t kStateWidth = 128;
    static constexpr std::size_t kInputCount = 3;
    static constexpr std::size_t kContextSamples16k = 64;
    static constexpr std::size_t kContextSamples8k = 32;

    OnnxSession session_;
    std::int64_t sampleRate_;
    std::size_t inputIndex_;
    std::size_t stateIndex_;
    std::size_t sampleRateIndex_;
    std::size_t outputIndex_;
    std::size_t nextStateIndex_;
    std::vector<float> state_;
    std::size_t contextSamples_;
    /// Context followed by the current frame: the tensor fed to the model.
    std::vector<float> input_;
};

}  // namespace jarvis::audio
