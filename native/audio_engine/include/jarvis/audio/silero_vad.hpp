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

    OnnxSession session_;
    std::int64_t sampleRate_;
    std::size_t inputIndex_;
    std::size_t stateIndex_;
    std::size_t sampleRateIndex_;
    std::size_t outputIndex_;
    std::size_t nextStateIndex_;
    std::vector<float> state_;
};

}  // namespace jarvis::audio
