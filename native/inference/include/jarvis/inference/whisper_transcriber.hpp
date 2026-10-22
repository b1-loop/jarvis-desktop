/// @file whisper_transcriber.hpp
/// @brief Offline speech-to-text over whisper.cpp.

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace jarvis::inference {

/// Sample rate whisper models are trained on; the audio engine captures at
/// the same rate, so no resampling happens between the two.
inline constexpr int kWhisperSampleRate = 16000;

/// Decoder search strategy. Greedy is several times faster; beam search is
/// more accurate on noisy audio.
enum class DecodingStrategy { Greedy, BeamSearch };

/// Per-call tuning for WhisperTranscriber::transcribe.
struct TranscribeOptions {
    /// ISO 639-1 code ("en", "bg", ...). std::nullopt lets the model detect
    /// the language; English-only models ignore the field.
    std::optional<std::string> language;
    /// Translate the speech to English instead of transcribing it.
    bool translate = false;
    /// Decoder threads; 0 picks whisper.cpp's default (min(4, cores)).
    int threads = 0;
    DecodingStrategy strategy = DecodingStrategy::Greedy;
    /// Beam width for DecodingStrategy::BeamSearch; ignored for Greedy.
    int beamSize = 5;
};

/// One timestamped piece of the transcript.
struct TranscriptSegment {
    std::int64_t startMs;
    std::int64_t endMs;
    std::string text;
};

/// Result of a transcription.
struct Transcript {
    /// All segments joined, whitespace-trimmed. Empty for silence.
    std::string text;
    /// Detected or requested language code.
    std::string language;
    std::vector<TranscriptSegment> segments;
};

/// Loads one ggml whisper model and transcribes 16 kHz mono float32 audio.
///
/// Thread-safe for sequential use: calls from different threads are
/// serialised by an internal mutex because a whisper context carries the
/// decoder state of the current call. Loading is the expensive part, so one
/// instance should live for the whole session.
class WhisperTranscriber {
public:
    /// Load @p model (a ggml-*.bin file). Throws ModelError if the file is
    /// missing or not a whisper model.
    explicit WhisperTranscriber(const std::filesystem::path& model);
    ~WhisperTranscriber();

    WhisperTranscriber(const WhisperTranscriber&) = delete;
    WhisperTranscriber& operator=(const WhisperTranscriber&) = delete;
    WhisperTranscriber(WhisperTranscriber&&) = delete;
    WhisperTranscriber& operator=(WhisperTranscriber&&) = delete;

    /// Transcribe @p pcm16k, mono float32 samples in [-1, 1] at 16 kHz.
    /// Throws ModelError if whisper.cpp fails. Empty audio yields an empty
    /// transcript.
    [[nodiscard]] Transcript transcribe(std::span<const float> pcm16k,
                                        const TranscribeOptions& options = {});

    /// Whether the model can handle languages other than English.
    [[nodiscard]] bool multilingual() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace jarvis::inference
