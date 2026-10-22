#include "jarvis/inference/whisper_transcriber.hpp"

#include "jarvis/common/errors.hpp"
#include "jarvis/inference/backend.hpp"

#include <whisper.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace jarvis::inference {

namespace {

/// whisper.cpp reports timestamps in units of 10 ms.
constexpr std::int64_t kTimestampUnitMs = 10;

struct ContextDeleter {
    void operator()(whisper_context* context) const noexcept { whisper_free(context); }
};
using ContextPtr = std::unique_ptr<whisper_context, ContextDeleter>;

ContextPtr loadContext(const std::filesystem::path& model) {
    if (!std::filesystem::exists(model)) {
        throw ModelError("model file not found: " + model.string());
    }
    whisper_context_params params = whisper_context_default_params();
    params.use_gpu = false;  // CPU-only build; see docs/adr/0008
    const std::string path = model.string();
    ContextPtr context{whisper_init_from_file_with_params(path.c_str(), params)};
    if (context == nullptr) {
        throw ModelError("failed to load whisper model " + path);
    }
    return context;
}

whisper_sampling_strategy toWhisper(DecodingStrategy strategy) noexcept {
    return strategy == DecodingStrategy::BeamSearch ? WHISPER_SAMPLING_BEAM_SEARCH
                                                    : WHISPER_SAMPLING_GREEDY;
}

/// Build whisper_full_params. The returned struct borrows @p language, which
/// must outlive the whisper_full call.
whisper_full_params makeParams(const TranscribeOptions& options,
                               const std::optional<std::string>& language) {
    whisper_full_params params = whisper_full_default_params(toWhisper(options.strategy));
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.translate = options.translate;
    params.no_context = true;  // every call is an independent utterance
    params.suppress_blank = true;
    if (options.threads > 0) {
        params.n_threads = options.threads;
    }
    if (options.strategy == DecodingStrategy::BeamSearch) {
        params.beam_search.beam_size = std::max(options.beamSize, 1);
    }
    // "auto" makes whisper detect the language and then transcribe; the
    // separate detect_language flag would stop after detection.
    params.language = language.has_value() ? language->c_str() : "auto";
    params.detect_language = false;
    return params;
}

std::string trimmed(std::string text) {
    const auto notSpace = [](unsigned char c) { return std::isspace(c) == 0; };
    text.erase(text.begin(), std::ranges::find_if(text, notSpace));
    text.erase(std::ranges::find_if(text.rbegin(), text.rend(), notSpace).base(), text.end());
    return text;
}

std::vector<TranscriptSegment> collectSegments(whisper_context* context) {
    const int count = whisper_full_n_segments(context);
    std::vector<TranscriptSegment> segments;
    segments.reserve(static_cast<std::size_t>(std::max(count, 0)));
    for (int i = 0; i < count; ++i) {
        std::string text = trimmed(whisper_full_get_segment_text(context, i));
        if (text.empty()) {
            continue;
        }
        segments.push_back(
            TranscriptSegment{.startMs = whisper_full_get_segment_t0(context, i) * kTimestampUnitMs,
                              .endMs = whisper_full_get_segment_t1(context, i) * kTimestampUnitMs,
                              .text = std::move(text)});
    }
    return segments;
}

std::string joinSegments(const std::vector<TranscriptSegment>& segments) {
    std::string text;
    for (const auto& segment : segments) {
        if (!text.empty()) {
            text += ' ';
        }
        text += segment.text;
    }
    return text;
}

std::string detectedLanguage(whisper_context* context) {
    const char* code = whisper_lang_str(whisper_full_lang_id(context));
    return code == nullptr ? std::string{} : std::string{code};
}

}  // namespace

struct WhisperTranscriber::Impl {
    std::mutex mutex;  // whisper_context holds per-call decoder state
    ContextPtr context;
    std::string modelName;
};

WhisperTranscriber::WhisperTranscriber(const std::filesystem::path& model)
    : impl_(std::make_unique<Impl>()) {
    ensureBackend();
    impl_->context = loadContext(model);
    impl_->modelName = model.filename().string();
}

WhisperTranscriber::~WhisperTranscriber() = default;

Transcript WhisperTranscriber::transcribe(std::span<const float> pcm16k,
                                          const TranscribeOptions& options) {
    if (pcm16k.empty()) {
        return Transcript{.text = {}, .language = options.language.value_or(""), .segments = {}};
    }
    const std::scoped_lock lock(impl_->mutex);
    // Copied so the lifetime of the C string handed to whisper is explicit.
    const std::optional<std::string> language = options.language;
    const whisper_full_params params = makeParams(options, language);
    whisper_context* context = impl_->context.get();
    const int status =
        whisper_full(context, params, pcm16k.data(), static_cast<int>(pcm16k.size()));
    if (status != 0) {
        throw ModelError("whisper_full failed with code " + std::to_string(status) + " for " +
                         impl_->modelName);
    }
    Transcript transcript;
    transcript.segments = collectSegments(context);
    transcript.text = joinSegments(transcript.segments);
    transcript.language = detectedLanguage(context);
    return transcript;
}

bool WhisperTranscriber::multilingual() const noexcept {
    return whisper_is_multilingual(impl_->context.get()) != 0;
}

}  // namespace jarvis::inference

// Design notes:
// - whisper_full() rather than the lower-level encode/decode calls: the
//   assistant transcribes whole utterances delimited by the audio engine's
//   VAD, so there is no need for whisper.cpp's streaming heuristics.
// - no_context=true on every call: carrying the previous utterance as a
//   decoder prompt helps long dictations but makes a voice assistant repeat
//   hallucinated text across turns.
// - Timestamps are converted to milliseconds at the boundary so no caller
//   needs to know whisper's 10 ms unit.
