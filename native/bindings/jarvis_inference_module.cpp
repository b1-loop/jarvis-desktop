/// @file jarvis_inference_module.cpp
/// @brief Python module @c jarvis_inference.
///
/// A thin layer: converts numpy arrays and message dicts to C++ types,
/// releases the GIL around model loading, transcription and token pulls,
/// and maps NativeError onto Python exceptions. Generation runs on a native
/// thread that never touches Python; GenerationStream is the pull-based
/// hand-off, mirroring the polling design of jarvis_audio.

#include "jarvis/common/errors.hpp"
#include "jarvis/inference/backend.hpp"
#include "jarvis/inference/chat_template.hpp"
#include "jarvis/inference/generation_stream.hpp"
#include "jarvis/inference/llama_generator.hpp"
#include "jarvis/inference/whisper_transcriber.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "module_info.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace jarvis::inference;  // NOLINT(google-build-using-namespace) - binding file only

namespace {

// NOLINTNEXTLINE(bugprone-signed-bitwise) - pybind11's documented flag idiom
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

std::span<const float> asSpan(const FloatArray& array) {
    if (array.ndim() != 1) {
        throw py::value_error("expected a one-dimensional float32 array of 16 kHz mono samples");
    }
    return {array.data(), static_cast<std::size_t>(array.shape(0))};
}

/// Accepts the ``{"role": ..., "content": ...}`` dicts the Python LLM layer
/// already produces, so no extra message class leaks into Python.
std::vector<ChatMessage> toMessages(const py::sequence& messages) {
    std::vector<ChatMessage> converted;
    converted.reserve(static_cast<std::size_t>(py::len(messages)));
    for (const auto& item : messages) {
        const auto message = item.cast<std::map<std::string, py::object>>();
        if (!message.contains("role") || !message.contains("content")) {
            throw py::value_error("each message needs 'role' and 'content' keys");
        }
        converted.push_back(ChatMessage{.role = message.at("role").cast<std::string>(),
                                        .content = message.at("content").cast<std::string>()});
    }
    return converted;
}

TranscribeOptions transcribeOptions(const std::optional<std::string>& language, bool translate,
                                    int threads, int beamSize) {
    return TranscribeOptions{
        .language = language,
        .translate = translate,
        .threads = threads,
        .strategy = beamSize > 1 ? DecodingStrategy::BeamSearch : DecodingStrategy::Greedy,
        .beamSize = beamSize};
}

GenerationOptions generationOptions(int maxTokens, float temperature, float topP, int topK,
                                    std::vector<std::string> stop,
                                    std::optional<std::uint32_t> seed) {
    return GenerationOptions{.maxTokens = maxTokens,
                             .temperature = temperature,
                             .topP = topP,
                             .topK = topK,
                             .stop = std::move(stop),
                             .seed = seed};
}

std::unique_ptr<GenerationStream> startStream(LlamaGenerator& generator, std::string prompt,
                                              GenerationOptions options) {
    return std::make_unique<GenerationStream>(
        [&generator, prompt = std::move(prompt), options = std::move(options)](
            const TokenCallback& emit, const jarvis::CancellationToken& cancel) {
            return generator.generate(prompt, options, emit, &cancel);
        });
}

const char* finishReasonName(FinishReason reason) {
    switch (reason) {
        case FinishReason::EndOfText:
            return "end_of_text";
        case FinishReason::MaxTokens:
            return "max_tokens";
        case FinishReason::StopString:
            return "stop_string";
        case FinishReason::Cancelled:
            return "cancelled";
        case FinishReason::ContextFull:
            return "context_full";
    }
    return "unknown";
}

// Translators are consulted most-recently-registered first, so the base
// class goes first and the derived class after it.
void defineExceptions(const py::module_& module) {
    const auto& nativeError =
        py::register_exception<jarvis::NativeError>(module, "NativeError", PyExc_RuntimeError);
    py::register_exception<jarvis::ModelError>(module, "ModelError", nativeError.ptr());
}

void defineTranscript(const py::module_& module) {
    py::class_<TranscriptSegment>(module, "TranscriptSegment", "One timestamped piece of text.")
        .def_readonly("start_ms", &TranscriptSegment::startMs)
        .def_readonly("end_ms", &TranscriptSegment::endMs)
        .def_readonly("text", &TranscriptSegment::text)
        .def("__repr__", [](const TranscriptSegment& segment) {
            return "TranscriptSegment(" + std::to_string(segment.startMs) + ", " +
                   std::to_string(segment.endMs) + ", " +
                   std::string{py::repr(py::str(segment.text))} + ")";
        });

    py::class_<Transcript>(module, "Transcript", "Result of WhisperTranscriber.transcribe().")
        .def_readonly("text", &Transcript::text)
        .def_readonly("language", &Transcript::language)
        .def_readonly("segments", &Transcript::segments)
        .def("__repr__", [](const Transcript& transcript) {
            return "Transcript(text=" + std::string{py::repr(py::str(transcript.text))} +
                   ", language=" + transcript.language +
                   ", segments=" + std::to_string(transcript.segments.size()) + ")";
        });
}

void defineWhisper(const py::module_& module) {
    py::class_<WhisperTranscriber>(module, "WhisperTranscriber",
                                   "Offline speech-to-text over whisper.cpp.")
        .def(py::init<const std::filesystem::path&>(), py::arg("model_path"),
             py::call_guard<py::gil_scoped_release>(), "Load a ggml whisper model file.")
        .def(
            "transcribe",
            [](WhisperTranscriber& self, const FloatArray& samples,
               const std::optional<std::string>& language, bool translate, int threads,
               int beamSize) {
                const std::span<const float> audio = asSpan(samples);
                const TranscribeOptions options =
                    transcribeOptions(language, translate, threads, beamSize);
                const py::gil_scoped_release release;
                return self.transcribe(audio, options);
            },
            py::arg("samples"), py::arg("language") = py::none(), py::arg("translate") = false,
            py::arg("threads") = 0, py::arg("beam_size") = 1,
            "Transcribe float32 mono samples at 16 kHz. language=None auto-detects; "
            "beam_size > 1 selects beam search.")
        .def("multilingual", &WhisperTranscriber::multilingual);
}

void defineStream(const py::module_& module) {
    py::class_<GenerationStream>(module, "GenerationStream",
                                 "Pull-based token stream. Iterate it or call next_token().")
        .def("next_token", &GenerationStream::next, py::call_guard<py::gil_scoped_release>(),
             "Block until the next piece of text arrives; None once generation has ended.")
        .def("cancel", &GenerationStream::cancel, py::call_guard<py::gil_scoped_release>(),
             "Stop generation after the current token.")
        .def("finished", &GenerationStream::finished)
        .def(
            "finish_reason",
            [](const GenerationStream& self) -> std::optional<std::string> {
                const auto result = self.result();
                if (!result.has_value()) {
                    return std::nullopt;
                }
                return std::string{finishReasonName(result->finishReason)};
            },
            "Why generation ended, or None while it is still running.")
        .def("generated_tokens",
             [](const GenerationStream& self) {
                 const auto result = self.result();
                 return result.has_value() ? result->generatedTokens : 0;
             })
        .def("__iter__", [](GenerationStream& self) -> GenerationStream& { return self; })
        .def("__next__", [](GenerationStream& self) {
            std::optional<std::string> piece;
            {
                const py::gil_scoped_release release;
                piece = self.next();
            }
            if (!piece.has_value()) {
                throw py::stop_iteration();
            }
            return *piece;
        });
}

/// The two streaming entry points; split out so each definer stays short.
void defineGeneration(py::class_<LlamaGenerator>& generator) {
    generator
        .def(
            "generate",
            [](LlamaGenerator& self, const py::sequence& messages, int maxTokens, float temperature,
               float topP, int topK, std::vector<std::string> stop,
               std::optional<std::uint32_t> seed) {
                return startStream(
                    self, self.formatPrompt(toMessages(messages)),
                    generationOptions(maxTokens, temperature, topP, topK, std::move(stop), seed));
            },
            py::arg("messages"), py::arg("max_tokens") = 256, py::arg("temperature") = 0.7F,
            py::arg("top_p") = 0.9F, py::arg("top_k") = 40,
            py::arg("stop") = std::vector<std::string>{}, py::arg("seed") = py::none(),
            py::keep_alive<0, 1>(),
            "Start generating a reply to a list of {'role', 'content'} messages on a "
            "native thread and return a GenerationStream.")
        .def(
            "generate_from_prompt",
            [](LlamaGenerator& self, std::string prompt, int maxTokens, float temperature,
               float topP, int topK, std::vector<std::string> stop,
               std::optional<std::uint32_t> seed) {
                return startStream(
                    self, std::move(prompt),
                    generationOptions(maxTokens, temperature, topP, topK, std::move(stop), seed));
            },
            py::arg("prompt"), py::arg("max_tokens") = 256, py::arg("temperature") = 0.7F,
            py::arg("top_p") = 0.9F, py::arg("top_k") = 40,
            py::arg("stop") = std::vector<std::string>{}, py::arg("seed") = py::none(),
            py::keep_alive<0, 1>(), "Like generate() but the prompt is tokenised verbatim.");
}

void defineLlama(const py::module_& module) {
    py::class_<LlamaGenerator> generator(module, "LlamaGenerator",
                                         "Offline text generation over llama.cpp.");
    generator
        .def(py::init([](const std::filesystem::path& model, int contextSize, int threads) {
                 return std::make_unique<LlamaGenerator>(
                     model, ModelOptions{.contextSize = contextSize, .threads = threads});
             }),
             py::arg("model_path"), py::arg("n_ctx") = 4096, py::arg("n_threads") = 0,
             py::call_guard<py::gil_scoped_release>(),
             "Load a GGUF model. n_threads=0 picks a default for this machine.")
        .def(
            "format_prompt",
            [](const LlamaGenerator& self, const py::sequence& messages) {
                return self.formatPrompt(toMessages(messages));
            },
            py::arg("messages"), "Render messages with the model's chat template.")
        .def("count_tokens", &LlamaGenerator::countTokens, py::arg("text"))
        .def("context_size", &LlamaGenerator::contextSize)
        .def("description", &LlamaGenerator::description);
    defineGeneration(generator);
}

}  // namespace

PYBIND11_MODULE(jarvis_inference, module) {
    module.doc() =
        "Local inference: whisper.cpp speech-to-text and llama.cpp text generation. "
        "See native/inference/README.md.";
    jarvis::bindings::defineModuleInfo(module, "implemented in step 12");

    const BackendVersions versions = backendVersions();
    module.attr("WHISPER_SAMPLE_RATE") = kWhisperSampleRate;
    module.attr("WHISPER_VERSION") = versions.whisper;
    module.attr("LLAMA_VERSION") = versions.llama;

    defineExceptions(module);
    defineTranscript(module);
    defineWhisper(module);
    defineStream(module);
    defineLlama(module);
}

// Design notes:
// - keep_alive<0, 1> ties each GenerationStream to its LlamaGenerator: the
//   producer thread holds a reference to the generator, so Python must not
//   be able to free the generator while a stream is still running.
// - The stream's destructor cancels and joins; it runs under the GIL when
//   Python drops the last reference, which is safe because the producer
//   never needs the GIL.
