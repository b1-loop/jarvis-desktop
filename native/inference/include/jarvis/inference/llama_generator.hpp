/// @file llama_generator.hpp
/// @brief Offline text generation over llama.cpp with streamed tokens.

#pragma once

#include "jarvis/common/cancellation_token.hpp"
#include "jarvis/inference/chat_template.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::inference {

/// How a model is loaded. Fixed for the lifetime of a LlamaGenerator.
struct ModelOptions {
    /// Context window in tokens; 4096 fits a system prompt, a few tool
    /// descriptions and a short conversation on a laptop CPU.
    int contextSize = 4096;
    /// Threads for decoding; 0 picks llama.cpp's default for this machine.
    int threads = 0;
    /// Layers to offload to a GPU. Always 0 in the CPU-only build; the
    /// field exists so the API does not change when CUDA arrives.
    int gpuLayers = 0;
};

/// Sampling settings for one generate() call.
struct GenerationOptions {
    int maxTokens = 256;
    /// 0 or below selects greedy decoding.
    float temperature = 0.7F;
    float topP = 0.9F;
    int topK = 40;
    /// Generation ends when any of these appears; the string itself is not
    /// delivered.
    std::vector<std::string> stop;
    /// Random seed for sampling; std::nullopt draws a fresh one per call.
    std::optional<std::uint32_t> seed;
};

/// Why generation ended.
enum class FinishReason {
    EndOfText,   ///< The model emitted an end-of-generation token.
    MaxTokens,   ///< GenerationOptions::maxTokens was reached.
    StopString,  ///< A stop string matched.
    Cancelled,   ///< The callback returned false or the token was cancelled.
    ContextFull  ///< The context window ran out.
};

/// Summary returned once generation stops.
struct GenerationResult {
    std::string text;  ///< Everything delivered through the callback.
    int promptTokens{0};
    int generatedTokens{0};
    FinishReason finishReason{FinishReason::EndOfText};
};

/// Receives each piece of text as soon as it is sampled. Return false to
/// stop generation after this piece.
using TokenCallback = std::function<bool(std::string_view piece)>;

/// Loads one GGUF model and generates text from prompts or chat messages.
///
/// Thread-safe for sequential use: generate() holds an internal mutex for
/// its whole duration because the llama context carries the KV cache of the
/// call in progress. Tokens are handed to the callback on the calling
/// thread, one at a time, as they are produced.
class LlamaGenerator {
public:
    /// Load @p model. Throws ModelError if the file is missing or invalid.
    explicit LlamaGenerator(const std::filesystem::path& model, ModelOptions options = {});
    ~LlamaGenerator();

    LlamaGenerator(const LlamaGenerator&) = delete;
    LlamaGenerator& operator=(const LlamaGenerator&) = delete;
    LlamaGenerator(LlamaGenerator&&) = delete;
    LlamaGenerator& operator=(LlamaGenerator&&) = delete;

    /// Generate a continuation of @p prompt, which is tokenised verbatim.
    /// @param cancel Optional, non-owning; checked before every token.
    /// Throws ModelError if the prompt does not fit the context or decoding
    /// fails.
    GenerationResult generate(std::string_view prompt, const GenerationOptions& options,
                              const TokenCallback& onToken,
                              const CancellationToken* cancel = nullptr);

    /// Render @p messages with the model's chat template and generate the
    /// assistant's reply.
    GenerationResult generate(std::span<const ChatMessage> messages,
                              const GenerationOptions& options, const TokenCallback& onToken,
                              const CancellationToken* cancel = nullptr);

    /// The prompt generate(messages, ...) would decode, for inspection and
    /// token budgeting on the Python side.
    [[nodiscard]] std::string formatPrompt(std::span<const ChatMessage> messages) const;

    /// Number of tokens @p text occupies in this model's vocabulary.
    [[nodiscard]] int countTokens(std::string_view text) const;

    [[nodiscard]] int contextSize() const noexcept;

    /// Short human-readable description from the GGUF metadata.
    [[nodiscard]] std::string description() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace jarvis::inference
