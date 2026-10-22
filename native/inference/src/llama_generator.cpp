#include "jarvis/inference/llama_generator.hpp"

#include "jarvis/common/cancellation_token.hpp"
#include "jarvis/common/errors.hpp"
#include "jarvis/inference/backend.hpp"
#include "jarvis/inference/chat_template.hpp"
#include "jarvis/inference/stop_matcher.hpp"

#include <llama.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace jarvis::inference {

namespace {

/// Tokens handed to llama_decode per call while ingesting the prompt. 512
/// matches llama.cpp's physical batch default; larger batches do not help on
/// a CPU and would only raise the compute buffer size.
constexpr std::uint32_t kPromptBatchTokens = 512;

struct ModelDeleter {
    void operator()(llama_model* model) const noexcept { llama_model_free(model); }
};
struct ContextDeleter {
    void operator()(llama_context* context) const noexcept { llama_free(context); }
};
struct SamplerDeleter {
    void operator()(llama_sampler* sampler) const noexcept { llama_sampler_free(sampler); }
};
using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

/// Half the logical cores approximates the physical core count on machines
/// with SMT, which is what llama.cpp's own tools default to: more threads
/// than physical cores slows matrix multiplication down.
int defaultThreads() noexcept {
    const unsigned logical = std::thread::hardware_concurrency();
    return static_cast<int>(std::max(1U, logical / 2));
}

ModelPtr loadModel(const std::filesystem::path& path, const ModelOptions& options) {
    if (!std::filesystem::exists(path)) {
        throw ModelError("model file not found: " + path.string());
    }
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = options.gpuLayers;
    const std::string file = path.string();
    ModelPtr model{llama_model_load_from_file(file.c_str(), params)};
    if (model == nullptr) {
        throw ModelError("failed to load GGUF model " + file);
    }
    return model;
}

ContextPtr createContext(llama_model* model, const ModelOptions& options) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = static_cast<std::uint32_t>(std::max(options.contextSize, 1));
    params.n_batch = std::min(kPromptBatchTokens, params.n_ctx);
    params.n_ubatch = params.n_batch;
    params.n_threads = options.threads > 0 ? options.threads : defaultThreads();
    params.n_threads_batch = params.n_threads;
    params.no_perf = true;
    ContextPtr context{llama_init_from_model(model, params)};
    if (context == nullptr) {
        throw ModelError("failed to create a llama context with n_ctx=" +
                         std::to_string(params.n_ctx));
    }
    return context;
}

SamplerPtr makeSampler(const GenerationOptions& options) {
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
    params.no_perf = true;
    SamplerPtr chain{llama_sampler_chain_init(params)};
    if (options.temperature <= 0.0F) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
        return chain;
    }
    // The order mirrors llama.cpp's common sampler: truncate the
    // distribution, then scale it, then draw from it.
    llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(options.topK));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(options.topP, 1));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(options.temperature));
    llama_sampler_chain_add(chain.get(),
                            llama_sampler_init_dist(options.seed.value_or(LLAMA_DEFAULT_SEED)));
    return chain;
}

std::vector<llama_token> tokenize(const llama_vocab* vocab, std::string_view text,
                                  bool addSpecial) {
    const std::string owned{text};  // llama_tokenize wants a contiguous C string
    const auto length = static_cast<std::int32_t>(owned.size());
    const std::int32_t needed =
        -llama_tokenize(vocab, owned.c_str(), length, nullptr, 0, addSpecial, true);
    std::vector<llama_token> tokens(static_cast<std::size_t>(std::max(needed, 0)));
    const std::int32_t count =
        llama_tokenize(vocab, owned.c_str(), length, tokens.data(), needed, addSpecial, true);
    if (count < 0) {
        throw ModelError("tokenisation failed");
    }
    tokens.resize(static_cast<std::size_t>(count));
    return tokens;
}

/// Text of one token. Control tokens render as empty strings so they never
/// reach the speech synthesiser.
std::string tokenToPiece(const llama_vocab* vocab, llama_token token) {
    std::array<char, 64> buffer{};
    std::int32_t written = llama_token_to_piece(vocab, token, buffer.data(),
                                                static_cast<std::int32_t>(buffer.size()), 0, false);
    if (written >= 0) {
        return std::string{buffer.data(), static_cast<std::size_t>(written)};
    }
    std::string piece(static_cast<std::size_t>(-written), '\0');
    written = llama_token_to_piece(vocab, token, piece.data(), -written, 0, false);
    piece.resize(static_cast<std::size_t>(std::max(written, 0)));
    return piece;
}

void checkDecode(std::int32_t status, std::string_view stage) {
    if (status != 0) {
        throw ModelError("llama_decode failed with code " + std::to_string(status) + " while " +
                         std::string{stage});
    }
}

/// One generation pass over a context whose memory has been cleared.
class Decoder {
public:
    Decoder(llama_context* context, const llama_vocab* vocab, GenerationOptions options,
            TokenCallback onToken, const CancellationToken* cancel)
        : context_(context),
          vocab_(vocab),
          options_(std::move(options)),
          onToken_(std::move(onToken)),
          cancel_(cancel),
          sampler_(makeSampler(options_)),
          matcher_(options_.stop) {}

    GenerationResult run(std::vector<llama_token> prompt) {
        ingest(prompt);
        result_.promptTokens = static_cast<int>(prompt.size());
        position_ = static_cast<int>(prompt.size());
        std::optional<FinishReason> reason;
        while (!reason.has_value()) {
            reason = step();
        }
        if (*reason != FinishReason::Cancelled) {
            deliver(matcher_.flush());
        }
        result_.finishReason = *reason;
        return std::move(result_);
    }

private:
    void ingest(std::vector<llama_token>& prompt) {
        // The context's logical batch may be smaller than the constant when
        // the context itself is tiny; exceeding it aborts inside llama.cpp.
        const auto batchSize = static_cast<std::size_t>(llama_n_batch(context_));
        for (std::size_t offset = 0; offset < prompt.size(); offset += batchSize) {
            const std::size_t count = std::min(batchSize, prompt.size() - offset);
            checkDecode(
                llama_decode(context_, llama_batch_get_one(prompt.data() + offset,
                                                           static_cast<std::int32_t>(count))),
                "decoding the prompt");
        }
    }

    [[nodiscard]] bool cancelled() const noexcept {
        return cancel_ != nullptr && cancel_->cancelled();
    }

    /// Hand @p text to the caller; false means the caller wants to stop.
    bool deliver(const std::string& text) {
        if (text.empty()) {
            return true;
        }
        result_.text += text;
        return onToken_(text);
    }

    /// Sample, deliver and decode one token. @return Why to stop, if at all.
    std::optional<FinishReason> step() {
        if (result_.generatedTokens >= options_.maxTokens) {
            return FinishReason::MaxTokens;
        }
        if (cancelled()) {
            return FinishReason::Cancelled;
        }
        if (std::cmp_greater_equal(position_, llama_n_ctx(context_))) {
            return FinishReason::ContextFull;
        }
        llama_token token = llama_sampler_sample(sampler_.get(), context_, -1);
        if (llama_vocab_is_eog(vocab_, token)) {
            return FinishReason::EndOfText;
        }
        ++result_.generatedTokens;
        if (!deliver(matcher_.feed(tokenToPiece(vocab_, token)))) {
            return FinishReason::Cancelled;
        }
        if (matcher_.stopped()) {
            return FinishReason::StopString;
        }
        checkDecode(llama_decode(context_, llama_batch_get_one(&token, 1)), "generating");
        ++position_;
        return std::nullopt;
    }

    llama_context* context_;
    const llama_vocab* vocab_;
    GenerationOptions options_;
    TokenCallback onToken_;
    const CancellationToken* cancel_;
    SamplerPtr sampler_;
    StopMatcher matcher_;
    GenerationResult result_;
    int position_{0};
};

}  // namespace

struct LlamaGenerator::Impl {
    std::mutex mutex;  // the context holds the KV cache of the call in progress
    ModelOptions options;
    ModelPtr model;
    ContextPtr context;
    const llama_vocab* vocab{nullptr};
    std::string chatTemplate;
};

LlamaGenerator::LlamaGenerator(const std::filesystem::path& model, ModelOptions options)
    : impl_(std::make_unique<Impl>()) {
    ensureBackend();
    impl_->options = options;
    impl_->model = loadModel(model, options);
    impl_->context = createContext(impl_->model.get(), options);
    impl_->vocab = llama_model_get_vocab(impl_->model.get());
    const char* tmpl = llama_model_chat_template(impl_->model.get(), nullptr);
    impl_->chatTemplate = tmpl == nullptr ? std::string{kDefaultChatTemplate} : std::string{tmpl};
}

LlamaGenerator::~LlamaGenerator() = default;

GenerationResult LlamaGenerator::generate(std::string_view prompt, const GenerationOptions& options,
                                          const TokenCallback& onToken,
                                          const CancellationToken* cancel) {
    const std::scoped_lock lock(impl_->mutex);
    std::vector<llama_token> tokens = tokenize(impl_->vocab, prompt, true);
    const auto limit = static_cast<std::size_t>(llama_n_ctx(impl_->context.get()));
    if (tokens.size() >= limit) {
        throw ModelError("prompt of " + std::to_string(tokens.size()) +
                         " tokens does not fit the context of " + std::to_string(limit));
    }
    llama_memory_clear(llama_get_memory(impl_->context.get()), true);
    Decoder decoder(impl_->context.get(), impl_->vocab, options, onToken, cancel);
    return decoder.run(std::move(tokens));
}

GenerationResult LlamaGenerator::generate(std::span<const ChatMessage> messages,
                                          const GenerationOptions& options,
                                          const TokenCallback& onToken,
                                          const CancellationToken* cancel) {
    return generate(formatPrompt(messages), options, onToken, cancel);
}

std::string LlamaGenerator::formatPrompt(std::span<const ChatMessage> messages) const {
    return renderChatTemplate(impl_->chatTemplate, messages);
}

int LlamaGenerator::countTokens(std::string_view text) const {
    return static_cast<int>(tokenize(impl_->vocab, text, false).size());
}

int LlamaGenerator::contextSize() const noexcept {
    return static_cast<int>(llama_n_ctx(impl_->context.get()));
}

std::string LlamaGenerator::description() const {
    std::array<char, 256> buffer{};
    const std::int32_t written = llama_model_desc(impl_->model.get(), buffer.data(), buffer.size());
    return std::string{buffer.data(), static_cast<std::size_t>(std::clamp<std::int32_t>(
                                          written, 0, static_cast<std::int32_t>(buffer.size())))};
}

}  // namespace jarvis::inference

// Design notes:
// - One context per generator, cleared before every call: a voice assistant
//   runs one conversation at a time, and the Python side re-sends the
//   history each turn, so prefix caching across calls is a later
//   optimisation rather than a requirement.
// - Sampling uses the llama_sampler chain API instead of reading logits by
//   hand so top-k/top-p/temperature behave exactly as in llama.cpp's tools.
// - The assistant tokens are decoded one at a time, which is what makes
//   streaming possible; prompt tokens go in batches of 512.
