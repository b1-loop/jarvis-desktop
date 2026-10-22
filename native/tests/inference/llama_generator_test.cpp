#include "jarvis/inference/llama_generator.hpp"

#include "jarvis/common/cancellation_token.hpp"
#include "jarvis/common/errors.hpp"
#include "jarvis/inference/chat_template.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

using jarvis::CancellationToken;
using jarvis::inference::ChatMessage;
using jarvis::inference::FinishReason;
using jarvis::inference::GenerationOptions;
using jarvis::inference::LlamaGenerator;
using jarvis::inference::ModelOptions;

namespace {

std::filesystem::path modelPath() {
    return std::filesystem::path{JARVIS_MODELS_DIR} / "qwen2.5-0.5b-instruct-q4_k_m.gguf";
}

void requireModel() {
    if (!std::filesystem::exists(modelPath())) {
        SKIP("Qwen2.5-0.5B GGUF not downloaded; run scripts/download_models.py --local");
    }
}

/// A small context keeps the KV cache allocation (and the test) fast.
ModelOptions smallContext() {
    return ModelOptions{.contextSize = 512, .threads = 2};
}

std::vector<ChatMessage> sayHello() {
    return {{.role = "system", .content = "You are a terse assistant."},
            {.role = "user", .content = "Say hello"}};
}

}  // namespace

TEST_CASE("a missing GGUF file raises ModelError", "[inference][llama][model]") {
    REQUIRE_THROWS_AS((LlamaGenerator{std::filesystem::path{"does-not-exist.gguf"}}),
                      jarvis::ModelError);
}

TEST_CASE("eight tokens from 'Say hello' stream non-empty text", "[inference][llama][model]") {
    requireModel();
    LlamaGenerator generator{modelPath(), smallContext()};
    std::vector<std::string> pieces;

    const auto result =
        generator.generate(sayHello(), GenerationOptions{.maxTokens = 8, .temperature = 0.0F},
                           [&pieces](std::string_view piece) {
                               pieces.emplace_back(piece);
                               return true;
                           });

    REQUIRE_FALSE(pieces.empty());
    REQUIRE_FALSE(result.text.empty());
    REQUIRE(result.generatedTokens <= 8);
    REQUIRE(result.generatedTokens >= 1);
    REQUIRE(result.promptTokens > 0);
    std::string joined;
    for (const auto& piece : pieces) {
        joined += piece;
    }
    REQUIRE(joined == result.text);
}

TEST_CASE("returning false from the callback stops after one piece", "[inference][llama][model]") {
    requireModel();
    LlamaGenerator generator{modelPath(), smallContext()};
    int calls = 0;

    const auto result = generator.generate(sayHello(), GenerationOptions{.maxTokens = 64},
                                           [&calls](std::string_view) {
                                               ++calls;
                                               return false;
                                           });

    REQUIRE(calls == 1);
    REQUIRE(result.finishReason == FinishReason::Cancelled);
}

TEST_CASE("a raised cancellation token stops generation", "[inference][llama][model]") {
    requireModel();
    LlamaGenerator generator{modelPath(), smallContext()};
    CancellationToken cancel;
    int calls = 0;

    const auto result = generator.generate(
        sayHello(), GenerationOptions{.maxTokens = 64},
        [&](std::string_view) {
            if (++calls == 2) {
                cancel.cancel();
            }
            return true;
        },
        &cancel);

    REQUIRE(calls == 2);
    REQUIRE(result.finishReason == FinishReason::Cancelled);
}

TEST_CASE("a stop string ends generation and is not delivered", "[inference][llama][model]") {
    requireModel();
    LlamaGenerator generator{modelPath(), smallContext()};

    const auto result =
        generator.generate("Count: 1, 2, 3, 4, 5, 6, 7",
                           GenerationOptions{.maxTokens = 32, .temperature = 0.0F, .stop = {"4"}},
                           [](std::string_view) { return true; });

    REQUIRE(result.text.find('4') == std::string::npos);
}

TEST_CASE("the prompt is rendered with the model's own template", "[inference][llama][model]") {
    requireModel();
    const LlamaGenerator generator{modelPath(), smallContext()};

    const std::string prompt = generator.formatPrompt(sayHello());

    REQUIRE(prompt.find("<|im_start|>user\nSay hello<|im_end|>") != std::string::npos);
    REQUIRE(prompt.ends_with("<|im_start|>assistant\n"));
    REQUIRE(generator.countTokens(prompt) > 5);
    REQUIRE(generator.contextSize() == 512);
    REQUIRE_FALSE(generator.description().empty());
}

TEST_CASE("a prompt longer than the context raises ModelError", "[inference][llama][model]") {
    requireModel();
    LlamaGenerator generator{modelPath(), ModelOptions{.contextSize = 64, .threads = 2}};
    // llama.cpp pads tiny contexts up to 256 tokens, so the prompt must be
    // well beyond that; a repeated phrase tokenises to roughly one token a
    // word, unlike a run of identical characters.
    std::string prompt;
    for (int i = 0; i < 400; ++i) {
        prompt += "the quick brown fox jumps over the lazy dog ";
    }

    REQUIRE_THROWS_AS(
        generator.generate(prompt, GenerationOptions{}, [](std::string_view) { return true; }),
        jarvis::ModelError);
}
