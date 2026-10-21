#include "jarvis/inference/chat_template.hpp"

#include "jarvis/common/errors.hpp"
#include "jarvis/inference/llama_generator.hpp"
#include "jarvis/inference/whisper_transcriber.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using jarvis::inference::ChatMessage;
using jarvis::inference::GenerationOptions;
using jarvis::inference::kDefaultChatTemplate;
using jarvis::inference::ModelOptions;
using jarvis::inference::renderChatTemplate;
using jarvis::inference::TranscribeOptions;

TEST_CASE("chatml wraps every turn and opens an assistant turn", "[inference][chat]") {
    const std::vector<ChatMessage> messages{{.role = "system", .content = "Be brief."},
                                            {.role = "user", .content = "Say hello"}};

    const std::string prompt = renderChatTemplate(kDefaultChatTemplate, messages);

    REQUIRE(prompt ==
            "<|im_start|>system\nBe brief.<|im_end|>\n"
            "<|im_start|>user\nSay hello<|im_end|>\n"
            "<|im_start|>assistant\n");
}

TEST_CASE("a Jinja template is recognised by its markers", "[inference][chat]") {
    const std::string jinja =
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
        "message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}";
    const std::vector<ChatMessage> messages{{.role = "user", .content = "hi"}};

    const std::string prompt = renderChatTemplate(jinja, messages);

    REQUIRE(prompt == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
}

TEST_CASE("an unknown template raises ModelError", "[inference][chat]") {
    const std::vector<ChatMessage> messages{{.role = "user", .content = "hi"}};
    REQUIRE_THROWS_AS(renderChatTemplate("{{ nothing recognisable }}", messages),
                      jarvis::ModelError);
}

TEST_CASE("an empty role raises ModelError", "[inference][chat]") {
    const std::vector<ChatMessage> messages{{.role = "", .content = "hi"}};
    REQUIRE_THROWS_AS(renderChatTemplate(kDefaultChatTemplate, messages), jarvis::ModelError);
}

TEST_CASE("long conversations are rendered in full", "[inference][chat]") {
    std::vector<ChatMessage> messages;
    for (int i = 0; i < 50; ++i) {
        messages.push_back({.role = "user", .content = std::string(200, 'x')});
        messages.push_back({.role = "assistant", .content = std::string(200, 'y')});
    }

    const std::string prompt = renderChatTemplate(kDefaultChatTemplate, messages);

    REQUIRE(prompt.size() > 20000);
    REQUIRE(prompt.ends_with("<|im_start|>assistant\n"));
}

TEST_CASE("option defaults are the documented ones", "[inference][options]") {
    const GenerationOptions generation;
    REQUIRE(generation.maxTokens == 256);
    REQUIRE(generation.temperature == 0.7F);
    REQUIRE(generation.topP == 0.9F);
    REQUIRE(generation.topK == 40);
    REQUIRE(generation.stop.empty());
    REQUIRE_FALSE(generation.seed.has_value());

    const ModelOptions model;
    REQUIRE(model.contextSize == 4096);
    REQUIRE(model.threads == 0);
    REQUIRE(model.gpuLayers == 0);

    const TranscribeOptions transcribe;
    REQUIRE_FALSE(transcribe.language.has_value());
    REQUIRE_FALSE(transcribe.translate);
    REQUIRE(transcribe.threads == 0);
    REQUIRE(transcribe.beamSize == 5);
}
