/// @file chat_template.hpp
/// @brief Chat messages and their rendering into a model prompt.

#pragma once

#include <span>
#include <string>
#include <string_view>

namespace jarvis::inference {

/// One turn of a conversation, in the OpenAI-style role/content shape that
/// the Python LLMClient interface already uses.
struct ChatMessage {
    std::string role;  ///< "system", "user" or "assistant".
    std::string content;
};

/// Name of the template used when a model ships none. ChatML is what
/// llama.cpp itself falls back to and what the pinned test model uses.
inline constexpr std::string_view kDefaultChatTemplate = "chatml";

/// Render @p messages with the chat template @p tmpl, ending with the marker
/// that starts an assistant turn.
///
/// @p tmpl is either a Jinja template string read from a GGUF file or the
/// name of one of llama.cpp's built-in templates ("chatml", "llama3", ...).
/// Throws ModelError when the template is not recognised or a message has
/// an empty role.
[[nodiscard]] std::string renderChatTemplate(std::string_view tmpl,
                                             std::span<const ChatMessage> messages);

}  // namespace jarvis::inference
