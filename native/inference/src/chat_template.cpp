#include "jarvis/inference/chat_template.hpp"

#include "jarvis/common/errors.hpp"

#include <llama.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::inference {

namespace {

std::vector<llama_chat_message> toLlama(std::span<const ChatMessage> messages) {
    std::vector<llama_chat_message> converted;
    converted.reserve(messages.size());
    for (const auto& message : messages) {
        if (message.role.empty()) {
            throw ModelError("chat message has an empty role");
        }
        converted.push_back(
            llama_chat_message{.role = message.role.c_str(), .content = message.content.c_str()});
    }
    return converted;
}

std::size_t totalLength(std::span<const ChatMessage> messages) {
    std::size_t total = 0;
    for (const auto& message : messages) {
        total += message.role.size() + message.content.size();
    }
    return total;
}

}  // namespace

std::string renderChatTemplate(std::string_view tmpl, std::span<const ChatMessage> messages) {
    const std::string templateText{tmpl};
    const std::vector<llama_chat_message> chat = toLlama(messages);
    // llama.cpp recommends twice the input as the initial buffer and reports
    // the required size when that is not enough.
    std::string buffer((2 * totalLength(messages)) + 256, '\0');
    std::int32_t written =
        llama_chat_apply_template(templateText.c_str(), chat.data(), chat.size(), true,
                                  buffer.data(), static_cast<std::int32_t>(buffer.size()));
    if (written < 0) {
        throw ModelError("unsupported chat template: " + templateText.substr(0, 64));
    }
    if (static_cast<std::size_t>(written) > buffer.size()) {
        buffer.resize(static_cast<std::size_t>(written));
        written = llama_chat_apply_template(templateText.c_str(), chat.data(), chat.size(), true,
                                            buffer.data(), written);
    }
    buffer.resize(static_cast<std::size_t>(written));
    return buffer;
}

}  // namespace jarvis::inference

// Design notes:
// - llama_chat_apply_template is not a Jinja engine; it recognises the
//   templates of the popular model families by their markers. That covers
//   every model Jarvis documents and keeps the Jinja dependency out of the
//   build. A model with an exotic template fails loudly with ModelError
//   rather than silently producing a mis-formatted prompt.
