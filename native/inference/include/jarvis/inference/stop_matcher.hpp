/// @file stop_matcher.hpp
/// @brief Pure helpers for turning raw token pieces into deliverable text.

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::inference {

/// Length of the longest prefix of @p bytes that ends on a UTF-8 character
/// boundary. Tokens frequently split multi-byte characters (an emoji or a
/// Cyrillic letter can be two tokens), and Python rejects partial UTF-8, so
/// the trailing incomplete sequence is held back until the next piece.
[[nodiscard]] std::size_t completeUtf8Prefix(std::string_view bytes) noexcept;

/// Streams generated text while watching for stop strings.
///
/// Text is released as soon as it can no longer be the start of any stop
/// string, so the caller sees output with at most (longest stop string - 1)
/// characters of delay. The stop string itself is never delivered. Incomplete
/// UTF-8 sequences are held back the same way.
class StopMatcher {
public:
    explicit StopMatcher(std::vector<std::string> stops);

    /// Feed @p piece. @return Text that is now safe to deliver (possibly
    /// empty). After a stop string matched, stopped() is true and further
    /// pieces are ignored.
    [[nodiscard]] std::string feed(std::string_view piece);

    /// Release everything still held back, except an incomplete trailing
    /// UTF-8 sequence, which is dropped. Call when generation ends.
    [[nodiscard]] std::string flush();

    [[nodiscard]] bool stopped() const noexcept { return stopped_; }

private:
    /// How many trailing characters of pending_ could still begin a stop
    /// string.
    [[nodiscard]] std::size_t heldBack() const noexcept;

    std::vector<std::string> stops_;
    std::string pending_;
    bool stopped_{false};
};

}  // namespace jarvis::inference
