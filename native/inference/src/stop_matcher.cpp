#include "jarvis/inference/stop_matcher.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jarvis::inference {

namespace {

constexpr unsigned char kContinuationMask = 0xC0;
constexpr unsigned char kContinuationBits = 0x80;

/// Number of bytes in the UTF-8 sequence that starts with @p lead, or 1 for
/// ASCII and for invalid lead bytes (which are passed through unchanged).
std::size_t sequenceLength(unsigned char lead) noexcept {
    if ((lead & 0x80U) == 0) {
        return 1;
    }
    if ((lead & 0xE0U) == 0xC0U) {
        return 2;
    }
    if ((lead & 0xF0U) == 0xE0U) {
        return 3;
    }
    if ((lead & 0xF8U) == 0xF0U) {
        return 4;
    }
    return 1;
}

bool isContinuation(unsigned char byte) noexcept {
    return (byte & kContinuationMask) == kContinuationBits;
}

}  // namespace

std::size_t completeUtf8Prefix(std::string_view bytes) noexcept {
    // Walk back over at most three continuation bytes to find the last lead
    // byte, then check whether its sequence is complete.
    std::size_t lead = bytes.size();
    while (lead > 0 && bytes.size() - lead < 3 &&
           isContinuation(static_cast<unsigned char>(bytes.at(lead - 1)))) {
        --lead;
    }
    if (lead == 0) {
        return bytes.size();
    }
    --lead;
    const std::size_t needed = sequenceLength(static_cast<unsigned char>(bytes.at(lead)));
    return bytes.size() - lead >= needed ? bytes.size() : lead;
}

StopMatcher::StopMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {
    std::erase_if(stops_, [](const std::string& stop) { return stop.empty(); });
}

std::string StopMatcher::feed(std::string_view piece) {
    if (stopped_) {
        return {};
    }
    pending_ += piece;
    for (const auto& stop : stops_) {
        const std::size_t at = pending_.find(stop);
        if (at != std::string::npos) {
            stopped_ = true;
            std::string released = pending_.substr(0, at);
            pending_.clear();
            return released;
        }
    }
    const std::size_t safe = std::min(pending_.size() - heldBack(), completeUtf8Prefix(pending_));
    std::string released = pending_.substr(0, safe);
    pending_.erase(0, safe);
    return released;
}

std::string StopMatcher::flush() {
    std::string released = pending_.substr(0, completeUtf8Prefix(pending_));
    pending_.clear();
    return released;
}

std::size_t StopMatcher::heldBack() const noexcept {
    std::size_t longest = 0;
    for (const auto& stop : stops_) {
        // The longest suffix of pending_ that is a proper prefix of stop.
        const std::size_t limit = std::min(pending_.size(), stop.size() - 1);
        for (std::size_t length = limit; length > longest; --length) {
            const std::string_view tail{pending_.data() + (pending_.size() - length), length};
            if (stop.compare(0, length, tail) == 0) {
                longest = length;
                break;
            }
        }
    }
    return longest;
}

}  // namespace jarvis::inference

// Design notes:
// - Matching is byte-wise on UTF-8, which is correct for stop strings
//   because a UTF-8 encoding never matches in the middle of a character.
// - The matcher is deliberately free of llama.cpp types so it is unit-tested
//   without a model and could serve a cloud streaming client equally well.
