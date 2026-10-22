#include "jarvis/overlay/model.hpp"

#include "jarvis/overlay/layout.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <iterator>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jarvis::overlay {

namespace {

constexpr std::array<std::pair<AssistantState, std::string_view>, 4> kStateNames{{
    {AssistantState::Idle, "IDLE"},
    {AssistantState::Listening, "LISTENING"},
    {AssistantState::Thinking, "THINKING"},
    {AssistantState::Speaking, "SPEAKING"},
}};

}  // namespace

std::string_view toString(AssistantState state) noexcept {
    const auto found =
        std::ranges::find(kStateNames, state, &std::pair<AssistantState, std::string_view>::first);
    return found == kStateNames.end() ? std::string_view{"UNKNOWN"} : found->second;
}

std::optional<AssistantState> parseAssistantState(std::string_view name) noexcept {
    const auto found =
        std::ranges::find(kStateNames, name, &std::pair<AssistantState, std::string_view>::second);
    if (found == kStateNames.end()) {
        return std::nullopt;
    }
    return found->first;
}

// --- LevelRing -------------------------------------------------------------

LevelRing::LevelRing(std::size_t capacity) : levels_(std::max<std::size_t>(capacity, 1), 0.0F) {}

void LevelRing::push(float level) noexcept {
    levels_.at(next_) = std::clamp(level, 0.0F, 1.0F);
    next_ = (next_ + 1) % levels_.size();
}

void LevelRing::clear() noexcept {
    std::ranges::fill(levels_, 0.0F);
    next_ = 0;
}

std::vector<float> LevelRing::ordered() const {
    std::vector<float> out;
    out.reserve(levels_.size());
    const auto split = std::next(levels_.begin(), static_cast<std::ptrdiff_t>(next_));
    out.insert(out.end(), split, levels_.end());
    out.insert(out.end(), levels_.begin(), split);
    return out;
}

float LevelRing::latest() const noexcept {
    const std::size_t last = (next_ + levels_.size() - 1) % levels_.size();
    return levels_.at(last);
}

// --- OverlayModel ----------------------------------------------------------

OverlayModel::OverlayModel(ModelLimits limits) : limits_(limits), levels_(limits.levelHistory) {}

void OverlayModel::setState(AssistantState state) {
    const std::scoped_lock lock(mutex_);
    if (state == state_) {
        return;
    }
    state_ = state;
    if (state == AssistantState::Idle) {
        levels_.clear();
    }
    touch();
}

void OverlayModel::pushLevel(float level) {
    const std::scoped_lock lock(mutex_);
    levels_.push(level);
    touch();
}

void OverlayModel::setTranscript(std::string_view utf8) {
    std::string bounded = truncateUtf8Tail(utf8, limits_.maxTextChars);
    const std::scoped_lock lock(mutex_);
    transcript_ = std::move(bounded);
    touch();
}

void OverlayModel::setResponse(std::string_view utf8) {
    std::string bounded = truncateUtf8Tail(utf8, limits_.maxTextChars);
    const std::scoped_lock lock(mutex_);
    response_ = std::move(bounded);
    touch();
}

void OverlayModel::setVisible(bool visible) {
    const std::scoped_lock lock(mutex_);
    if (visible == visible_) {
        return;
    }
    visible_ = visible;
    touch();
}

void OverlayModel::flashWakeWord(Clock::time_point now) {
    const std::scoped_lock lock(mutex_);
    wakeFlashAt_ = now;
    touch();
}

OverlaySnapshot OverlayModel::snapshot(Clock::time_point now) const {
    const std::scoped_lock lock(mutex_);
    return OverlaySnapshot{
        .state = state_,
        .levels = levels_.ordered(),
        .transcript = transcript_,
        .response = response_,
        .visible = visible_,
        .wakeFlash = wakeFlashAt(now),
        .revision = revision_.load(std::memory_order_relaxed),
    };
}

AssistantState OverlayModel::state() const {
    const std::scoped_lock lock(mutex_);
    return state_;
}

bool OverlayModel::visible() const {
    const std::scoped_lock lock(mutex_);
    return visible_;
}

void OverlayModel::touch() noexcept {
    revision_.fetch_add(1, std::memory_order_release);
}

float OverlayModel::wakeFlashAt(Clock::time_point now) const {
    if (!wakeFlashAt_.has_value() || limits_.wakeFlashDuration.count() <= 0) {
        return 0.0F;
    }
    const auto elapsed = std::chrono::duration<float, std::milli>(now - *wakeFlashAt_);
    const auto duration = std::chrono::duration<float, std::milli>(limits_.wakeFlashDuration);
    return std::clamp(1.0F - (elapsed.count() / duration.count()), 0.0F, 1.0F);
}

}  // namespace jarvis::overlay

// Design notes:
// - Text is bounded at the setter, not at draw time, so the mutex is held for
//   a string move rather than a UTF-8 walk, and the stored size is capped no
//   matter what Python sends.
// - The wake flash is a timestamp, not a counter the renderer decrements:
//   the model does not know the frame rate and should not have to.
// - revision_ is atomic so the UI thread can decide whether to render at all
//   without taking the lock on every tick.
