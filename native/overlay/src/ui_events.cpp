#include "jarvis/overlay/ui_events.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace jarvis::overlay {

namespace {

/// Names in the order of the UiEvent alternatives.
constexpr std::array<std::string_view, std::variant_size_v<UiEvent>> kKindNames{
    "submit_text",     "close",    "quit",    "toggle_mute",     "set_expanded",
    "setting_changed", "open_log", "restart", "settings_opened", "placement_changed",
    "minimize",        "restore"};

}  // namespace

std::string_view kindOf(const UiEvent& event) {
    return kKindNames.at(event.index());
}

bool coalesces(const UiEvent& newest, const UiEvent& next) noexcept {
    if (const auto* setting = std::get_if<SettingChanged>(&next); setting != nullptr) {
        const auto* previous = std::get_if<SettingChanged>(&newest);
        return previous != nullptr && previous->key == setting->key;
    }
    if (const auto* placement = std::get_if<PlacementChanged>(&next); placement != nullptr) {
        const auto* previous = std::get_if<PlacementChanged>(&newest);
        return previous != nullptr && previous->expanded == placement->expanded &&
               previous->orb == placement->orb;
    }
    return false;
}

UiEventQueue::UiEventQueue(std::size_t capacity) : capacity_(std::max<std::size_t>(capacity, 1)) {}

void UiEventQueue::push(UiEvent event) {
    const std::scoped_lock lock(mutex_);
    if (!events_.empty() && coalesces(events_.back(), event)) {
        events_.back() = std::move(event);
        return;
    }
    if (events_.size() >= capacity_) {
        events_.pop_front();
        ++dropped_;
    }
    events_.push_back(std::move(event));
}

std::vector<UiEvent> UiEventQueue::drain() {
    std::deque<UiEvent> taken;
    {
        const std::scoped_lock lock(mutex_);
        taken.swap(events_);
    }
    return {std::make_move_iterator(taken.begin()), std::make_move_iterator(taken.end())};
}

std::size_t UiEventQueue::size() const {
    const std::scoped_lock lock(mutex_);
    return events_.size();
}

std::uint64_t UiEventQueue::dropped() const {
    const std::scoped_lock lock(mutex_);
    return dropped_;
}

}  // namespace jarvis::overlay

// Design notes:
// - A mutex-guarded deque, not a lock-free ring: events arrive at human
//   speed (a few per second at most, coalesced), the critical sections are
//   a handful of moves, and a variant with strings is not trivially
//   copyable, which every simple lock-free ring would need.
// - drain() swaps the deque out under the lock and builds the vector after
//   releasing it, so Python (holding the GIL) never makes the UI thread
//   wait for an allocation.
// - Coalescing only touches the newest event. Replacing an older pending
//   SettingChanged for the same key would reorder it past events queued
//   after it (a Restart, say), which could change what the restart sees.
