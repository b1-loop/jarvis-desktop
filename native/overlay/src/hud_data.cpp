#include "jarvis/overlay/hud_data.hpp"

#include "jarvis/overlay/layout.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace jarvis::overlay {

HudDataModel::HudDataModel(HudDataLimits limits, Appearance appearance) : limits_(limits) {
    limits_.maxEntries = std::max<std::size_t>(limits_.maxEntries, 1);
    data_.history = std::make_shared<const std::vector<HistoryEntry>>();
    data_.settings = std::make_shared<const SettingsPage>();
    data_.appearance = appearance;
}

void HudDataModel::setMuted(bool muted) {
    const std::scoped_lock lock(mutex_);
    data_.muted = muted;
    ++data_.revision;
}

void HudDataModel::addHistory(HistoryEntry entry) {
    entry.text = truncateUtf8Head(entry.text, limits_.maxEntryChars);
    const std::scoped_lock lock(mutex_);
    // Copy-on-write: frames already holding the old block keep drawing it.
    auto next = std::make_shared<std::vector<HistoryEntry>>();
    const std::vector<HistoryEntry>& current = *data_.history;
    const std::size_t keep = std::min(current.size(), limits_.maxEntries - 1);
    next->reserve(keep + 1);
    next->insert(next->end(), current.end() - static_cast<std::ptrdiff_t>(keep), current.end());
    next->push_back(std::move(entry));
    data_.history = std::move(next);
    ++data_.historyRevision;
    ++data_.revision;
}

void HudDataModel::clearHistory() {
    const std::scoped_lock lock(mutex_);
    data_.history = std::make_shared<const std::vector<HistoryEntry>>();
    ++data_.historyRevision;
    ++data_.revision;
}

void HudDataModel::setSettings(SettingsPage page) {
    auto shared = std::make_shared<const SettingsPage>(std::move(page));
    const std::scoped_lock lock(mutex_);
    data_.settings = std::move(shared);
    ++data_.settingsRevision;
    ++data_.revision;
}

void HudDataModel::setAppearance(Appearance appearance) {
    const std::scoped_lock lock(mutex_);
    data_.appearance = appearance;
    ++data_.revision;
}

void HudDataModel::requestExpanded(bool expanded) {
    const std::scoped_lock lock(mutex_);
    data_.expandRequest = expanded;
    ++data_.expandRequestId;
    ++data_.revision;
}

HudData HudDataModel::snapshot() const {
    const std::scoped_lock lock(mutex_);
    return data_;
}

}  // namespace jarvis::overlay

// Design notes:
// - Appending copies the history once (at most maxEntries lines) instead of
//   sharing a mutable vector: lines arrive a few times per turn, frames
//   arrive 15-60 times per second, and the UI thread must never hold the
//   lock while it lays out hundreds of lines.
// - A limit of 0 lines is raised to 1 so the newest line always shows and
//   the "keep" arithmetic cannot underflow.
