#include "jarvis/os/hotkey.hpp"

#include "message_loop_thread.hpp"
#include "win32_util.hpp"
#include <windows.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <ranges>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jarvis::os {

struct HotkeyManager::Impl {
    mutable std::mutex mutex;  // guards registered, events, nextId
    std::unordered_map<int, Hotkey> registered;
    std::vector<HotkeyEvent> events;
    int nextId{1};
    // Declared last so it is destroyed first: the loop must stop before
    // the state it reads goes away.
    win32::MessageLoopThread loop;

    Impl()
        : loop([this](const MSG& message) { onThreadMessage(message); },
               [this] { unregisterAll(); }) {}

    void onThreadMessage(const MSG& message) {
        if (message.message != WM_HOTKEY) {
            return;
        }
        const int id = static_cast<int>(message.wParam);
        const std::scoped_lock lock(mutex);
        const auto found = registered.find(id);
        if (found != registered.end()) {
            events.push_back(HotkeyEvent{.id = id, .hotkey = found->second});
        }
    }

    /// Runs on the loop thread.
    static void registerOnLoop(int id, const Hotkey& hotkey) {
        // MOD_NOREPEAT: holding the keys down fires once, which is what a
        // push-to-talk trigger wants.
        constexpr auto kNoRepeat = static_cast<UINT>(MOD_NOREPEAT);
        if (RegisterHotKey(nullptr, id, hotkey.modifiers | kNoRepeat, hotkey.virtualKey) == FALSE) {
            win32::throwLastError("RegisterHotKey(" + formatHotkey(hotkey) + ")");
        }
    }

    /// Runs on the loop thread as it exits. Hotkeys die with their thread
    /// anyway; releasing them explicitly keeps the ordering deterministic
    /// for anyone re-registering right after.
    void unregisterAll() const {
        const std::scoped_lock lock(mutex);
        for (const int id : registered | std::views::keys) {
            UnregisterHotKey(nullptr, id);
        }
    }
};

HotkeyManager::HotkeyManager() : impl_(std::make_unique<Impl>()) {}

HotkeyManager::~HotkeyManager() = default;

int HotkeyManager::registerHotkey(const Hotkey& hotkey) {
    int id = 0;
    {
        const std::scoped_lock lock(impl_->mutex);
        id = impl_->nextId++;
    }
    impl_->loop.run([id, hotkey] { Impl::registerOnLoop(id, hotkey); });
    const std::scoped_lock lock(impl_->mutex);
    impl_->registered.emplace(id, hotkey);
    return id;
}

void HotkeyManager::unregisterHotkey(int id) {
    {
        const std::scoped_lock lock(impl_->mutex);
        if (impl_->registered.erase(id) == 0) {
            return;
        }
    }
    impl_->loop.run([id] { UnregisterHotKey(nullptr, id); });
}

std::vector<HotkeyEvent> HotkeyManager::pollEvents() {
    const std::scoped_lock lock(impl_->mutex);
    return std::exchange(impl_->events, {});
}

std::size_t HotkeyManager::registeredCount() const {
    const std::scoped_lock lock(impl_->mutex);
    return impl_->registered.size();
}

}  // namespace jarvis::os

// Design notes:
// - One MessageLoopThread per manager rather than a process-wide singleton:
//   the Python side creates exactly one manager, and the destructor then
//   has a well-defined moment to stop pumping.
// - Registration happens on the loop thread via run() so RegisterHotKey's
//   thread affinity is satisfied and its failure (typically "already
//   registered" by another application) propagates to the caller as OsError.
// - Cleanup is the loop's exit hook, not a destructor marshalling a task:
//   destructors must not throw, and run() can.
