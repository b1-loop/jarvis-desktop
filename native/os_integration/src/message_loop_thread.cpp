#include "message_loop_thread.hpp"

#include "jarvis/os/errors.hpp"

#include <windows.h>

#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace jarvis::os::win32 {

namespace {

/// Private thread message that says "tasks are waiting". WM_APP is reserved
/// for exactly this kind of application-defined traffic.
constexpr UINT kRunTasksMessage = WM_APP + 1;

}  // namespace

struct MessageLoopThread::Queue {
    std::mutex mutex;
    std::deque<std::function<void()>> tasks;
};

MessageLoopThread::MessageLoopThread(ThreadMessageHandler onThreadMessage, ExitHook onExit)
    : onThreadMessage_(std::move(onThreadMessage)),
      onExit_(std::move(onExit)),
      queue_(std::make_unique<Queue>()) {
    std::promise<void> ready;
    std::future<void> started = ready.get_future();
    thread_ = std::thread([this, &ready] { loop(ready); });
    started.get();  // rethrows if the loop failed to come up
}

MessageLoopThread::~MessageLoopThread() {
    PostThreadMessageW(threadId_, WM_QUIT, 0, 0);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void MessageLoopThread::post(std::function<void()> task) {
    {
        const std::scoped_lock lock(queue_->mutex);
        queue_->tasks.push_back(std::move(task));
    }
    if (PostThreadMessageW(threadId_, kRunTasksMessage, 0, 0) == FALSE) {
        throw OsError("the message loop thread is gone; cannot run the task");
    }
}

void MessageLoopThread::loop(std::promise<void>& ready) {
    MSG message{};
    // A thread has no message queue until it first asks for one; PeekMessage
    // creates it so that PostThreadMessage from other threads succeeds.
    PeekMessageW(&message, nullptr, WM_USER, WM_USER, PM_NOREMOVE);
    threadId_ = GetCurrentThreadId();
    ready.set_value();

    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        if (message.message == kRunTasksMessage) {
            drainTasks();
        } else if (message.hwnd == nullptr) {
            if (onThreadMessage_) {
                onThreadMessage_(message);
            }
        } else {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
    if (onExit_) {
        onExit_();
    }
}

void MessageLoopThread::drainTasks() {
    std::deque<std::function<void()>> pending;
    {
        const std::scoped_lock lock(queue_->mutex);
        pending.swap(queue_->tasks);
    }
    for (const auto& task : pending) {
        task();  // a packaged_task stores its own exception; nothing escapes
    }
}

}  // namespace jarvis::os::win32
