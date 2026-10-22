/// @file message_loop_thread.hpp
/// @brief Private helper: a thread that pumps a Win32 message queue.
///
/// Several Win32 facilities are thread-affine: RegisterHotKey posts to the
/// registering thread, and a window is destroyed when its creating thread
/// exits. Python threads (asyncio executors) come and go, so anything with
/// that lifetime rule lives on one of these dedicated threads instead.

#pragma once

#include <windows.h>

#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>

namespace jarvis::os::win32 {

/// A thread running GetMessage/DispatchMessage until destroyed.
///
/// run() executes a callable on that thread and returns its result (or
/// rethrows its exception) to the caller; onThreadMessage receives the
/// messages that are posted to the thread itself rather than to a window;
/// onExit runs on the loop thread after WM_QUIT, so owners can release
/// thread-affine resources without marshalling from a destructor.
class MessageLoopThread {
public:
    using ThreadMessageHandler = std::function<void(const MSG&)>;
    using ExitHook = std::function<void()>;

    /// Start the thread and wait until its message queue exists, so a
    /// PostThreadMessage issued right after construction cannot be lost.
    explicit MessageLoopThread(ThreadMessageHandler onThreadMessage, ExitHook onExit = {});

    /// Post WM_QUIT, let onExit run on the loop thread, then join.
    /// Tasks still queued are not run.
    ~MessageLoopThread();

    MessageLoopThread(const MessageLoopThread&) = delete;
    MessageLoopThread& operator=(const MessageLoopThread&) = delete;
    MessageLoopThread(MessageLoopThread&&) = delete;
    MessageLoopThread& operator=(MessageLoopThread&&) = delete;

    /// Run @p task on the loop thread and wait for it. Must not be called
    /// from the loop thread itself (it would deadlock).
    template <typename Task>
    std::invoke_result_t<Task&> run(Task&& task) {
        using Result = std::invoke_result_t<Task&>;
        // shared_ptr because std::function needs a copyable callable and
        // packaged_task is move-only.
        const auto packaged =
            std::make_shared<std::packaged_task<Result()>>(std::forward<Task>(task));
        std::future<Result> future = packaged->get_future();
        post([packaged] { (*packaged)(); });
        return future.get();
    }

    [[nodiscard]] DWORD threadId() const noexcept { return threadId_; }

private:
    void post(std::function<void()> task);
    void loop(std::promise<void>& ready);
    void drainTasks();

    ThreadMessageHandler onThreadMessage_;
    ExitHook onExit_;
    DWORD threadId_{0};
    struct Queue;
    std::unique_ptr<Queue> queue_;
    std::thread thread_;
};

}  // namespace jarvis::os::win32
