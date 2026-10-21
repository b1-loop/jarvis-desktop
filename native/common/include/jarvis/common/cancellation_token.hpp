/// @file cancellation_token.hpp
/// @brief A thread-safe "please stop" flag shared between a caller and a worker.

#pragma once

#include <atomic>

namespace jarvis {

/// One-way flag a caller raises to ask long-running native work to stop.
///
/// The worker polls cancelled() between units of work (a generated token, a
/// decoded batch) and returns early; the caller never blocks on it. It is
/// deliberately not std::stop_token: that is tied to std::jthread ownership,
/// whereas a Jarvis cancellation may arrive from Python on any thread.
class CancellationToken {
public:
    CancellationToken() = default;
    CancellationToken(const CancellationToken&) = delete;
    CancellationToken& operator=(const CancellationToken&) = delete;
    CancellationToken(CancellationToken&&) = delete;
    CancellationToken& operator=(CancellationToken&&) = delete;
    ~CancellationToken() = default;

    /// Request cancellation. Idempotent and safe from any thread.
    void cancel() noexcept { cancelled_.store(true, std::memory_order_release); }

    /// Whether cancel() has been called.
    [[nodiscard]] bool cancelled() const noexcept {
        return cancelled_.load(std::memory_order_acquire);
    }

    /// Lower the flag so the token can be reused for the next operation.
    void reset() noexcept { cancelled_.store(false, std::memory_order_release); }

private:
    std::atomic<bool> cancelled_{false};
};

}  // namespace jarvis
