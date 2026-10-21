/// @file ring_buffer.hpp
/// @brief Lock-free single-producer / single-consumer sample buffer.

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <iterator>
#include <span>
#include <vector>

namespace jarvis::audio {

/// Lock-free SPSC ring buffer of float samples.
///
/// The producer is the audio device callback, which must never block or
/// allocate; the consumer is the engine worker thread. When the consumer
/// falls behind, incoming samples are dropped rather than stalling the
/// callback, and every dropped sample is counted so the engine can report
/// it instead of silently losing audio.
class SpscRingBuffer {
public:
    /// @param capacity Minimum number of samples to hold; rounded up to a power of two.
    explicit SpscRingBuffer(std::size_t capacity)
        : buffer_(std::bit_ceil(std::max<std::size_t>(capacity, kMinimumCapacity))),
          mask_(buffer_.size() - 1) {}

    /// Producer side. Copies as many samples as fit and drops the rest.
    /// @return Number of samples actually written.
    std::size_t push(std::span<const float> samples) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        const std::size_t free = buffer_.size() - (head - tail);
        const std::size_t count = std::min(free, samples.size());
        copyIn(head, samples.first(count));
        head_.store(head + count, std::memory_order_release);
        if (count < samples.size()) {
            dropped_.fetch_add(samples.size() - count, std::memory_order_relaxed);
        }
        return count;
    }

    /// Consumer side. Reads up to @p out.size() samples.
    /// @return Number of samples actually read.
    std::size_t pop(std::span<float> out) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t count = std::min(head - tail, out.size());
        copyOut(tail, out.first(count));
        tail_.store(tail + count, std::memory_order_release);
        return count;
    }

    /// Samples currently buffered. Approximate while both threads are active.
    [[nodiscard]] std::size_t size() const noexcept {
        return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire);
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return buffer_.size(); }

    /// Total samples dropped because the buffer was full.
    [[nodiscard]] std::size_t droppedSamples() const noexcept {
        return dropped_.load(std::memory_order_relaxed);
    }

private:
    static constexpr std::size_t kMinimumCapacity = 2;
    static constexpr std::size_t kCacheLineBytes = 64;

    void copyIn(std::size_t head, std::span<const float> samples) noexcept {
        const std::size_t start = head & mask_;
        const std::size_t firstPart = std::min(samples.size(), buffer_.size() - start);
        std::ranges::copy(samples.first(firstPart),
                          std::next(buffer_.begin(), static_cast<std::ptrdiff_t>(start)));
        std::ranges::copy(samples.subspan(firstPart), buffer_.begin());
    }

    void copyOut(std::size_t tail, std::span<float> out) const noexcept {
        const std::size_t start = tail & mask_;
        const std::size_t firstPart = std::min(out.size(), buffer_.size() - start);
        const std::span<const float> storage{buffer_};
        std::ranges::copy(storage.subspan(start, firstPart), out.begin());
        std::ranges::copy(storage.first(out.size() - firstPart),
                          std::next(out.begin(), static_cast<std::ptrdiff_t>(firstPart)));
    }

    std::vector<float> buffer_;
    std::size_t mask_;
    // Indices grow monotonically and are masked on access; keeping them on
    // separate cache lines stops the two threads from false-sharing.
    alignas(kCacheLineBytes) std::atomic<std::size_t> head_{0};
    alignas(kCacheLineBytes) std::atomic<std::size_t> tail_{0};
    alignas(kCacheLineBytes) std::atomic<std::size_t> dropped_{0};
};

}  // namespace jarvis::audio
