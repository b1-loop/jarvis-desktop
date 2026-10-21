#include "jarvis/audio/playback.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace jarvis::audio {

namespace {

float rootMeanSquare(std::span<const float> samples) noexcept {
    if (samples.empty()) {
        return 0.0F;
    }
    const float sumOfSquares = std::transform_reduce(samples.begin(), samples.end(), 0.0F,
                                                     std::plus<>{}, [](float s) { return s * s; });
    return std::sqrt(sumOfSquares / static_cast<float>(samples.size()));
}

}  // namespace

Playback::Playback(std::unique_ptr<AudioSink> sink) : sink_(std::move(sink)) {
    sink_->start([this](std::span<float> out) { render(out); });
}

Playback::~Playback() {
    sink_->stop();
}

void Playback::enqueue(std::span<const float> samples) {
    if (samples.empty()) {
        return;
    }
    const std::scoped_lock lock(mutex_);
    chunks_.emplace_back(samples.begin(), samples.end());
    queued_.fetch_add(samples.size(), std::memory_order_release);
    endOfStream_.store(false, std::memory_order_release);
}

void Playback::stop() noexcept {
    const std::scoped_lock lock(mutex_);
    chunks_.clear();
    offset_ = 0;
    queued_.store(0, std::memory_order_release);
    endOfStream_.store(true, std::memory_order_release);
    stopCount_.fetch_add(1, std::memory_order_release);
}

std::uint64_t Playback::stopCount() const noexcept {
    return stopCount_.load(std::memory_order_acquire);
}

void Playback::markEndOfStream() noexcept {
    endOfStream_.store(true, std::memory_order_release);
}

bool Playback::isPlaying() const noexcept {
    return queued_.load(std::memory_order_acquire) > 0 ||
           !endOfStream_.load(std::memory_order_acquire);
}

std::size_t Playback::queuedSamples() const noexcept {
    return queued_.load(std::memory_order_acquire);
}

float Playback::outputLevel() const noexcept {
    return outputLevel_.load(std::memory_order_relaxed);
}

void Playback::render(std::span<float> out) noexcept {
    std::ranges::fill(out, 0.0F);
    const std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        return;  // one period of silence is inaudible; a stalled callback is not
    }
    const std::size_t written = fillFromQueue(out);
    outputLevel_.store(rootMeanSquare(out.first(written)), std::memory_order_relaxed);
}

std::size_t Playback::fillFromQueue(std::span<float> out) noexcept {
    std::size_t written = 0;
    while (written < out.size() && !chunks_.empty()) {
        const std::span<const float> chunk{chunks_.front()};
        const std::size_t count = std::min(out.size() - written, chunk.size() - offset_);
        std::ranges::copy(chunk.subspan(offset_, count), out.subspan(written).begin());
        written += count;
        offset_ += count;
        queued_.fetch_sub(count, std::memory_order_release);
        if (offset_ == chunk.size()) {
            chunks_.pop_front();
            offset_ = 0;
        }
    }
    return written;
}

}  // namespace jarvis::audio
