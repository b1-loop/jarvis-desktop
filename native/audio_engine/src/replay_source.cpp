#include "jarvis/audio/replay_source.hpp"

#include "jarvis/audio/audio_format.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace jarvis::audio {

void ReplayFeed::push(std::span<const float> samples) {
    const std::scoped_lock lock(mutex_);
    queue_.insert(queue_.end(), samples.begin(), samples.end());
}

std::size_t ReplayFeed::take(std::span<float> out) {
    std::size_t audio = 0;
    {
        const std::scoped_lock lock(mutex_);
        audio = std::min(out.size(), queue_.size());
        const auto end = queue_.begin() + static_cast<std::ptrdiff_t>(audio);
        std::copy(queue_.begin(), end, out.begin());
        queue_.erase(queue_.begin(), end);
    }
    std::fill(out.begin() + static_cast<std::ptrdiff_t>(audio), out.end(), 0.0F);
    delivered_.fetch_add(out.size());
    return audio;
}

std::size_t ReplayFeed::pending() const {
    const std::scoped_lock lock(mutex_);
    return queue_.size();
}

ReplaySource::ReplaySource(std::shared_ptr<ReplayFeed> feed, std::chrono::milliseconds period)
    : feed_(std::move(feed)), period_(period) {}

ReplaySource::~ReplaySource() {
    stop();
}

void ReplaySource::start(FrameCallback onFrames, DeviceCallback /*onDevice*/) {
    // A replayed feed has no device that could stop, move or be interrupted.
    stop();
    thread_ = std::jthread([this, callback = std::move(onFrames)](const std::stop_token& token) {
        run(token, callback);
    });
}

void ReplaySource::stop() {
    if (thread_.joinable()) {
        thread_.request_stop();
        thread_.join();
    }
}

void ReplaySource::run(const std::stop_token& stopToken, const FrameCallback& onFrames) const {
    constexpr auto kMillisecondsPerSecond = 1000;
    const auto frameSamples = static_cast<std::size_t>(kSampleRate) *
                              static_cast<std::size_t>(period_.count()) / kMillisecondsPerSecond;
    std::vector<float> frame(frameSamples);
    // Deadlines advance by whole periods, so scheduling jitter never adds
    // up into drift: one minute of replay is one minute of engine time.
    auto deadline = std::chrono::steady_clock::now();
    while (!stopToken.stop_requested()) {
        feed_->take(frame);
        onFrames(frame);
        deadline += period_;
        std::this_thread::sleep_until(deadline);
    }
}

}  // namespace jarvis::audio
