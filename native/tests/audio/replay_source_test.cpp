#include "jarvis/audio/replay_source.hpp"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

using jarvis::audio::ReplayFeed;
using jarvis::audio::ReplaySource;

namespace {

constexpr auto kTimeout = std::chrono::seconds(5);
constexpr auto kPeriod = std::chrono::milliseconds(10);
constexpr std::size_t kFrameSamples = 160;  // 10 ms at 16 kHz

bool waitUntil(const std::function<bool()>& condition) {
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (condition()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return condition();
}

}  // namespace

TEST_CASE("ReplayFeed hands out queued audio first, then silence", "[audio][replay]") {
    ReplayFeed feed;
    const std::vector<float> audio{0.5F, -0.5F, 0.25F};
    feed.push(audio);
    std::vector<float> out(5, 9.0F);

    REQUIRE(feed.take(out) == 3);
    CHECK(out == std::vector<float>{0.5F, -0.5F, 0.25F, 0.0F, 0.0F});
    CHECK(feed.pending() == 0);
    CHECK(feed.delivered() == 5);
}

TEST_CASE("ReplayFeed keeps the remainder for the next frame", "[audio][replay]") {
    ReplayFeed feed;
    const std::vector<float> audio{1.0F, 2.0F, 3.0F};
    feed.push(audio);
    std::vector<float> out(2);

    REQUIRE(feed.take(out) == 2);
    CHECK(feed.pending() == 1);
    REQUIRE(feed.take(out) == 1);
    CHECK(out == std::vector<float>{3.0F, 0.0F});
}

TEST_CASE("ReplaySource delivers fixed-size frames until stopped", "[audio][replay]") {
    const auto feed = std::make_shared<ReplayFeed>();
    const std::vector<float> audio(kFrameSamples * 3, 0.1F);
    feed->push(audio);
    ReplaySource source(feed, kPeriod);
    std::mutex mutex;
    std::vector<std::size_t> sizes;
    std::atomic<std::size_t> frames{0};

    source.start([&](std::span<const float> frame) {
        const std::scoped_lock lock(mutex);
        sizes.push_back(frame.size());
        ++frames;
    });
    REQUIRE(waitUntil([&] { return frames.load() >= 4; }));
    source.stop();
    const std::size_t afterStop = frames.load();
    std::this_thread::sleep_for(kPeriod * 3);

    CHECK(frames.load() == afterStop);
    CHECK(feed->pending() == 0);
    const std::scoped_lock lock(mutex);
    for (const std::size_t size : sizes) {
        CHECK(size == kFrameSamples);
    }
}
