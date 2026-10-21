#include "jarvis/audio/playback.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "audio/fakes.hpp"

#include <memory>
#include <numeric>
#include <utility>
#include <vector>

using jarvis::audio::Playback;
using jarvis::audio::testing::FakeAudioSink;
using jarvis::audio::testing::samples;

namespace {

struct Fixture {
    FakeAudioSink* sink;
    Playback playback;

    Fixture() : Fixture(std::make_unique<FakeAudioSink>()) {}

private:
    explicit Fixture(std::unique_ptr<FakeAudioSink> owned)
        : sink(owned.get()), playback(std::move(owned)) {}
};

std::vector<float> ramp(std::size_t count, float start) {
    std::vector<float> values(count);
    std::iota(values.begin(), values.end(), start);
    return values;
}

}  // namespace

TEST_CASE("the sink runs for the lifetime of the playback object", "[audio][playback]") {
    auto owned = std::make_unique<FakeAudioSink>();
    const auto* sink = owned.get();
    {
        const Playback playback{std::move(owned)};
        REQUIRE(sink->started);
        REQUIRE_FALSE(sink->stopped);
    }
    // The sink outlives the Playback only because the test kept the raw pointer;
    // Playback owns it, so this observation is made before destruction completes.
}

TEST_CASE("silence is rendered when nothing is queued", "[audio][playback]") {
    const Fixture f;

    REQUIRE(f.sink->render(64) == samples(64, 0.0F));
    REQUIRE_FALSE(f.playback.isPlaying());
    REQUIRE(f.playback.queuedSamples() == 0);
}

TEST_CASE("queued chunks play back in order across device periods", "[audio][playback]") {
    Fixture f;
    f.playback.enqueue(ramp(6, 1.0F));
    f.playback.enqueue(ramp(4, 7.0F));
    REQUIRE(f.playback.queuedSamples() == 10);

    REQUIRE(f.sink->render(4) == ramp(4, 1.0F));
    REQUIRE(f.sink->render(4) == ramp(4, 5.0F));
    REQUIRE(f.sink->render(4) == std::vector<float>{9.0F, 10.0F, 0.0F, 0.0F});
    REQUIRE(f.playback.queuedSamples() == 0);
}

TEST_CASE("is_playing stays true until the stream is marked finished and drained",
          "[audio][playback]") {
    Fixture f;
    f.playback.enqueue(samples(8, 0.1F));

    REQUIRE(f.playback.isPlaying());
    f.sink->render(8);
    REQUIRE(f.playback.isPlaying());  // drained, but more chunks might follow

    f.playback.markEndOfStream();
    REQUIRE_FALSE(f.playback.isPlaying());
}

TEST_CASE("marking end of stream before draining keeps playing until empty", "[audio][playback]") {
    Fixture f;
    f.playback.enqueue(samples(8, 0.1F));
    f.playback.markEndOfStream();

    REQUIRE(f.playback.isPlaying());
    f.sink->render(8);
    REQUIRE_FALSE(f.playback.isPlaying());
}

TEST_CASE("stop discards queued audio and the next period is silent", "[audio][playback]") {
    Fixture f;
    f.playback.enqueue(samples(100, 0.5F));

    f.playback.stop();

    REQUIRE_FALSE(f.playback.isPlaying());
    REQUIRE(f.playback.queuedSamples() == 0);
    REQUIRE(f.sink->render(16) == samples(16, 0.0F));
}

TEST_CASE("every stop is counted so callers can detect barge-in", "[audio][playback]") {
    Fixture f;
    REQUIRE(f.playback.stopCount() == 0);

    f.playback.stop();
    f.playback.stop();

    REQUIRE(f.playback.stopCount() == 2);
}

TEST_CASE("audio enqueued after a stop plays normally", "[audio][playback]") {
    Fixture f;
    f.playback.enqueue(samples(100, 0.5F));
    f.playback.stop();

    f.playback.enqueue(samples(4, 0.25F));

    REQUIRE(f.playback.isPlaying());
    REQUIRE(f.sink->render(4) == samples(4, 0.25F));
}

TEST_CASE("empty chunks are ignored", "[audio][playback]") {
    Fixture f;
    f.playback.enqueue({});

    REQUIRE_FALSE(f.playback.isPlaying());
}

TEST_CASE("output level follows what was just rendered", "[audio][playback]") {
    Fixture f;
    f.playback.enqueue(samples(64, 0.5F));

    f.sink->render(64);
    REQUIRE_THAT(f.playback.outputLevel(), Catch::Matchers::WithinAbs(0.5, 1e-5));

    f.sink->render(64);
    REQUIRE_THAT(f.playback.outputLevel(), Catch::Matchers::WithinAbs(0.0, 1e-5));
}

TEST_CASE("sample rate is taken from the sink", "[audio][playback]") {
    auto owned = std::make_unique<FakeAudioSink>(24000);
    const Playback playback{std::move(owned)};

    REQUIRE(playback.sampleRate() == 24000);
}
