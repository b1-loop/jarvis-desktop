#include "jarvis/audio/ring_buffer.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <numeric>
#include <vector>

using jarvis::audio::SpscRingBuffer;

namespace {

std::vector<float> ramp(std::size_t count, float start = 0.0F) {
    std::vector<float> values(count);
    std::iota(values.begin(), values.end(), start);
    return values;
}

}  // namespace

TEST_CASE("capacity is rounded up to a power of two", "[audio][ring_buffer]") {
    REQUIRE(SpscRingBuffer{100}.capacity() == 128);
    REQUIRE(SpscRingBuffer{128}.capacity() == 128);
    REQUIRE(SpscRingBuffer{1}.capacity() == 2);
}

TEST_CASE("samples come out in the order they went in", "[audio][ring_buffer]") {
    SpscRingBuffer buffer{16};
    const auto input = ramp(10);

    REQUIRE(buffer.push(input) == 10);
    REQUIRE(buffer.size() == 10);

    std::vector<float> output(10);
    REQUIRE(buffer.pop(output) == 10);
    REQUIRE(output == input);
    REQUIRE(buffer.size() == 0);
}

TEST_CASE("reads and writes wrap around the end of storage", "[audio][ring_buffer]") {
    SpscRingBuffer buffer{8};
    std::vector<float> scratch(6);

    REQUIRE(buffer.push(ramp(6)) == 6);
    REQUIRE(buffer.pop(scratch) == 6);  // tail now at 6, so the next push wraps
    const auto wrapped = ramp(6, 100.0F);
    REQUIRE(buffer.push(wrapped) == 6);

    REQUIRE(buffer.pop(scratch) == 6);
    REQUIRE(scratch == wrapped);
}

TEST_CASE("overflow drops the excess and counts it", "[audio][ring_buffer]") {
    SpscRingBuffer buffer{8};

    REQUIRE(buffer.push(ramp(12)) == 8);
    REQUIRE(buffer.droppedSamples() == 4);
    REQUIRE(buffer.size() == 8);

    std::vector<float> output(8);
    REQUIRE(buffer.pop(output) == 8);
    REQUIRE(output == ramp(8));
}

TEST_CASE("popping from an empty buffer reads nothing", "[audio][ring_buffer]") {
    SpscRingBuffer buffer{8};
    std::vector<float> output(4, 42.0F);

    REQUIRE(buffer.pop(output) == 0);
    REQUIRE(output == std::vector<float>(4, 42.0F));
}

TEST_CASE("partial pops leave the remainder", "[audio][ring_buffer]") {
    SpscRingBuffer buffer{16};
    REQUIRE(buffer.push(ramp(10)) == 10);

    std::vector<float> first(4);
    std::vector<float> rest(10);
    REQUIRE(buffer.pop(first) == 4);
    REQUIRE(buffer.pop(rest) == 6);

    REQUIRE(first == ramp(4));
    rest.resize(6);
    REQUIRE(rest == ramp(6, 4.0F));
}
