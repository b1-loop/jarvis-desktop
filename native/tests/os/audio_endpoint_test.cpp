#include "jarvis/os/audio_endpoint.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <optional>
#include <string_view>

using jarvis::os::AudioEndpointInfo;
using jarvis::os::defaultOutputEndpoint;
using jarvis::os::EndpointFormFactor;
using jarvis::os::formFactorFromValue;
using jarvis::os::formFactorName;

TEST_CASE("form factor values map onto the Core Audio enum", "[os][audio_endpoint]") {
    CHECK(formFactorFromValue(1) == EndpointFormFactor::Speakers);
    CHECK(formFactorFromValue(3) == EndpointFormFactor::Headphones);
    CHECK(formFactorFromValue(5) == EndpointFormFactor::Headset);
    CHECK(formFactorFromValue(10) == EndpointFormFactor::Unknown);
}

TEST_CASE("form factor values outside the enum are unknown", "[os][audio_endpoint]") {
    CHECK(formFactorFromValue(11) == EndpointFormFactor::Unknown);
    CHECK(formFactorFromValue(UINT32_MAX) == EndpointFormFactor::Unknown);
}

TEST_CASE("form factors have stable snake_case names", "[os][audio_endpoint]") {
    CHECK(formFactorName(EndpointFormFactor::Speakers) == std::string_view{"speakers"});
    CHECK(formFactorName(EndpointFormFactor::Headphones) == std::string_view{"headphones"});
    CHECK(formFactorName(EndpointFormFactor::Headset) == std::string_view{"headset"});
    CHECK(formFactorName(EndpointFormFactor::DigitalAudioDisplayDevice) ==
          std::string_view{"digital_audio_display_device"});
    CHECK(formFactorName(EndpointFormFactor::Unknown) == std::string_view{"unknown"});
}

TEST_CASE("the default output endpoint is read without changing anything", "[os][audio_endpoint]") {
    const std::optional<AudioEndpointInfo> endpoint = defaultOutputEndpoint();
    if (!endpoint.has_value()) {
        SKIP("no audio output device on this machine");
    }
    const AudioEndpointInfo first = endpoint.value_or(AudioEndpointInfo{});
    CHECK_FALSE(formFactorName(first.formFactor).empty());
    // Reading twice gives the same answer: the call has no side effects.
    const AudioEndpointInfo again = defaultOutputEndpoint().value_or(AudioEndpointInfo{});
    CHECK(again.formFactor == first.formFactor);
}
