/// @file audio_endpoint.hpp
/// @brief What kind of device the default audio output is: speakers or headphones.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace jarvis::os {

/// Mirror of Core Audio's @c EndpointFormFactor, in the same numeric order,
/// so a value read from @c PKEY_AudioEndpoint_FormFactor converts directly.
enum class EndpointFormFactor : std::uint8_t {
    RemoteNetworkDevice = 0,
    Speakers = 1,
    LineLevel = 2,
    Headphones = 3,
    Microphone = 4,
    Headset = 5,
    Handset = 6,
    UnknownDigitalPassthrough = 7,
    Spdif = 8,
    DigitalAudioDisplayDevice = 9,
    Unknown = 10,
};

/// The default render endpoint as the user sees it in the Sound settings.
struct AudioEndpointInfo {
    std::string name;  ///< Friendly name, UTF-8 ("Speakers (Realtek(R) Audio)").
    EndpointFormFactor formFactor{EndpointFormFactor::Unknown};
};

/// @p value as a form factor; anything outside the enum is @c Unknown.
[[nodiscard]] EndpointFormFactor formFactorFromValue(std::uint32_t value) noexcept;

/// Lower-case snake_case name of @p formFactor ("headphones", "speakers").
[[nodiscard]] std::string_view formFactorName(EndpointFormFactor formFactor) noexcept;

/// The default console output device, or @c std::nullopt when there is none
/// (no audio service in this session, no output device) or its properties
/// cannot be read. Read-only; failures are reported as nullopt, not OsError.
[[nodiscard]] std::optional<AudioEndpointInfo> defaultOutputEndpoint();

}  // namespace jarvis::os
