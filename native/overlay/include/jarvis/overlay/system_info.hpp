/// @file system_info.hpp
/// @brief Real values for the boot sequence's diagnostic readout.

#pragma once

#include <optional>
#include <string>

namespace jarvis::overlay {

/// Friendly name of the default capture endpoint (the microphone the
/// audio engine opens, since miniaudio uses the default device), e.g.
/// "Microphone (Realtek(R) Audio)"; std::nullopt when there is none or
/// Core Audio cannot be asked. Takes a few milliseconds; call it once.
[[nodiscard]] std::optional<std::string> defaultMicrophoneName();

}  // namespace jarvis::overlay
