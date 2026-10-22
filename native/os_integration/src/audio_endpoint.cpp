#include "jarvis/os/audio_endpoint.hpp"

// Order matters: the Core Audio headers rely on <windows.h> types, and
// <initguid.h> must precede the property-key headers so that
// PKEY_AudioEndpoint_FormFactor and PKEY_Device_FriendlyName are defined in
// this translation unit (they are selectany, so other units may define them too).
// clang-format off
#include <windows.h>
#include <combaseapi.h>
#include <initguid.h>
#include <propkeydef.h>
#include <functiondiscoverykeys_devpkey.h>
#include <mmdeviceapi.h>
#include <propidl.h>
#include <propvarutil.h>
#include <wrl/client.h>
// clang-format on

#include "jarvis/os/errors.hpp"
#include "jarvis/os/text_encoding.hpp"

#include "win32_util.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace jarvis::os {

namespace {

using Microsoft::WRL::ComPtr;

constexpr std::array<std::string_view, 11> kFormFactorNames{
    "remote_network_device",
    "speakers",
    "line_level",
    "headphones",
    "microphone",
    "headset",
    "handset",
    "unknown_digital_passthrough",
    "spdif",
    "digital_audio_display_device",
    "unknown",
};

/// Frees a CoTaskMemAlloc'd wide string.
struct CoTaskMemDeleter {
    void operator()(wchar_t* text) const noexcept { CoTaskMemFree(text); }
};

/// Clears a PROPVARIANT on destruction.
class PropVariant {
public:
    PropVariant() noexcept { PropVariantInit(&value_); }
    ~PropVariant() { PropVariantClear(&value_); }
    PropVariant(const PropVariant&) = delete;
    PropVariant& operator=(const PropVariant&) = delete;
    PropVariant(PropVariant&&) = delete;
    PropVariant& operator=(PropVariant&&) = delete;

    [[nodiscard]] PROPVARIANT* get() noexcept { return &value_; }
    [[nodiscard]] const PROPVARIANT& value() const noexcept { return value_; }

private:
    PROPVARIANT value_{};
};

/// Property store of the default console render endpoint, or null.
ComPtr<IPropertyStore> defaultRenderProperties() {
    ComPtr<IMMDeviceEnumerator> enumerator;
    ComPtr<IMMDevice> device;
    ComPtr<IPropertyStore> properties;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                IID_PPV_ARGS(&enumerator))) ||
        FAILED(enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device)) ||
        FAILED(device->OpenPropertyStore(STGM_READ, &properties))) {
        return nullptr;
    }
    return properties;
}

std::string friendlyName(IPropertyStore& properties) {
    PropVariant name;
    wchar_t* raw = nullptr;
    if (FAILED(properties.GetValue(PKEY_Device_FriendlyName, name.get())) ||
        FAILED(PropVariantToStringAlloc(name.value(), &raw))) {
        return {};
    }
    const std::unique_ptr<wchar_t, CoTaskMemDeleter> text{raw};
    return wideToUtf8(text.get());
}

EndpointFormFactor formFactor(IPropertyStore& properties) {
    PropVariant value;
    ULONG raw = 0;
    if (FAILED(properties.GetValue(PKEY_AudioEndpoint_FormFactor, value.get())) ||
        FAILED(PropVariantToUInt32(value.value(), &raw))) {
        return EndpointFormFactor::Unknown;
    }
    return formFactorFromValue(raw);
}

}  // namespace

EndpointFormFactor formFactorFromValue(std::uint32_t value) noexcept {
    if (value >= static_cast<std::uint32_t>(EndpointFormFactor::Unknown)) {
        return EndpointFormFactor::Unknown;
    }
    return static_cast<EndpointFormFactor>(value);
}

std::string_view formFactorName(EndpointFormFactor formFactor) noexcept {
    const auto index = static_cast<std::size_t>(formFactor);
    return index < kFormFactorNames.size() ? kFormFactorNames.at(index) : kFormFactorNames.back();
}

std::optional<AudioEndpointInfo> defaultOutputEndpoint() {
    try {
        const win32::ComScope com{win32::Apartment::MultiThreaded};
        const ComPtr<IPropertyStore> properties = defaultRenderProperties();
        if (properties == nullptr) {
            return std::nullopt;  // no output device, or no audio service in this session
        }
        return AudioEndpointInfo{.name = friendlyName(*properties.Get()),
                                 .formFactor = formFactor(*properties.Get())};
    } catch (const OsError&) {
        return std::nullopt;  // COM unusable on this thread
    }
}

}  // namespace jarvis::os

// Design notes:
// - Read-only and failure-tolerant on purpose: the only consumer decides
//   whether voice barge-in is safe (headphones) or would hear Jarvis itself
//   (speakers), and "unknown" is a valid answer there; it picks the safe mode.
// - The form factor is what the driver declares for the jack or Bluetooth
//   profile, which is more reliable than guessing from the localised name;
//   the name is returned anyway for the log line and as a fallback hint.
