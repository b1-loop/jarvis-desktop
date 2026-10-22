#include "jarvis/overlay/system_info.hpp"

// Order matters: the Core Audio headers rely on <windows.h> types, and
// <initguid.h> must precede the property-key header so that
// PKEY_Device_FriendlyName is defined in this translation unit.
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

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace jarvis::overlay {

namespace {

using Microsoft::WRL::ComPtr;

/// Balances a successful CoInitializeEx on this thread; a thread that was
/// already initialised in another apartment (RPC_E_CHANGED_MODE) can still
/// use COM and must not be uninitialised by us.
class ComScope {
public:
    ComScope() noexcept : result_(CoInitializeEx(nullptr, COINIT_MULTITHREADED)) {}
    ~ComScope() {
        if (SUCCEEDED(result_)) {
            CoUninitialize();
        }
    }
    ComScope(const ComScope&) = delete;
    ComScope& operator=(const ComScope&) = delete;
    ComScope(ComScope&&) = delete;
    ComScope& operator=(ComScope&&) = delete;

    [[nodiscard]] bool usable() const noexcept {
        return SUCCEEDED(result_) || result_ == RPC_E_CHANGED_MODE;
    }

private:
    HRESULT result_;
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

std::string toUtf8(std::wstring_view wide) {
    if (wide.empty()) {
        return {};
    }
    const std::wstring owned{wide};
    const int size = static_cast<int>(owned.size());
    const int needed =
        WideCharToMultiByte(CP_UTF8, 0, owned.c_str(), size, nullptr, 0, nullptr, nullptr);
    if (needed <= 0) {
        return {};
    }
    std::string utf8(static_cast<std::size_t>(needed), '\0');
    WideCharToMultiByte(CP_UTF8, 0, owned.c_str(), size, utf8.data(), needed, nullptr, nullptr);
    return utf8;
}

}  // namespace

std::optional<std::string> defaultMicrophoneName() {
    const ComScope com;
    if (!com.usable()) {
        return std::nullopt;
    }
    ComPtr<IMMDeviceEnumerator> enumerator;
    ComPtr<IMMDevice> device;
    ComPtr<IPropertyStore> properties;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                IID_PPV_ARGS(&enumerator))) ||
        FAILED(enumerator->GetDefaultAudioEndpoint(eCapture, eConsole, &device)) ||
        FAILED(device->OpenPropertyStore(STGM_READ, &properties))) {
        return std::nullopt;  // no capture device, or no audio service in this session
    }
    PropVariant name;
    wchar_t* raw = nullptr;
    if (FAILED(properties->GetValue(PKEY_Device_FriendlyName, name.get())) ||
        FAILED(PropVariantToStringAlloc(name.value(), &raw))) {
        return std::nullopt;
    }
    const std::unique_ptr<wchar_t, CoTaskMemDeleter> text{raw};
    std::string utf8 = toUtf8(text.get());
    if (utf8.empty()) {
        return std::nullopt;
    }
    return utf8;
}

}  // namespace jarvis::overlay

// Design notes:
// - Lives with the overlay rather than the audio engine because the only
//   consumer is the boot readout, and the engine itself opens the default
//   device through miniaudio without ever needing its name.
// - Every failure is "no name" rather than an exception: the readout is
//   cosmetic, and the caller shows "system default" instead, which is
//   exactly the device the engine uses.
