#include "jarvis/os/system_controls.hpp"

#include "jarvis/os/errors.hpp"

#include "win32_util.hpp"
#include <comdef.h>
#include <endpointvolume.h>
#include <mmdeviceapi.h>
#include <wbemidl.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace jarvis::os {

namespace {

using Microsoft::WRL::ComPtr;

constexpr int kMaxPercent = 100;
constexpr float kMaxPercentScale = 100.0F;

int clampPercent(int percent) {
    return std::clamp(percent, 0, kMaxPercent);
}

// --- Core Audio ------------------------------------------------------------

/// The master volume control of the default playback device.
ComPtr<IAudioEndpointVolume> endpointVolume() {
    ComPtr<IMMDeviceEnumerator> enumerator;
    win32::checkHresult("CoCreateInstance(MMDeviceEnumerator)",
                        CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                         IID_PPV_ARGS(&enumerator)));
    ComPtr<IMMDevice> device;
    const HRESULT found = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
    if (FAILED(found)) {
        throw OsError("no default audio output device: " +
                      win32::errorMessage(static_cast<DWORD>(found)));
    }
    ComPtr<IAudioEndpointVolume> volume;
    win32::checkHresult(
        "IMMDevice::Activate(IAudioEndpointVolume)",
        device->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_ALL, nullptr, &volume));
    return volume;
}

// --- WMI brightness --------------------------------------------------------

/// Connected ROOT\WMI namespace with the proxy security WMI requires.
ComPtr<IWbemServices> connectWmi() {
    ComPtr<IWbemLocator> locator;
    win32::checkHresult(
        "CoCreateInstance(WbemLocator)",
        CoCreateInstance(CLSID_WbemLocator, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&locator)));
    ComPtr<IWbemServices> services;
    win32::checkHresult("IWbemLocator::ConnectServer(ROOT\\WMI)",
                        locator->ConnectServer(_bstr_t(L"ROOT\\WMI"), nullptr, nullptr, nullptr, 0,
                                               nullptr, nullptr, &services));
    win32::checkHresult(
        "CoSetProxyBlanket",
        CoSetProxyBlanket(services.Get(), RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, nullptr,
                          RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE));
    return services;
}

/// First instance returned by @p query, or UnsupportedError when there is
/// none: no active monitor exposes the WMI brightness classes. A query that
/// WMI itself rejects (class absent on this edition) is the same condition.
ComPtr<IWbemClassObject> firstInstance(IWbemServices& services, const wchar_t* query) {
    ComPtr<IEnumWbemClassObject> enumerator;
    const HRESULT queried = services.ExecQuery(_bstr_t(L"WQL"), _bstr_t(query),
                                               WBEM_FLAG_FORWARD_ONLY, nullptr, &enumerator);
    ComPtr<IWbemClassObject> instance;
    ULONG returned = 0;
    const HRESULT advanced =
        FAILED(queried) ? queried : enumerator->Next(WBEM_INFINITE, 1, &instance, &returned);
    if (FAILED(advanced) || returned == 0) {
        throw UnsupportedError(
            "no display exposes software brightness control (built-in laptop panels "
            "only; external monitors are not supported)");
    }
    return instance;
}

_variant_t property(IWbemClassObject& object, const wchar_t* name) {
    _variant_t value;
    win32::checkHresult("IWbemClassObject::Get", object.Get(name, 0, &value, nullptr, nullptr));
    return value;
}

/// Build the WmiSetBrightness(Timeout, Brightness) input object.
ComPtr<IWbemClassObject> brightnessArguments(IWbemServices& services, int percent) {
    ComPtr<IWbemClassObject> methodsClass;
    win32::checkHresult("IWbemServices::GetObject(WmiMonitorBrightnessMethods)",
                        services.GetObject(_bstr_t(L"WmiMonitorBrightnessMethods"), 0, nullptr,
                                           &methodsClass, nullptr));
    ComPtr<IWbemClassObject> signature;
    win32::checkHresult("IWbemClassObject::GetMethod(WmiSetBrightness)",
                        methodsClass->GetMethod(L"WmiSetBrightness", 0, &signature, nullptr));
    ComPtr<IWbemClassObject> arguments;
    win32::checkHresult("IWbemClassObject::SpawnInstance", signature->SpawnInstance(0, &arguments));
    _variant_t timeout(static_cast<long>(0));
    _variant_t brightness(static_cast<BYTE>(percent));
    win32::checkHresult("Put(Timeout)", arguments->Put(L"Timeout", 0, &timeout, 0));
    win32::checkHresult("Put(Brightness)", arguments->Put(L"Brightness", 0, &brightness, 0));
    return arguments;
}

}  // namespace

int Win32SystemControls::volume() const {
    const win32::ComScope com{win32::Apartment::MultiThreaded};
    float scalar = 0.0F;
    win32::checkHresult("GetMasterVolumeLevelScalar",
                        endpointVolume()->GetMasterVolumeLevelScalar(&scalar));
    return clampPercent(static_cast<int>(std::lround(scalar * kMaxPercentScale)));
}

void Win32SystemControls::setVolume(int percent) {
    const win32::ComScope com{win32::Apartment::MultiThreaded};
    const float scalar = static_cast<float>(clampPercent(percent)) / kMaxPercentScale;
    win32::checkHresult("SetMasterVolumeLevelScalar",
                        endpointVolume()->SetMasterVolumeLevelScalar(scalar, nullptr));
}

bool Win32SystemControls::muted() const {
    const win32::ComScope com{win32::Apartment::MultiThreaded};
    BOOL muted = FALSE;
    win32::checkHresult("GetMute", endpointVolume()->GetMute(&muted));
    return muted != FALSE;
}

void Win32SystemControls::setMuted(bool muted) {
    const win32::ComScope com{win32::Apartment::MultiThreaded};
    win32::checkHresult("SetMute", endpointVolume()->SetMute(muted ? TRUE : FALSE, nullptr));
}

int Win32SystemControls::brightness() const {
    const win32::ComScope com{win32::Apartment::MultiThreaded};
    const ComPtr<IWbemServices> services = connectWmi();
    const ComPtr<IWbemClassObject> instance = firstInstance(
        *services.Get(), L"SELECT CurrentBrightness FROM WmiMonitorBrightness WHERE Active=TRUE");
    const _variant_t current = property(*instance.Get(), L"CurrentBrightness");
    return clampPercent(static_cast<BYTE>(current));
}

void Win32SystemControls::setBrightness(int percent) {
    const win32::ComScope com{win32::Apartment::MultiThreaded};
    const ComPtr<IWbemServices> services = connectWmi();
    const ComPtr<IWbemClassObject> instance = firstInstance(
        *services.Get(), L"SELECT * FROM WmiMonitorBrightnessMethods WHERE Active=TRUE");
    const _variant_t path = property(*instance.Get(), L"__PATH");
    const ComPtr<IWbemClassObject> arguments =
        brightnessArguments(*services.Get(), clampPercent(percent));
    win32::checkHresult("IWbemServices::ExecMethod(WmiSetBrightness)",
                        services->ExecMethod(_bstr_t(path), _bstr_t(L"WmiSetBrightness"), 0,
                                             nullptr, arguments.Get(), nullptr, nullptr));
}

void Win32SystemControls::lockScreen() {
    if (LockWorkStation() == FALSE) {
        win32::throwLastError("LockWorkStation");
    }
}

}  // namespace jarvis::os

// Design notes:
// - COM is initialised per call because the callers are Python executor
//   threads of unknown apartment state; a per-call scope costs microseconds
//   on an already-initialised thread and leaves the thread as it found it.
// - WMI brightness covers the panels where software brightness exists at
//   all (laptops, tablets). External monitors need DDC/CI, a separate
//   feature with its own failure modes, so it is reported as unsupported.
