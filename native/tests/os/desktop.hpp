/// @file desktop.hpp
/// @brief Gates for OS tests: an interactive, unlocked desktop, and permission to change state.

#pragma once

#include <catch2/catch_test_macros.hpp>

#include <windows.h>
#include <wtsapi32.h>

#include <array>
#include <string_view>

namespace jarvis::os::testing {

/// Value of environment variable @p name, or empty when it is not set.
inline std::wstring_view environmentVariable(const wchar_t* name, std::array<wchar_t, 64>& buffer) {
    const DWORD length =
        GetEnvironmentVariableW(name, buffer.data(), static_cast<DWORD>(buffer.size()));
    return length == 0 || length >= buffer.size() ? std::wstring_view{}
                                                  : std::wstring_view{buffer.data(), length};
}

/// True when Windows reports this process's session as locked.
///
/// A locked workstation keeps the user's desktop alive but denies clipboard
/// and window calls, and OpenInputDesktop alone does not reveal it.
inline bool sessionLocked() {
    LPWSTR buffer = nullptr;
    DWORD bytes = 0;
    if (WTSQuerySessionInformationW(WTS_CURRENT_SERVER_HANDLE, WTS_CURRENT_SESSION,
                                    WTSSessionInfoEx, &buffer, &bytes) == FALSE) {
        return false;  // unknown: let the test run and report what it finds
    }
    const auto* info = reinterpret_cast<const WTSINFOEXW*>(buffer);  // NOLINT(*-reinterpret-cast)
    const bool locked =
        info->Level == 1 && info->Data.WTSInfoExLevel1.SessionFlags == WTS_SESSIONSTATE_LOCK;
    WTSFreeMemory(buffer);
    return locked;
}

/// Skip unless the process can use an interactive, unlocked desktop (CI
/// runners have one; a service session, a headless container or a locked
/// workstation do not).
inline void requireDesktop() {
    const HDESK desktop = OpenInputDesktop(0, FALSE, GENERIC_READ);
    if (desktop == nullptr) {
        SKIP("no interactive desktop session");
    }
    CloseDesktop(desktop);
    if (sessionLocked()) {
        SKIP("the workstation is locked");
    }
}

/// Skip unless the developer opted in to tests that change machine state
/// (volume, clipboard, notifications). Never set in CI.
inline void requireStateChangesAllowed() {
    std::array<wchar_t, 64> buffer{};
    if (environmentVariable(L"JARVIS_NATIVE_STATE_TESTS", buffer) != L"1") {
        SKIP("set JARVIS_NATIVE_STATE_TESTS=1 to run tests that change machine state");
    }
}

}  // namespace jarvis::os::testing
