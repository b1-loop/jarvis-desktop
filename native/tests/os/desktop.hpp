/// @file desktop.hpp
/// @brief Gates for OS tests: an interactive desktop, and permission to change state.

#pragma once

#include <catch2/catch_test_macros.hpp>

#include <windows.h>

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

/// Skip unless the process can see an interactive desktop (CI runners
/// have one; a service session or a headless container does not).
/// OpenInputDesktop fails in session 0 and on a locked secure desktop.
inline void requireDesktop() {
    const HDESK desktop = OpenInputDesktop(0, FALSE, GENERIC_READ);
    if (desktop == nullptr) {
        SKIP("no interactive desktop session");
    }
    CloseDesktop(desktop);
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
