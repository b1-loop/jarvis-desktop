/// @file win32_util.hpp
/// @brief Private RAII helpers over Win32 and COM. Never included by a public header.
///
/// WIN32_LEAN_AND_MEAN, NOMINMAX and UNICODE are set by CMake for the whole
/// library, so this header can include <windows.h> without local defines.

#pragma once

#include "jarvis/os/errors.hpp"
#include "jarvis/os/window_manager.hpp"

#include <windows.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace jarvis::os::win32 {

/// Text of @p code as FormatMessage renders it, trailing newline removed.
[[nodiscard]] std::string errorMessage(DWORD code);

/// Throw OsError describing GetLastError() for @p apiName.
[[noreturn]] void throwLastError(std::string_view apiName);

/// Throw OsError describing @p result (an HRESULT) for @p apiName.
[[noreturn]] void throwHresult(std::string_view apiName, long result);

/// Throw unless @p result succeeded.
inline void checkHresult(std::string_view apiName, long result) {
    if (result < 0) {
        throwHresult(apiName, result);
    }
}

/// Closes a kernel HANDLE; tolerates the two "no handle" values.
struct HandleCloser {
    void operator()(HANDLE handle) const noexcept {
        if (handle != nullptr && handle != INVALID_HANDLE_VALUE) {
            CloseHandle(handle);
        }
    }
};
using UniqueHandle = std::unique_ptr<void, HandleCloser>;

/// Convert between our opaque WindowHandle and HWND. The two casts are the
/// only int<->pointer conversions in the library; the handle is an integer
/// in the public API precisely so that Python never sees a pointer.
[[nodiscard]] HWND toHwnd(WindowHandle handle) noexcept;
[[nodiscard]] WindowHandle fromHwnd(HWND hwnd) noexcept;

/// Throw WindowNotFoundError unless @p handle names a live window.
[[nodiscard]] HWND requireWindow(WindowHandle handle);

/// Which COM apartment a ComScope joins.
enum class Apartment : std::uint8_t { MultiThreaded, SingleThreaded };

/// CoInitializeEx on construction, CoUninitialize on destruction.
///
/// If the thread already belongs to the other apartment type, COM reports
/// RPC_E_CHANGED_MODE; the thread is usable as it is, so that is accepted
/// and no CoUninitialize is issued. Any other failure throws OsError.
class ComScope {
public:
    explicit ComScope(Apartment apartment);
    ~ComScope();

    ComScope(const ComScope&) = delete;
    ComScope& operator=(const ComScope&) = delete;
    ComScope(ComScope&&) = delete;
    ComScope& operator=(ComScope&&) = delete;

private:
    bool uninitialise_{false};
};

/// Makes the current thread per-monitor DPI aware for the scope's lifetime
/// and restores the previous context afterwards.
class DpiAwarenessScope {
public:
    DpiAwarenessScope();
    ~DpiAwarenessScope();

    DpiAwarenessScope(const DpiAwarenessScope&) = delete;
    DpiAwarenessScope& operator=(const DpiAwarenessScope&) = delete;
    DpiAwarenessScope(DpiAwarenessScope&&) = delete;
    DpiAwarenessScope& operator=(DpiAwarenessScope&&) = delete;

private:
    DPI_AWARENESS_CONTEXT previous_;
};

}  // namespace jarvis::os::win32
