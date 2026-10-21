#include "win32_util.hpp"

#include "jarvis/os/errors.hpp"
#include "jarvis/os/text_encoding.hpp"

#include <objbase.h>
#include <windows.h>

#include <cstdint>
#include <format>
#include <memory>
#include <string>
#include <string_view>

namespace jarvis::os::win32 {

namespace {

/// FormatMessage allocates the buffer; LocalFree releases it.
struct LocalFreeDeleter {
    void operator()(wchar_t* buffer) const noexcept { LocalFree(buffer); }
};

std::string hex(std::uint32_t value) {
    return std::format("0x{:08X}", value);
}

/// Text of @p code from the system message table, owned by the returned
/// pointer; nullptr if the code has no message.
std::unique_ptr<wchar_t, LocalFreeDeleter> systemMessage(DWORD code, DWORD& length) {
    constexpr auto kAllocate = static_cast<DWORD>(FORMAT_MESSAGE_ALLOCATE_BUFFER);
    constexpr auto kFromSystem = static_cast<DWORD>(FORMAT_MESSAGE_FROM_SYSTEM);
    constexpr auto kIgnoreInserts = static_cast<DWORD>(FORMAT_MESSAGE_IGNORE_INSERTS);
    wchar_t* raw = nullptr;
    // With ALLOCATE_BUFFER, FormatMessage writes the buffer pointer through
    // the LPWSTR argument, which must therefore point at our pointer.
    auto* const target = reinterpret_cast<LPWSTR>(&raw);  // NOLINT(*-reinterpret-cast)
    length = FormatMessageW(kAllocate | kFromSystem | kIgnoreInserts, nullptr, code, 0, target, 0,
                            nullptr);
    return std::unique_ptr<wchar_t, LocalFreeDeleter>{raw};
}

}  // namespace

std::string errorMessage(DWORD code) {
    DWORD length = 0;
    const auto buffer = systemMessage(code, length);
    if (length == 0 || buffer == nullptr) {
        return "error " + hex(code);
    }
    std::wstring_view text{buffer.get(), length};
    while (!text.empty() && (text.back() == L'\r' || text.back() == L'\n' || text.back() == L'.')) {
        text.remove_suffix(1);
    }
    return wideToUtf8(text) + " (" + hex(code) + ")";
}

void throwLastError(std::string_view apiName) {
    const DWORD code = GetLastError();
    throw OsError(std::string{apiName} + " failed: " + errorMessage(code));
}

void throwHresult(std::string_view apiName, long result) {
    throw OsError(std::string{apiName} + " failed: " +
                  errorMessage(static_cast<DWORD>(static_cast<std::uint32_t>(result))));
}

HWND toHwnd(WindowHandle handle) noexcept {
    // NOLINTNEXTLINE(performance-no-int-to-ptr,cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<HWND>(static_cast<std::uintptr_t>(handle));
}

WindowHandle fromHwnd(HWND hwnd) noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return static_cast<WindowHandle>(reinterpret_cast<std::uintptr_t>(hwnd));
}

HWND requireWindow(WindowHandle handle) {
    HWND hwnd = toHwnd(handle);
    if (hwnd == nullptr || IsWindow(hwnd) == FALSE) {
        throw WindowNotFoundError("window " + hex(static_cast<std::uint32_t>(handle)) +
                                  " no longer exists");
    }
    return hwnd;
}

ComScope::ComScope(Apartment apartment) {
    constexpr auto kMulti = static_cast<DWORD>(COINIT_MULTITHREADED);
    constexpr auto kSingle = static_cast<DWORD>(COINIT_APARTMENTTHREADED);
    constexpr auto kNoOle1 = static_cast<DWORD>(COINIT_DISABLE_OLE1DDE);
    const DWORD model = apartment == Apartment::MultiThreaded ? kMulti : (kSingle | kNoOle1);
    const HRESULT result = CoInitializeEx(nullptr, model);
    if (result == RPC_E_CHANGED_MODE) {
        return;  // the thread is already in the other apartment; that works too
    }
    checkHresult("CoInitializeEx", result);
    uninitialise_ = true;
}

ComScope::~ComScope() {
    if (uninitialise_) {
        CoUninitialize();
    }
}

DpiAwarenessScope::DpiAwarenessScope()
    : previous_(SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {}

DpiAwarenessScope::~DpiAwarenessScope() {
    if (previous_ != nullptr) {
        SetThreadDpiAwarenessContext(previous_);
    }
}

}  // namespace jarvis::os::win32
