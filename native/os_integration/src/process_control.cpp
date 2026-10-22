#include "jarvis/os/process_control.hpp"

#include "jarvis/os/errors.hpp"
#include "jarvis/os/text_encoding.hpp"

#include "win32_util.hpp"
#include <shellapi.h>
#include <tlhelp32.h>
#include <windows.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cwctype>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::os {

namespace {

std::wstring loweredWide(std::wstring_view text) {
    std::wstring result{text};
    std::ranges::transform(result, result.begin(), [](wchar_t c) { return std::towlower(c); });
    return result;
}

/// "code", "Code.exe" and "CODE" all name the same executable.
std::wstring normaliseExecutableName(std::string_view name) {
    std::wstring wide = loweredWide(utf8ToWide(name));
    constexpr std::wstring_view kSuffix = L".exe";
    if (!wide.ends_with(kSuffix)) {
        wide += kSuffix;
    }
    return wide;
}

struct CloseRequest {
    DWORD pid;
    std::size_t windowsClosed;
};

BOOL CALLBACK closeWindowOfProcess(HWND hwnd, LPARAM lParam) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr,cppcoreguidelines-pro-type-reinterpret-cast)
    auto* request = reinterpret_cast<CloseRequest*>(lParam);
    DWORD pid = 0;
    GetWindowThreadProcessId(hwnd, &pid);
    if (pid == request->pid && GetWindow(hwnd, GW_OWNER) == nullptr &&
        IsWindowVisible(hwnd) != FALSE) {
        PostMessageW(hwnd, WM_CLOSE, 0, 0);
        ++request->windowsClosed;
    }
    return TRUE;
}

win32::UniqueHandle openProcess(DWORD pid, DWORD access) {
    win32::UniqueHandle process{OpenProcess(access, FALSE, pid)};
    if (process == nullptr) {
        const DWORD code = GetLastError();
        if (code == ERROR_INVALID_PARAMETER) {
            throw ProcessNotFoundError("no process with pid " + std::to_string(pid));
        }
        throw OsError("OpenProcess(" + std::to_string(pid) +
                      ") failed: " + win32::errorMessage(code));
    }
    return process;
}

/// Throw ProcessNotFoundError unless @p pid is running.
void requireProcess(DWORD pid) {
    const win32::UniqueHandle probe = openProcess(pid, PROCESS_QUERY_LIMITED_INFORMATION);
}

void closeGracefully(DWORD pid) {
    CloseRequest request{.pid = pid, .windowsClosed = 0};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    EnumWindows(&closeWindowOfProcess, reinterpret_cast<LPARAM>(&request));
    if (request.windowsClosed == 0) {
        throw OsError("process " + std::to_string(pid) +
                      " has no top-level window to close; use force to terminate it");
    }
}

void terminate(DWORD pid) {
    const win32::UniqueHandle process = openProcess(pid, PROCESS_TERMINATE);
    if (TerminateProcess(process.get(), 1) == FALSE) {
        win32::throwLastError("TerminateProcess(" + std::to_string(pid) + ")");
    }
}

/// Pids of every running process whose image name equals @p wantedLower.
std::vector<DWORD> findProcesses(const std::wstring& wantedLower) {
    const win32::UniqueHandle snapshot{CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)};
    if (snapshot.get() == INVALID_HANDLE_VALUE) {
        win32::throwLastError("CreateToolhelp32Snapshot");
    }
    std::vector<DWORD> pids;
    PROCESSENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    for (BOOL more = Process32FirstW(snapshot.get(), &entry); more != FALSE;
         more = Process32NextW(snapshot.get(), &entry)) {
        if (loweredWide(std::data(entry.szExeFile)) == wantedLower) {
            pids.push_back(entry.th32ProcessID);
        }
    }
    return pids;
}

}  // namespace

std::uint32_t Win32ProcessControl::launch(std::string_view target, std::string_view arguments) {
    // ShellExecuteEx may hand off to shell extensions that need COM.
    const win32::ComScope com{win32::Apartment::SingleThreaded};
    const std::wstring wideTarget = utf8ToWide(target);
    const std::wstring wideArguments = utf8ToWide(arguments);

    constexpr auto kKeepProcessHandle = static_cast<ULONG>(SEE_MASK_NOCLOSEPROCESS);
    constexpr auto kSynchronous = static_cast<ULONG>(SEE_MASK_NOASYNC);
    constexpr auto kNoErrorDialog = static_cast<ULONG>(SEE_MASK_FLAG_NO_UI);
    SHELLEXECUTEINFOW info{};
    info.cbSize = sizeof(info);
    info.fMask = kKeepProcessHandle | kSynchronous | kNoErrorDialog;
    info.lpVerb = L"open";
    info.lpFile = wideTarget.c_str();
    info.lpParameters = wideArguments.empty() ? nullptr : wideArguments.c_str();
    info.nShow = SW_SHOWNORMAL;
    if (ShellExecuteExW(&info) == FALSE) {
        win32::throwLastError("ShellExecuteEx(" + std::string{target} + ")");
    }
    const win32::UniqueHandle process{info.hProcess};
    return process == nullptr ? 0 : GetProcessId(process.get());
}

void Win32ProcessControl::close(std::uint32_t pid, bool force) {
    requireProcess(pid);
    if (force) {
        terminate(pid);
    } else {
        closeGracefully(pid);
    }
}

std::size_t Win32ProcessControl::closeByName(std::string_view executableName, bool force) {
    const std::vector<DWORD> pids = findProcesses(normaliseExecutableName(executableName));
    if (pids.empty()) {
        throw ProcessNotFoundError("no running process named '" + std::string{executableName} +
                                   "'");
    }
    for (const DWORD pid : pids) {
        close(pid, force);
    }
    return pids.size();
}

}  // namespace jarvis::os

// Design notes:
// - ShellExecuteEx rather than CreateProcess so "notepad", a .docx or a URL
//   all work the way a user typing into Win+R expects; the price is that a
//   document handed to an existing instance yields no pid.
// - Graceful close raises when the process has no window instead of
//   escalating to TerminateProcess: escalation is a separate, confirmable
//   decision for the agent, never an implicit fallback.
