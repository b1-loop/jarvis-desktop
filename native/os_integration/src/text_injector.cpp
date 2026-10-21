#include "jarvis/os/text_injector.hpp"

#include "jarvis/os/errors.hpp"
#include "jarvis/os/text_encoding.hpp"

#include "win32_util.hpp"
#include <windows.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::os {

namespace {

constexpr auto kUnicode = static_cast<DWORD>(KEYEVENTF_UNICODE);
constexpr auto kKeyUp = static_cast<DWORD>(KEYEVENTF_KEYUP);

// INPUT is a tagged union by Win32 design; the .ki accesses below are the
// only way to fill it. NOLINT reason: cppcoreguidelines-pro-type-union-access.

INPUT unicodeKey(wchar_t unit, bool keyUp) {
    INPUT input{};
    input.type = INPUT_KEYBOARD;
    input.ki.wScan = unit;                                      // NOLINT
    input.ki.dwFlags = kUnicode | (keyUp ? kKeyUp : DWORD{0});  // NOLINT
    return input;
}

INPUT virtualKey(WORD code, bool keyUp) {
    INPUT input{};
    input.type = INPUT_KEYBOARD;
    input.ki.wVk = code;                           // NOLINT
    input.ki.dwFlags = keyUp ? kKeyUp : DWORD{0};  // NOLINT
    return input;
}

void appendEnter(std::vector<INPUT>& inputs) {
    inputs.push_back(virtualKey(VK_RETURN, false));
    inputs.push_back(virtualKey(VK_RETURN, true));
}

/// '\n' becomes an Enter key press because most controls ignore a Unicode
/// line feed; '\r' is dropped so Windows line endings do not press twice.
void appendUnit(std::vector<INPUT>& inputs, wchar_t unit) {
    if (unit == L'\n') {
        appendEnter(inputs);
    } else if (unit != L'\r') {
        inputs.push_back(unicodeKey(unit, false));
        inputs.push_back(unicodeKey(unit, true));
    }
}

}  // namespace

void SendInputTextInjector::typeText(std::string_view utf8, bool pressEnter) {
    std::vector<INPUT> inputs;
    for (const wchar_t unit : utf8ToWide(utf8)) {
        appendUnit(inputs, unit);
    }
    if (pressEnter) {
        appendEnter(inputs);
    }
    if (inputs.empty()) {
        return;
    }
    const auto count = static_cast<UINT>(inputs.size());
    if (SendInput(count, inputs.data(), sizeof(INPUT)) != count) {
        win32::throwLastError("SendInput");
    }
}

}  // namespace jarvis::os

// Design notes:
// - KEYEVENTF_UNICODE sends code units, not virtual keys, so the text lands
//   verbatim regardless of keyboard layout; a surrogate pair is two units
//   and arrives as one character.
// - The whole string goes in one SendInput call so no other input can
//   interleave with it. A UI-automation blocked window (UIPI, secure
//   desktop) makes SendInput report fewer events, which surfaces as OsError.
