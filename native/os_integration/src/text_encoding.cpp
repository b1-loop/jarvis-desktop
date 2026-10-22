#include "jarvis/os/text_encoding.hpp"

#include "jarvis/os/errors.hpp"

#include <windows.h>

#include <string>
#include <string_view>

namespace jarvis::os {

namespace {

/// The conversion APIs take an int length; refuse anything that would not fit
/// rather than silently truncating.
int checkedLength(std::size_t size) {
    constexpr auto kMax = static_cast<std::size_t>(INT32_MAX);
    if (size > kMax) {
        throw OsError("string too long to convert");
    }
    return static_cast<int>(size);
}

}  // namespace

// The explicit length argument that follows each data() pointer below is what
// makes the non-terminated view safe; clang-tidy cannot see that pairing.
// NOLINTBEGIN(bugprone-suspicious-stringview-data-usage)

std::wstring utf8ToWide(std::string_view utf8) {
    if (utf8.empty()) {
        return {};
    }
    const int length = checkedLength(utf8.size());
    const int needed =
        MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), length, nullptr, 0);
    if (needed <= 0) {
        throw OsError("invalid UTF-8 input");
    }
    std::wstring wide(static_cast<std::size_t>(needed), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), length, wide.data(), needed);
    return wide;
}

std::string wideToUtf8(std::wstring_view wide) {
    if (wide.empty()) {
        return {};
    }
    const int length = checkedLength(wide.size());
    const int needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(), length,
                                           nullptr, 0, nullptr, nullptr);
    if (needed <= 0) {
        throw OsError("invalid UTF-16 input (unpaired surrogate)");
    }
    std::string utf8(static_cast<std::size_t>(needed), '\0');
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(), length, utf8.data(), needed,
                        nullptr, nullptr);
    return utf8;
}

// NOLINTEND(bugprone-suspicious-stringview-data-usage)

}  // namespace jarvis::os
