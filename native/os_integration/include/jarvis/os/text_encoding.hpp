/// @file text_encoding.hpp
/// @brief UTF-8 <-> UTF-16 conversion for the Win32 "W" APIs.
///
/// Python hands us UTF-8 through pybind11; every Win32 call we make takes
/// UTF-16. These two functions are the only place that conversion happens,
/// and they reject malformed input instead of substituting U+FFFD so a bad
/// string fails loudly at the boundary rather than typing garbage.

#pragma once

#include <string>
#include <string_view>

namespace jarvis::os {

/// Convert UTF-8 to UTF-16. Throws OsError on an invalid byte sequence.
[[nodiscard]] std::wstring utf8ToWide(std::string_view utf8);

/// Convert UTF-16 to UTF-8. Throws OsError on an unpaired surrogate.
[[nodiscard]] std::string wideToUtf8(std::wstring_view wide);

}  // namespace jarvis::os
