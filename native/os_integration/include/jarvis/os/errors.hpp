/// @file errors.hpp
/// @brief Exceptions raised by the OS integration layer.
///
/// All derive from jarvis::NativeError so the jarvis_os Python module maps
/// them onto one hierarchy; the subclasses exist so a tool can tell "that
/// window is gone" from "this machine cannot do that" without parsing text.

#pragma once

#include "jarvis/common/errors.hpp"

namespace jarvis::os {

/// A Win32, COM or WMI call failed. The message carries the API name and the
/// system error text.
class OsError : public NativeError {
public:
    using NativeError::NativeError;
};

/// The capability exists in the API but not on this machine, for example
/// brightness control on a desktop with external monitors only.
class UnsupportedError : public OsError {
public:
    using OsError::OsError;
};

/// A window handle is stale or no window matched a title search.
class WindowNotFoundError : public OsError {
public:
    using OsError::OsError;
};

/// A process id is gone or no running process matched a name.
class ProcessNotFoundError : public OsError {
public:
    using OsError::OsError;
};

}  // namespace jarvis::os
