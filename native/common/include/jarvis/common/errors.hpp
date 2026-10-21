/// @file errors.hpp
/// @brief Exception hierarchy for the native layer.
///
/// Every exception thrown on purpose by Jarvis C++ code derives from
/// NativeError so the pybind11 modules can map the whole family onto one
/// Python exception class per module, mirroring jarvis.utils.errors.

#pragma once

#include <stdexcept>

namespace jarvis {

/// Base class for all errors raised by the native layer.
class NativeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// An audio device could not be opened, started or configured.
class AudioDeviceError : public NativeError {
public:
    using NativeError::NativeError;
};

/// A model file is missing, malformed or does not match the expected API.
class ModelError : public NativeError {
public:
    using NativeError::NativeError;
};

}  // namespace jarvis
