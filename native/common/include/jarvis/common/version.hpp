/// @file version.hpp
/// @brief Version information for the native layer.
///
/// The version string is injected by CMake from the project() declaration so
/// that there is exactly one place to bump it. Every pybind11 module exposes
/// it as @c __version__ so the Python side can refuse to run against a stale
/// build.

#pragma once

#include <string_view>

#ifndef JARVIS_NATIVE_VERSION
#error "JARVIS_NATIVE_VERSION must be defined by the build system"
#endif

namespace jarvis::common {

/// Semantic version of the native layer, e.g. "0.1.0".
inline constexpr std::string_view kVersion{JARVIS_NATIVE_VERSION};

}  // namespace jarvis::common
