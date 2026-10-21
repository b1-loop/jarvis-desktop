/// @file module_info.hpp
/// @brief Attributes shared by every Jarvis extension module.

#pragma once

#include "jarvis/common/version.hpp"

#include <pybind11/pybind11.h>

#include <string>
#include <string_view>

namespace jarvis::bindings {

/// Attach the attributes every Jarvis module exposes.
///
/// @param module  The module being defined.
/// @param status  Human-readable implementation status. Scaffolded modules say
///                which step delivers their real API so nobody mistakes an
///                empty module for a finished one.
inline void defineModuleInfo(const pybind11::module_& module, std::string_view status) {
    module.attr("__version__") = std::string{common::kVersion};
    module.attr("IMPLEMENTATION_STATUS") = std::string{status};
}

}  // namespace jarvis::bindings
